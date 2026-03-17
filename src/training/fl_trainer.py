"""Federated Learning Trainer for Medical Image Segmentation.

Simulates N federated clients on a single machine. Each FL round:
  1. Server broadcasts the global model to selected clients.
  2. Each client trains locally for `local_iters` iterations.
  3. Server aggregates client updates via FedAvg (or FedProx).
  4. Server evaluates the global model on the validation set.

Data partitioning supports:
  - IID: uniform random split across clients.
  - Non-IID: Dirichlet distribution (controlled by `dirichlet_alpha`).
"""

import os
import random
import logging
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import json

import numpy as np
import torch
import torch.nn.functional as N
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from rich.logging import RichHandler
from rich.console import Console
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .al_trainer import ALConfig

from datasets import (
    ACDCDataset,
    ActiveDataset,
    ExtendableDataset,
    TN3KDataset,
    TG3KDataset,
    FUGCDataset,
    BUSIDataset,
)
from losses.compound_losses import DiceAndCELoss
from losses.dice_loss import DiceLoss
from scheduler.lr_scheduler import PolyLRScheduler
from models.unet import UNet, UnetProcessor
from metric import cal_hd
from transforms.normalization import ZScoreNormalize
from transforms.image_transform import (
    RandomGamma,
    RandomContrast,
    RandomBrightness,
    RandomGaussianBlur,
    RandomGaussianNoise,
    SimulateLowRes,
)
from transforms.joint_transform import (
    JointResize,
    RandomRotation,
    RandomAffine,
    RandomCrop2D,
    MirrorTransform,
    RandomRotation90,
)
from transforms.common import (
    RandomTransform,
    ComposeTransform,
    RandomChoiceTransform,
)
from utils import get_path, draw_mask, dummy_context
from federated import FedAvgAggregator, FedProxAggregator, FedNovaAggregator, FedPerAggregator


def _worker_init_fn(worker_id):
    seed = int(os.environ.get("FL_SEED", 0)) + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class FLConfig(ALConfig):
    """Extends ALConfig with federated learning parameters."""

    def __init__(
        self,
        # FL-specific parameters
        num_clients: int = 5,
        num_fl_rounds: int = 10,
        local_iters: int = 200,
        client_fraction: float = 1.0,
        aggregation: Literal["fedavg", "fedprox", "fednova", "fedper"] = "fedavg",
        fedprox_mu: float = 0.01,
        dirichlet_alpha: float | None = None,
        fedper_shared_prefix: str = "encoder.",
        # Inherited parameters
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_clients = num_clients
        self.num_fl_rounds = num_fl_rounds
        self.local_iters = local_iters
        self.client_fraction = client_fraction
        self.aggregation = aggregation
        self.fedprox_mu = fedprox_mu
        # None means IID partition; float > 0 means non-IID via Dirichlet
        self.dirichlet_alpha = dirichlet_alpha
        self.fedper_shared_prefix = fedper_shared_prefix


class FLTrainer(BaseTrainer):
    """Federated Learning trainer that simulates multiple clients locally."""

    def __init__(
        self,
        work_path: Path | str = Path.cwd(),
        deterministic: bool = True,
        device: torch.device | str = torch.device("cuda"),
        config: FLConfig | dict | None = None,
        # Log parameters
        verbose: bool = True,
        log_path: Path | str | None = None,
        config_path: Path | str | None = None,
        log_mode: str = "a",
        log_override: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        **kwargs,
    ):
        if isinstance(config, FLConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = FLConfig(**config)
        else:
            self.config = FLConfig()

        self.deterministic = deterministic
        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.work_path = get_path(work_path)
        self.device = torch.device("cpu")
        self.to(device)

        self._set_seed(self.config.seed)

        self.verbose = verbose
        self.log_path = log_path
        self.config_path = config_path
        self.log_mode = log_mode
        self.log_override = log_override
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key

        self.current_fl_round = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        self._set_snapshot_work_dir()
        self._setup_logger()
        self._build_model()
        self.model.to(self.device)

    def _set_seed(self, seed: int):
        os.environ["FL_SEED"] = str(seed)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _set_snapshot_work_dir(self):
        current_time_str = datetime.now().strftime("%Y%m%d_%H")
        parts = [
            self.config.dataset,
            current_time_str,
            f"fl-clients-{self.config.num_clients}",
            f"fl-rounds-{self.config.num_fl_rounds}",
            f"local-iters-{self.config.local_iters}",
            f"agg-{self.config.aggregation}",
            f"alpha-{self.config.dirichlet_alpha}",
            f"imgsz-{self.config.image_size}",
            f"batchsz-{self.config.batch_size}",
            f"optimizer-{self.config.optimizer_name}",
            f"startlr-{self.config.start_lr}",
        ]
        if self.config.exp_name:
            parts.append(self.config.exp_name)
        self.work_path = self.work_path / "_".join(parts)
        self.work_path.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self):
        self.logger = logging.getLogger("MIA.FLTrainer")
        self.logger.setLevel(logging.DEBUG)

        if not self.log_path:
            self.log_path = self.work_path / "log.txt"
        self.log_path = get_path(self.log_path)

        if self.log_path.exists() and not self.log_override:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = (
                self.log_path.parent
                / f"{self.log_path.stem}@{ts}{self.log_path.suffix}"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(self.log_path, self.log_mode)
        fh.setFormatter(logging.Formatter("%(levelname)s <%(asctime)s>: %(message)s"))
        self.logger.addHandler(fh)

        if self.verbose:
            sh = RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                show_time=False,
                show_path=False,
                show_level=False,
                keywords=["FL Round", "Client", "Aggregation", "Valid", "Global"],
            )
            sh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(sh)

    def _build_model(self):
        self.model = UNet(
            dimension=2,
            input_channels=self.config.in_channels,
            output_classes=self.config.num_classes + 1,
            channels_list=[32, 64, 128, 256, 512],
            deep_supervision=self.config.deep_supervision,
            ds_layer=self.config.ds_layer,
            block_type=self.config.block_type,
            dropout_prob=self.config.dropout_prob,
            normalization=self.config.block_normalization,
        )
        self.model_processor = UnetProcessor(image_size=self.config.image_size)

        if self.config.model_ckpt:
            self._load_model_checkpoint(self.config.model_ckpt)

    def _load_model_checkpoint(self, ckpt: str | Path):
        try:
            state_dict = torch.load(ckpt, map_location=self.device)
            if "model" in state_dict:
                self.model.load_state_dict(state_dict["model"])
            else:
                self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded checkpoint from {ckpt}")
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint from {ckpt}: {e}")

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _get_dataset_cls(self):
        mapping = {
            "ACDC": ACDCDataset,
            "tn3k": TN3KDataset,
            "tg3k": TG3KDataset,
            "fugc": FUGCDataset,
            "busi": BUSIDataset,
        }
        if self.config.dataset not in mapping:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        return mapping[self.config.dataset]

    def _get_train_transform(self):
        transforms = []
        if self.config.do_augment:
            if self.config.dataset in ["fugc", "busi"]:
                transforms += [
                    RandomTransform(RandomAffine(scale=(0.7, 1.4)), p=0.2),
                    RandomTransform(RandomAffine(degrees=(-15, 15)), p=0.2),
                    RandomTransform(RandomGaussianNoise(sigma=(0, 0.1)), p=0.1),
                    RandomTransform(RandomGaussianBlur(sigma=(0.5, 1)), p=0.2),
                    RandomTransform(RandomBrightness(brightness=0.25), p=0.15),
                    RandomTransform(RandomContrast(contrast=0.25), p=0.15),
                    RandomTransform(SimulateLowRes(scale=(0.5, 1)), p=0.15),
                    RandomTransform(RandomGamma(gamma=(0.7, 1.5)), p=0.1),
                ]
            else:
                transforms += [
                    RandomTransform(
                        ComposeTransform([
                            RandomRotation90(),
                            RandomChoiceTransform([
                                MirrorTransform((-2)),
                                MirrorTransform((-1)),
                            ]),
                        ]),
                        p=0.5,
                    ),
                    RandomTransform(RandomAffine(degrees=(-20, 20)), p=0.5),
                ]
        return ComposeTransform(transforms)

    def _get_train_normalize(self):
        return ZScoreNormalize() if self.config.do_normalize else None

    def _get_valid_transform(self):
        return ComposeTransform([])

    def _get_valid_normalize(self):
        return ZScoreNormalize() if self.config.do_normalize else None

    def _get_full_train_dataset(self):
        cls = self._get_dataset_cls()
        return cls(
            data_path=self.config.data_path,
            split="train",
            normalize=self._get_train_normalize(),
            transform=self._get_train_transform(),
            logger=self.logger,
            image_channels=self.config.in_channels,
            image_size=self.config.image_size,
        )

    def _get_valid_dataset(self):
        cls = self._get_dataset_cls()
        return cls(
            data_path=self.config.data_path,
            split="valid",
            normalize=self._get_valid_normalize(),
            transform=self._get_valid_transform(),
            logger=self.logger,
            image_channels=self.config.in_channels,
        )

    def _partition_iid(self, n_samples: int, n_clients: int) -> list[list[int]]:
        """Randomly shuffle indices and split evenly across clients."""
        indices = list(range(n_samples))
        random.shuffle(indices)
        splits = np.array_split(indices, n_clients)
        return [list(s) for s in splits]

    def _partition_dirichlet(
        self, dataset, n_clients: int, alpha: float
    ) -> list[list[int]]:
        """Non-IID split via Dirichlet distribution over class labels.

        Samples are assigned to clients based on a Dirichlet draw over
        label proportions. If labels are not available (e.g., 3-D volumes),
        falls back to IID.
        """
        n_samples = len(dataset)

        # Try to collect per-sample labels for class-aware partitioning
        labels = []
        for i in range(n_samples):
            try:
                sample = dataset.get_sample(i, normalize=False)
                label = sample.get("label", None)
                if label is None:
                    raise KeyError
                # Use the most frequent non-background class as proxy
                lbl_tensor = torch.as_tensor(label).long()
                counts = torch.bincount(lbl_tensor.flatten(), minlength=2)
                labels.append(int(counts[1:].argmax().item() + 1))
            except Exception:
                labels.append(0)

        unique_classes = list(set(labels))
        client_indices: list[list[int]] = [[] for _ in range(n_clients)]

        for cls in unique_classes:
            cls_idx = [i for i, l in enumerate(labels) if l == cls]
            proportions = np.random.dirichlet([alpha] * n_clients)
            proportions = (proportions * len(cls_idx)).astype(int)
            # Fix rounding so we don't drop samples
            diff = len(cls_idx) - proportions.sum()
            proportions[np.argmax(proportions)] += diff

            start = 0
            for c, count in enumerate(proportions):
                client_indices[c].extend(cls_idx[start: start + count])
                start += count

        # Shuffle within each client
        for c in range(n_clients):
            random.shuffle(client_indices[c])

        return client_indices

    # ------------------------------------------------------------------
    # Optimizer & loss
    # ------------------------------------------------------------------

    def _build_optimizer(self, model: nn.Module):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if self.config.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                parameters, betas=(0.9, 0.999), **self.config.optimizer_kwargs
            )
        elif self.config.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                parameters, betas=(0.9, 0.999), **self.config.optimizer_kwargs
            )
        elif self.config.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                parameters, momentum=0.9, **self.config.optimizer_kwargs
            )
        else:
            raise ValueError(f"Optimizer '{self.config.optimizer_name}' not supported")

        if self.config.lr_scheduler_name == "poly":
            lr_scheduler = PolyLRScheduler(
                optimizer,
                initial_lr=self.config.start_lr,
                max_steps=self.config.local_iters,
                warmup_steps=self.config.lr_warmup_iter,
                interval=self.config.lr_interval,
            )
        elif self.config.lr_scheduler_name == "none":
            lr_scheduler = None
        else:
            raise ValueError(
                f"LR scheduler '{self.config.lr_scheduler_name}' not supported"
            )

        return optimizer, lr_scheduler

    def _build_loss(self):
        return DiceAndCELoss(
            dice_loss=DiceLoss,
            dice_kwargs={
                "num_classes": self.config.num_classes,
                "smooth": 1e-5,
                "do_bg": True,
                "softmax": True,
                "batch": False,
                "squared": False,
            },
            ce_loss=torch.nn.CrossEntropyLoss,
            ce_kwargs={},
            default_dice_weight=self.config.dice_weight,
            default_ce_weight=self.config.ce_weight,
        )

    def _build_aggregator(self):
        if self.config.aggregation == "fedavg":
            return FedAvgAggregator()
        elif self.config.aggregation == "fedprox":
            return FedProxAggregator(mu=self.config.fedprox_mu)
        elif self.config.aggregation == "fednova":
            return FedNovaAggregator()
        elif self.config.aggregation == "fedper":
            return FedPerAggregator(shared_prefix=self.config.fedper_shared_prefix)
        else:
            raise ValueError(f"Aggregation '{self.config.aggregation}' not supported")

    # ------------------------------------------------------------------
    # Local client training
    # ------------------------------------------------------------------

    def _local_train(
        self,
        client_id: int,
        local_model: nn.Module,
        client_dataset,
        global_model: nn.Module | None = None,
    ) -> dict[str, Any]:
        """Train a single client model for `local_iters` iterations.

        Returns a dict with the trained model and training stats.
        When using FedProx, `global_model` is used to compute the proximal term.
        """
        local_model = local_model.to(self.device)
        local_model.train()

        optimizer, lr_scheduler = self._build_optimizer(local_model)
        loss_fn = self._build_loss()

        use_proximal = (
            self.config.aggregation == "fedprox"
            and global_model is not None
        )
        mu = self.config.fedprox_mu if use_proximal else 0.0

        if use_proximal:
            global_params = [
                p.detach().clone() for p in global_model.parameters()
            ]

        dataloader = DataLoader(
            dataset=client_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            worker_init_fn=_worker_init_fn,
        )
        data_iter = iter(dataloader)

        total_loss = 0.0
        n_iters = 0

        pbar = tqdm(
            range(self.config.local_iters),
            desc=f"  Client {client_id}",
            leave=False,
        )
        for step in pbar:
            if lr_scheduler is not None:
                lr_scheduler.step(step)

            # Cycle through dataloader
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            with (
                torch.autocast(self.device.type)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                output = local_model(images)
                loss: torch.Tensor = loss_fn(output, labels)

                if use_proximal:
                    prox = sum(
                        ((p - g) ** 2).sum()
                        for p, g in zip(local_model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                local_model.parameters(), self.config.grad_norm
            )
            optimizer.step()

            total_loss += loss.item()
            n_iters += 1
            pbar.set_postfix({"loss": f"{total_loss / n_iters:.4f}"})

        avg_loss = total_loss / max(n_iters, 1)
        self.logger.info(
            f"  Client {client_id}: avg_loss={avg_loss:.4f} "
            f"over {n_iters} iters, data_size={len(client_dataset)}"
        )

        return {
            "model": local_model.cpu(),
            "data_size": len(client_dataset),
            "avg_loss": avg_loss,
            "actual_steps": n_iters,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, model: nn.Module) -> dict[str, float]:
        model.eval()
        model.to(self.device)

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config.valid_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
        )

        loss_fn = self._build_loss()
        total_loss = 0.0
        all_dice = []
        all_hd = []

        pbar = tqdm(valid_dataloader, desc="  Valid", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with (
                torch.autocast(self.device.type)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                output = model(images)
                loss = loss_fn(output, labels)

            total_loss += loss.item()

            # Dice per batch
            pred = output.argmax(dim=1)
            for cls in range(1, self.config.num_classes + 1):
                pred_cls = (pred == cls).float()
                gt_cls = (labels[:, 0] == cls).float()
                inter = (pred_cls * gt_cls).sum()
                union = pred_cls.sum() + gt_cls.sum()
                dice = (2 * inter / (union + 1e-5)).item()
                all_dice.append(dice)

            # HD (best-effort)
            try:
                pred_np = pred[0].cpu().numpy().astype(np.int32)
                gt_np = labels[0, 0].cpu().numpy().astype(np.int32)
                hd_val = cal_hd(pred_np, gt_np)
                if np.isfinite(hd_val):
                    all_hd.append(float(hd_val))
            except Exception:
                pass

        avg_loss = total_loss / max(len(valid_dataloader), 1)
        avg_dice = float(np.mean(all_dice)) if all_dice else 0.0
        avg_hd = float(np.mean(all_hd)) if all_hd else float("inf")

        return {"loss": avg_loss, "dice": avg_dice, "hd": avg_hd}

    # ------------------------------------------------------------------
    # FL round
    # ------------------------------------------------------------------

    def _select_clients(self) -> list[int]:
        n = max(1, int(self.config.num_clients * self.config.client_fraction))
        return random.sample(range(self.config.num_clients), n)

    def _fl_round(self, fl_round: int):
        if self.config.aggregation == "fedper":
            return self._fl_round_fedper(fl_round)

        self.logger.info(f"FL Round {fl_round + 1}/{self.config.num_fl_rounds}")

        selected = self._select_clients()
        self.logger.info(f"  Selected clients: {selected}")

        client_results = []
        for cid in selected:
            client_dataset = self.client_datasets[cid]
            if len(client_dataset) == 0:
                self.logger.warning(f"  Client {cid} has no data — skipping")
                continue

            local_model = deepcopy(self.model)
            result = self._local_train(
                client_id=cid,
                local_model=local_model,
                client_dataset=client_dataset,
                global_model=self.model if self.config.aggregation == "fedprox" else None,
            )
            client_results.append(result)

        if not client_results:
            self.logger.warning("No client results to aggregate — skipping round")
            return

        # Aggregate
        self.logger.info(f"  Aggregation: {self.config.aggregation}")
        client_models = [r["model"] for r in client_results]
        client_weights = [r["data_size"] for r in client_results]

        if isinstance(self.aggregator, FedNovaAggregator):
            client_local_steps = [r["actual_steps"] for r in client_results]
            self.aggregator.aggregate(
                self.model, client_models, client_weights,
                client_local_steps=client_local_steps,
            )
        else:
            self.aggregator.aggregate(self.model, client_models, client_weights)

        # Log client losses
        avg_client_loss = np.mean([r["avg_loss"] for r in client_results])
        self.logger.info(f"  Mean client loss: {avg_client_loss:.4f}")

    def _fl_round_fedper(self, fl_round: int):
        """FedPer round: aggregate only encoder; restore per-client decoders."""
        self.logger.info(f"FL Round {fl_round + 1}/{self.config.num_fl_rounds} [FedPer]")

        selected = self._select_clients()
        self.logger.info(f"  Selected clients: {selected}")

        client_results = []
        for cid in selected:
            client_dataset = self.client_datasets[cid]
            if len(client_dataset) == 0:
                self.logger.warning(f"  Client {cid} has no data — skipping")
                continue

            # Build client model: shared global encoder + client's own decoder
            local_model = deepcopy(self.model)
            if self._client_decoders[cid] is not None:
                state = local_model.state_dict()
                for key, val in self._client_decoders[cid].items():
                    state[key] = val.clone()
                local_model.load_state_dict(state)

            result = self._local_train(
                client_id=cid,
                local_model=local_model,
                client_dataset=client_dataset,
                global_model=None,
            )

            # Save updated decoder weights back to client store
            trained_state = result["model"].state_dict()
            self._client_decoders[cid] = {
                k: v.clone()
                for k, v in trained_state.items()
                if not k.startswith(self.config.fedper_shared_prefix)
            }

            client_results.append(result)

        if not client_results:
            self.logger.warning("No client results to aggregate — skipping round")
            return

        # Aggregate only encoder parameters
        self.logger.info("  Aggregation: fedper (encoder only)")
        client_models = [r["model"] for r in client_results]
        client_weights = [r["data_size"] for r in client_results]
        self.aggregator.aggregate(self.model, client_models, client_weights)

        avg_client_loss = np.mean([r["avg_loss"] for r in client_results])
        self.logger.info(f"  Mean client loss: {avg_client_loss:.4f}")

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self):
        self._setup_loss = self._build_loss
        self.aggregator = self._build_aggregator()

        # Build full datasets
        self.logger.info("Partitioning data across clients …")
        full_train_dataset = self._get_full_train_dataset()
        self.valid_dataset = self._get_valid_dataset()

        n_samples = len(full_train_dataset)
        if self.config.dirichlet_alpha is not None:
            self.logger.info(
                f"Non-IID partition: Dirichlet α={self.config.dirichlet_alpha}"
            )
            client_indices = self._partition_dirichlet(
                full_train_dataset,
                self.config.num_clients,
                self.config.dirichlet_alpha,
            )
        else:
            self.logger.info("IID partition: uniform random split")
            client_indices = self._partition_iid(n_samples, self.config.num_clients)

        self.client_datasets = [
            Subset(full_train_dataset, idx) for idx in client_indices
        ]

        # Save partition info
        partition_info = {str(i): [int(x) for x in idxs] for i, idxs in enumerate(client_indices)}
        partition_path = self.work_path / "client_partition.json"
        with open(partition_path, "w") as f:
            json.dump(partition_info, f, indent=2)

        sizes = [len(ds) for ds in self.client_datasets]
        self.logger.info(f"Client data sizes: {sizes}")

        self._best_valid_metric = None
        self.current_fl_round = 0

        # FedPer: store per-client decoder weights (None until first local train)
        self._client_decoders: dict[int, dict | None] = {
            i: None for i in range(self.config.num_clients)
        }

    def on_train_end(self):
        ckpt_path = self.work_path / "global_model_final.pth"
        torch.save({"model": self.model.state_dict()}, ckpt_path)
        self.logger.info(f"Saved final global model to {ckpt_path}")

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_start(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def train_step(self, *args, **kwargs):
        pass

    def valid_step(self, *args, **kwargs):
        pass

    def train(self):
        self.on_train_start()

        for fl_round in range(self.current_fl_round, self.config.num_fl_rounds):
            self.current_fl_round = fl_round
            round_start = time.time()

            self._fl_round(fl_round)

            # Validate global model
            self.logger.info("  Global model validation …")
            metrics = self._validate(self.model)
            self.logger.info(
                f"  Global valid — loss: {metrics['loss']:.4f}, "
                f"dice: {metrics['dice']:.4f}, hd: {metrics['hd']:.4f}"
            )

            # Determine if this is the best model
            save_metric = self.config.save_metric_name
            cur_val = metrics.get(save_metric, metrics["dice"])
            maximize = save_metric == "dice"

            is_best = (
                self._best_valid_metric is None
                or (maximize and cur_val > self._best_valid_metric)
                or (not maximize and cur_val < self._best_valid_metric)
            )
            if is_best:
                self._best_valid_metric = cur_val
                best_path = self.work_path / "best_model.pth"
                torch.save({"model": self.model.state_dict()}, best_path)
                self.logger.info(
                    f"  New best {save_metric}: {cur_val:.4f} → saved to {best_path}"
                )

            # Per-round checkpoint
            round_path = self.work_path / f"round_{fl_round}" / "model.pth"
            round_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": self.model.state_dict()}, round_path)

            elapsed = time.time() - round_start
            self.logger.info(f"  Round time: {elapsed:.1f}s")
            self.logger.info("")

        self.on_train_end()

    def run_training(self):
        self.train()

    def perform_real_test(self):
        self.logger.info("Running test on validation set …")
        metrics = self._validate(self.model)
        self.logger.info(
            f"Test — loss: {metrics['loss']:.4f}, "
            f"dice: {metrics['dice']:.4f}, hd: {metrics['hd']:.4f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "current_fl_round": self.current_fl_round,
            "best_valid_metric": self._best_valid_metric,
        }

    def load_state_dict(self, save_path: str | Path):
        state = torch.load(save_path, map_location="cpu")
        if "model" in state:
            self.model.load_state_dict(state["model"])
        if "current_fl_round" in state:
            self.current_fl_round = state["current_fl_round"]
        if "best_valid_metric" in state:
            self._best_valid_metric = state["best_valid_metric"]
        self.logger.info(f"Loaded state from {save_path}")

    def save_state_dict(self, save_path: str | Path):
        save_path = get_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(save_path))
        self.logger.info(f"Saved state to {save_path}")

    def to(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            self.device = device
        elif device.type == "mps" and torch.backends.mps.is_available():
            self.device = device
        else:
            self.device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Federated Active Learning
# ---------------------------------------------------------------------------

class FedALConfig(FLConfig):
    """Extends FLConfig with federated active learning parameters."""

    def __init__(
        self,
        al_strategy: Literal["entropy", "random"] = "entropy",
        al_budget_per_round: int = 5,
        al_initial_labeled: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.al_strategy = al_strategy
        self.al_budget_per_round = al_budget_per_round
        self.al_initial_labeled = al_initial_labeled


class FedALTrainer(FLTrainer):
    """Federated Learning + Active Learning trainer.

    Each client maintains a labeled set and an unlabeled pool.
    Before local training each round, the client queries the pool
    using the current global model and moves the selected samples
    into the labeled set.
    """

    def __init__(self, config: FedALConfig | dict | None = None, **kwargs):
        if isinstance(config, FedALConfig):
            pass
        elif isinstance(config, dict):
            config = FedALConfig(**config)
        else:
            config = FedALConfig()
        super().__init__(config=config, **kwargs)
        self.config: FedALConfig

    # ------------------------------------------------------------------
    # AL helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_pool(self, model: nn.Module, pool_indices: list[int], base_dataset) -> list[float]:
        """Score pool samples by predictive entropy (higher = more informative)."""
        model.eval()
        model.to(self.device)
        scores = []
        for idx in pool_indices:
            sample = base_dataset[idx]
            image = sample["image"].unsqueeze(0).to(self.device)
            with (
                torch.autocast(self.device.type)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                logits = model(image)
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
            scores.append(entropy)
        model.cpu()
        return scores

    def _select_from_pool(
        self,
        model: nn.Module,
        pool_indices: list[int],
        base_dataset,
        budget: int,
    ) -> list[int]:
        """Select `budget` samples from pool_indices based on the configured strategy."""
        if not pool_indices or budget <= 0:
            return []
        budget = min(budget, len(pool_indices))

        if self.config.al_strategy == "random":
            return random.sample(pool_indices, budget)

        # entropy (default)
        scores = self._score_pool(model, pool_indices, base_dataset)
        ranked = sorted(zip(scores, pool_indices), key=lambda x: -x[0])
        return [idx for _, idx in ranked[:budget]]

    # ------------------------------------------------------------------
    # Override on_train_start to build per-client AL state
    # ------------------------------------------------------------------

    def on_train_start(self):
        super().on_train_start()

        # Build per-client {labeled_indices, pool_indices} from their Subset
        self._client_al_state: list[dict] = []
        for cid, subset in enumerate(self.client_datasets):
            all_idx = list(subset.indices)
            random.shuffle(all_idx)
            n_init = min(self.config.al_initial_labeled, len(all_idx))
            labeled = all_idx[:n_init]
            pool = all_idx[n_init:]
            self._client_al_state.append({
                "labeled": labeled,
                "pool": pool,
                "base_dataset": subset.dataset,
            })
            self.logger.info(
                f"  Client {cid}: {len(labeled)} labeled, {len(pool)} pool"
            )

    # ------------------------------------------------------------------
    # Override _fl_round to use AL-gated local training
    # ------------------------------------------------------------------

    def _fl_round(self, fl_round: int):
        if self.config.aggregation == "fedper":
            return self._fl_round_fedper(fl_round)

        self.logger.info(
            f"FL Round {fl_round + 1}/{self.config.num_fl_rounds} [FedAL/{self.config.al_strategy}]"
        )

        selected = self._select_clients()
        self.logger.info(f"  Selected clients: {selected}")

        client_results = []
        for cid in selected:
            al_state = self._client_al_state[cid]
            if not al_state["labeled"]:
                self.logger.warning(f"  Client {cid} has no labeled data — skipping")
                continue

            # Query pool
            if al_state["pool"]:
                new_indices = self._select_from_pool(
                    model=deepcopy(self.model),
                    pool_indices=al_state["pool"],
                    base_dataset=al_state["base_dataset"],
                    budget=self.config.al_budget_per_round,
                )
                for idx in new_indices:
                    al_state["pool"].remove(idx)
                    al_state["labeled"].append(idx)
                self.logger.info(
                    f"  Client {cid}: queried {len(new_indices)} → "
                    f"labeled={len(al_state['labeled'])}, pool={len(al_state['pool'])}"
                )

            labeled_dataset = Subset(al_state["base_dataset"], al_state["labeled"])

            local_model = deepcopy(self.model)
            result = self._local_train(
                client_id=cid,
                local_model=local_model,
                client_dataset=labeled_dataset,
                global_model=self.model if self.config.aggregation == "fedprox" else None,
            )
            client_results.append(result)

        if not client_results:
            self.logger.warning("No client results to aggregate — skipping round")
            return

        self.logger.info(f"  Aggregation: {self.config.aggregation}")
        client_models = [r["model"] for r in client_results]
        client_weights = [r["data_size"] for r in client_results]

        if isinstance(self.aggregator, FedNovaAggregator):
            client_local_steps = [r["actual_steps"] for r in client_results]
            self.aggregator.aggregate(
                self.model, client_models, client_weights,
                client_local_steps=client_local_steps,
            )
        else:
            self.aggregator.aggregate(self.model, client_models, client_weights)

        avg_client_loss = np.mean([r["avg_loss"] for r in client_results])
        self.logger.info(f"  Mean client loss: {avg_client_loss:.4f}")
