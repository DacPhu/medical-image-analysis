from argparse import ArgumentParser

from training.fl_trainer import FLConfig, FLTrainer


def parse_args():
    parser = ArgumentParser(
        description="Federated Learning training for medical image segmentation"
    )

    parser.add_argument("--work-path", default=".", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")

    # >>> Model parameters
    parser.add_argument("--in-channels", default=1, type=int)
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--block-type", default="plain", type=str)
    parser.add_argument("--block-normalization", default="batch", type=str)
    parser.add_argument("--dropout-prob", default=0.1, type=float)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--ds-layer", default=3, type=int)
    parser.add_argument("--image-size", default=[256], nargs="+", type=int)
    parser.add_argument("--model-ckpt", default=None, type=str)
    # <<< Model parameters

    # >>> Data parameters
    parser.add_argument("--dataset", default="ACDC", type=str)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--do-oversample", action="store_true")
    parser.add_argument("--do-augment", action="store_true")
    parser.add_argument("--do-normalize", action="store_true")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--valid-batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--pin-memory", action="store_true")
    # <<< Data parameters

    # >>> FL parameters
    parser.add_argument("--num-clients", default=5, type=int,
                        help="Total number of simulated FL clients")
    parser.add_argument("--num-fl-rounds", default=10, type=int,
                        help="Number of federated learning communication rounds")
    parser.add_argument("--local-iters", default=200, type=int,
                        help="Number of local SGD iterations per client per round")
    parser.add_argument("--client-fraction", default=1.0, type=float,
                        help="Fraction of clients selected per round (C in FedAvg)")
    parser.add_argument("--aggregation", default="fedavg",
                        choices=["fedavg", "fedprox", "fednova", "fedper"],
                        help="Aggregation strategy")
    parser.add_argument("--fedprox-mu", default=0.01, type=float,
                        help="FedProx proximal term coefficient μ")
    parser.add_argument("--dirichlet-alpha", default=None, type=float,
                        help="Dirichlet α for non-IID partitioning (None = IID)")
    # <<< FL parameters

    # >>> Optimizer parameters
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--start-lr", default=1e-3, type=float)
    parser.add_argument("--lr-scheduler", default="poly", type=str)
    parser.add_argument("--lr-interval", default=1, type=int)
    parser.add_argument("--lr-warmup-iter", default=50, type=int)
    parser.add_argument("--grad-norm", default=10.0, type=float)
    # <<< Optimizer parameters

    # >>> Loss parameters
    parser.add_argument("--loss", default="dice+ce", type=str)
    parser.add_argument("--dice-weight", default=1.0, type=float)
    parser.add_argument("--ce-weight", default=1.0, type=float)
    # <<< Loss parameters

    # >>> Validation parameters
    parser.add_argument("--valid-freq-iter", default=200, type=int)
    parser.add_argument("--save-metric", default="dice", type=str)
    # <<< Validation parameters

    # >>> Log parameters
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.add_argument("--log-path", default=None, type=str)
    parser.add_argument("--config-path", default=None, type=str)
    parser.add_argument("--exp-name", default="", type=str)
    # <<< Log parameters

    return parser.parse_args()


def train_entry():
    args = parse_args()

    config = FLConfig(
        seed=args.seed,
        # Model
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        block_type=args.block_type,
        block_normalization=args.block_normalization,
        dropout_prob=args.dropout_prob,
        deep_supervision=args.deep_supervision,
        ds_layer=args.ds_layer,
        image_size=(
            tuple(args.image_size) if len(args.image_size) > 1 else args.image_size[0]
        ),
        model_ckpt=args.model_ckpt,
        # Data
        dataset=args.dataset,
        data_path=args.data_path,
        do_oversample=args.do_oversample,
        do_augment=args.do_augment,
        do_normalize=args.do_normalize,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # FL
        num_clients=args.num_clients,
        num_fl_rounds=args.num_fl_rounds,
        local_iters=args.local_iters,
        client_fraction=args.client_fraction,
        aggregation=args.aggregation,
        fedprox_mu=args.fedprox_mu,
        dirichlet_alpha=args.dirichlet_alpha,
        # Optimizer
        optimizer_name=args.optimizer,
        optimizer_kwargs={"weight_decay": args.weight_decay},
        start_lr=args.start_lr,
        lr_scheduler_name=args.lr_scheduler,
        lr_interval=args.lr_interval,
        lr_warmup_iter=args.lr_warmup_iter,
        grad_norm=args.grad_norm,
        # Loss
        loss_name=args.loss,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        # Validation
        valid_freq_iter=args.valid_freq_iter,
        save_metric_name=args.save_metric,
        # Misc
        exp_name=args.exp_name,
    )

    trainer = FLTrainer(
        work_path=args.work_path,
        deterministic=args.deterministic,
        device=args.device,
        config=config,
        verbose=args.verbose,
        log_path=args.log_path,
        config_path=args.config_path,
    )
    trainer.initialize()

    if args.test_only:
        if args.resume:
            trainer.load_state_dict(args.resume)
        trainer.perform_real_test()
    else:
        if args.resume:
            trainer.load_state_dict(args.resume)
        trainer.run_training()
