"""Comparison evaluation script.

Reads log.txt files from experiment work directories, extracts the best
dice/hd values, and writes summary.csv + summary.tex (LaTeX table ready
to paste into the paper).

Usage (collect only — no new training):
    python -m evaluation.compare_methods \\
        --work-path /path/to/experiments \\
        --collect-only

Usage (run all experiments then collect):
    python -m evaluation.compare_methods \\
        --dataset ACDC --data-path /data/ACDC \\
        --work-path /path/to/experiments
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "Centralized",
        "aggregation": "fedavg",
        "num_clients": 1,
        "num_fl_rounds": 50,
        "local_iters": 500,
        "dirichlet_alpha": None,
        "extra": [],
    },
    {
        "name": "FedAvg (IID)",
        "aggregation": "fedavg",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": None,
        "extra": [],
    },
    {
        "name": "FedAvg (α=1.0)",
        "aggregation": "fedavg",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 1.0,
        "extra": [],
    },
    {
        "name": "FedAvg (α=0.5)",
        "aggregation": "fedavg",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.5,
        "extra": [],
    },
    {
        "name": "FedAvg (α=0.1)",
        "aggregation": "fedavg",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.1,
        "extra": [],
    },
    {
        "name": "FedProx μ=0.001 (α=0.1)",
        "aggregation": "fedprox",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.1,
        "extra": ["--fedprox-mu", "0.001"],
    },
    {
        "name": "FedProx μ=0.01 (α=0.1)",
        "aggregation": "fedprox",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.1,
        "extra": ["--fedprox-mu", "0.01"],
    },
    {
        "name": "FedProx μ=0.1 (α=0.1)",
        "aggregation": "fedprox",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.1,
        "extra": ["--fedprox-mu", "0.1"],
    },
    {
        "name": "FedNova (α=0.1)",
        "aggregation": "fednova",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.1,
        "extra": [],
    },
    {
        "name": "FedPer (α=0.1)",
        "aggregation": "fedper",
        "num_clients": 5,
        "num_fl_rounds": 10,
        "local_iters": 200,
        "dirichlet_alpha": 0.1,
        "extra": [],
    },
]


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _parse_log(log_path: Path) -> dict[str, float]:
    """Extract the best dice and corresponding hd from a log.txt file."""
    best_dice = 0.0
    best_hd = float("inf")

    dice_pattern = re.compile(r"dice:\s*([\d.]+)")
    hd_pattern = re.compile(r"hd:\s*([\d.inf]+)")

    if not log_path.exists():
        return {"dice": float("nan"), "hd": float("nan"), "found": False}

    with open(log_path) as f:
        for line in f:
            dm = dice_pattern.search(line)
            hm = hd_pattern.search(line)
            if dm and hm:
                dice_val = float(dm.group(1))
                hd_str = hm.group(1)
                hd_val = float("inf") if hd_str == "inf" else float(hd_str)
                if dice_val > best_dice:
                    best_dice = dice_val
                    best_hd = hd_val

    return {"dice": best_dice, "hd": best_hd, "found": True}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class CompareRunner:
    def __init__(
        self,
        dataset: str,
        data_path: str,
        work_path: str,
        extra_train_args: list[str] | None = None,
    ):
        self.dataset = dataset
        self.data_path = data_path
        self.work_path = Path(work_path)
        self.extra_train_args = extra_train_args or []

    def _exp_dir_name(self, exp: dict) -> str:
        alpha_str = str(exp["dirichlet_alpha"]) if exp["dirichlet_alpha"] is not None else "iid"
        return f"{self.dataset}_{exp['aggregation']}_K{exp['num_clients']}_alpha{alpha_str}"

    def _run_experiment(self, exp: dict):
        exp_dir = self.work_path / self._exp_dir_name(exp)
        cmd = [
            sys.executable, "-m", "entry.federated.train",
            "--dataset", self.dataset,
            "--data-path", self.data_path,
            "--work-path", str(exp_dir),
            "--num-clients", str(exp["num_clients"]),
            "--num-fl-rounds", str(exp["num_fl_rounds"]),
            "--local-iters", str(exp["local_iters"]),
            "--aggregation", exp["aggregation"],
            "--do-augment",
            "--do-normalize",
            "--exp-name", exp["name"].replace(" ", "_"),
        ]
        if exp["dirichlet_alpha"] is not None:
            cmd += ["--dirichlet-alpha", str(exp["dirichlet_alpha"])]
        cmd += exp.get("extra", [])
        cmd += self.extra_train_args

        print(f"\n[RUN] {exp['name']}")
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  [WARNING] Experiment '{exp['name']}' exited with code {result.returncode}")

    def _collect_results(self) -> list[dict]:
        rows = []
        for exp in EXPERIMENTS:
            exp_dir = self.work_path / self._exp_dir_name(exp)
            # Search for log files (there may be timestamped variants)
            log_candidates = sorted(exp_dir.glob("log*.txt"))
            log_path = log_candidates[0] if log_candidates else exp_dir / "log.txt"
            metrics = _parse_log(log_path)
            rows.append({
                "Method": exp["name"],
                "Dice (%)": f"{metrics['dice'] * 100:.2f}" if metrics["found"] else "--",
                "HD (mm)": f"{metrics['hd']:.2f}" if metrics["found"] else "--",
                "log": str(log_path),
            })
        return rows

    def _write_csv(self, rows: list[dict], out_path: Path):
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Method", "Dice (%)", "HD (mm)", "log"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[CSV] Written to {out_path}")

    def _write_latex(self, rows: list[dict], out_path: Path):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{Comparison of FL methods on {self.dataset}.}}",
            rf"\label{{tab:results_{self.dataset.lower()}}}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Method & DSC (\%) $\uparrow$ & HD (mm) $\downarrow$ \\",
            r"\midrule",
        ]
        for row in rows:
            method = row["Method"].replace("α", r"$\alpha$").replace("μ", r"$\mu$")
            lines.append(
                f"{method} & {row['Dice (%)']} & {row['HD (mm)']} \\\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        out_path.write_text("\n".join(lines))
        print(f"[TEX] Written to {out_path}")

    def run(self, collect_only: bool = False):
        self.work_path.mkdir(parents=True, exist_ok=True)

        if not collect_only:
            for exp in EXPERIMENTS:
                self._run_experiment(exp)

        rows = self._collect_results()
        self._write_csv(rows, self.work_path / "summary.csv")
        self._write_latex(rows, self.work_path / "summary.tex")

        print("\n=== Summary ===")
        header = f"{'Method':<35} {'Dice (%)':>10} {'HD (mm)':>10}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['Method']:<35} {row['Dice (%)']:>10} {row['HD (mm)']:>10}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run comparison experiments and collect results into a LaTeX table"
    )
    parser.add_argument("--dataset", default="ACDC", type=str)
    parser.add_argument("--data-path", default="", type=str)
    parser.add_argument("--work-path", required=True, type=str)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip training; only parse existing logs and write summary",
    )
    args, extra = parser.parse_known_args()

    runner = CompareRunner(
        dataset=args.dataset,
        data_path=args.data_path,
        work_path=args.work_path,
        extra_train_args=extra,
    )
    runner.run(collect_only=args.collect_only)


if __name__ == "__main__":
    main()
