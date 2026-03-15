# Medical Image Analysis

## How to run

To run the active learning trainining, follow the following steps:

1. Install this repo:

```bash
pip install git+https://github.com/dacphu/medical-image-analysis
```

2. Download datasets:
- FUGC 2025: [\[Raw\]](https://drive.google.com/file/d/1VMsbOKzJaSKekAKdtH4eHZA-KGL_HAbM/view?usp=sharing) [\[Processed\]](https://drive.google.com/file/d/1OK1EQgNQG2BSDzAH4wQ4zq3Z3IIiP819/view?usp=sharing)
- BUSI: [\[Processed\]](https://drive.google.com/file/d/1JEXKTKV9XHvunD4E3fAHmUXYQH2IHd1q/view?usp=sharing)

3. Download model checkponts (Optional):
- UNet: [Initial round](https://drive.google.com/file/d/1aOQicgo_EF-PiZJPVSKKOqMEI9AHOOd4/view?usp=sharing)

4. Run commands:

To run training using active learning:

```bash
al_train --work-path <work-path> --dataset {fugc,busi} --data-path <path/to/data> --budget <budget> --num-rounds <num-rounds>
```

To run training using federated learning (FedAvg, IID split):

```bash
fl_train \
  --work-path <work-path> \
  --dataset {ACDC,fugc,busi,tn3k,tg3k} \
  --data-path <path/to/data> \
  --num-clients <num-clients> \
  --num-fl-rounds <num-fl-rounds> \
  --local-iters <local-iters>
```

Key federated learning options:

| Flag | Default | Description |
|---|---|---|
| `--num-clients` | `5` | Number of simulated FL clients |
| `--num-fl-rounds` | `10` | Number of communication rounds |
| `--local-iters` | `200` | Local SGD steps per client per round |
| `--client-fraction` | `1.0` | Fraction of clients selected each round |
| `--aggregation` | `fedavg` | Aggregation strategy (`fedavg` or `fedprox`) |
| `--fedprox-mu` | `0.01` | Proximal term coefficient μ (FedProx only) |
| `--dirichlet-alpha` | *(IID)* | Dirichlet α for non-IID data split (e.g. `0.5`) |

Example — FedProx with non-IID data across 10 clients:

```bash
fl_train \
  --work-path ./fl_output \
  --dataset ACDC \
  --data-path <path/to/data> \
  --num-clients 10 \
  --num-fl-rounds 20 \
  --local-iters 200 \
  --client-fraction 0.5 \
  --aggregation fedprox \
  --fedprox-mu 0.01 \
  --dirichlet-alpha 0.5 \
  --do-augment --do-normalize
```

To run the demo:

```bash
demo_serve
```