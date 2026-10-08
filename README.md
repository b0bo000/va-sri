# VA-SRI: Learning what interpolation misses for block imputation of multivariate time series

Code for the paper *VA-SRI: Learning what interpolation misses for block imputation of multivariate
time series*. VA-SRI keeps the linear interpolation of the visible values (the *skeleton*) as a fixed
base and learns only the residual; the residual is computed by a variable-axis encoder that gives each
variable its own sequence of patch tokens and alternates attention over time and over variables.

## Setup

Python 3.10, PyTorch 2.x and PyPOTS 0.4.x (for SAITS, TimesNet and Crossformer):

```bash
pip install -r requirements.txt
```

## Data

The datasets are the public releases of [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
Place `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`, `weather.csv` and `electricity.csv` in `data/`.
Each series is split chronologically 7:1:2; windows of length 96 do not cross split boundaries.

## Reproducing the main results

Pretrain the shared encoder on the four ETT training splits (once per seed):

```bash
python -m scripts.va_sri.pretrain --datasets ETTh1 ETTh2 ETTm1 ETTm2 --updates 40000 --seed 42 \
    --out experiments/va_sri/pretrain/pre_p2ett_s42
```

Finetune and evaluate VA-SRI with the final configuration (200 epochs for ETTh1/ETTh2, 100 otherwise):

```bash
python -m scripts.va_sri.train --dataset ETTh2 --seed 42 --vga beta_g --epochs 200 \
    --ema 0.999 --patience 0 --unweighted --select std_mse --lr 1e-3 --mae-weight 1 \
    --init experiments/va_sri/pretrain/pre_p2ett_s42/shared.pt --eval-test \
    --out experiments/va_sri/runs/ft_etth2_s42_final
```

The paper uses seeds 42-46. Tuned SAITS-blocksup:

```bash
python run_saits_block_supervised.py --root_path ./data --data_path ETTh2.csv --seq_len 96 \
    --block_len 24 --batch_size 64 --reproducible_epoch_mask --mask_mode block --seed 42 \
    --lr 1e-3 --select_std --epochs 200 --patience 200 --ema_decay 0.999 --mse_mae_loss
```

`scripts/va_sri/make_jobs.py` lists the exact command of every experiment in the paper (tuning,
main comparison, factorial, ablations, gap lengths, missingness patterns, Electricity, TimesNet and
Crossformer). Tuning decisions used only the ETTh2 and ETTm1 validation splits.

## Layout

| Path | Content |
|---|---|
| `scripts/va_sri/model.py` | VA-SRI (skeleton, variable-axis encoder, visibility bias, residual output, refiner) |
| `scripts/va_sri/train.py`, `pretrain.py` | finetuning / evaluation and pretraining |
| `scripts/va_sri/run_backbone.py` | TimesNet and Crossformer under the same protocol |
| `run_saits_block_supervised.py` | SAITS trained on the gap targets |
| `scripts/va_sri/render_paper.py` | tables and numbers of the paper from the result files |
| `scripts/va_sri/gap_position.py`, `example_grid.py` | analyses behind Figures 2 and 3 |
| `data/`, `models/`, `losses/`, `utils/`, `engine/` | data loading, masks, interpolation and shared modules |

## Citation

The citation will be added once the paper is published.
