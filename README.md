# ELEC 475 Lab 2: LeNet-5 to AlexNet

LeNet-5, a modernized LeNet, AlexNet-64 (no normalization / LRN / BatchNorm) and the torchvision AlexNet trained on CIFAR-10 and Tiny ImageNet-200 at 64×64, with a data augmentation study.

## Results (best validation accuracy, seed 42)

| Model | CIFAR-10 | Tiny ImageNet |
|---|---|---|
| LeNet-5 | 55.23% | 12.41% |
| Modernized LeNet | 66.65% | 20.77% |
| AlexNet-64 (no norm) | 73.52% | 16.46% |
| AlexNet-64 + LRN | 75.14% | 17.01% (seed 1) |
| AlexNet-64 + BatchNorm | 79.93% | 30.15% |
| AlexNet-64 + BatchNorm + augmentation | 82.83% | 29.20% |
| torchvision AlexNet | 65.57% | 11.83% |

Trained on a Tesla T4 (Google Colab), Adam lr 1e-3, batch 128, 10 epochs (15 for the augmentation study). Full training logs are in `logs/` and curves in `plots/`.

## Layout

```text
models/      lenet.py, lenet_modern.py, alexnet.py, zoo.py
utils/       dataset.py, transforms.py, metrics.py, profiler.py, seed.py, eda.py, visualization.py
configs/     config.py
train.py             training script
compare_models.py    params / FLOPs / latency benchmark
run_all.sh           runs every experiment in order
commands.txt         every training command used
logs/  plots/        results
```

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

CIFAR-10 and Tiny ImageNet download automatically into `data/`. A single run:

```bash
python train.py --model alexnet --norm bn --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_bn.pth --plot-file plots/loss_alexnet_bn.png
```
