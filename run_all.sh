#!/usr/bin/env bash
set -o pipefail
mkdir -p logs plots checkpoints

sync_results() {
    [ -n "$RESULTS_DIR" ] && mkdir -p "$RESULTS_DIR" && cp -ru logs plots checkpoints "$RESULTS_DIR"/
}

run() {
    local name=$1; shift
    local log="logs/run_${name}.txt"
    if grep -q "Training completed" "$log" 2>/dev/null; then echo "[skip] $name"; return; fi
    echo "================ [$(date +%H:%M:%S)] $name ================"
    "$@" 2>&1 | tee "$log" || echo "[FAILED] $name (see $log)"
    sync_results
}

run lenet          python train.py --model lenet --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/lenet_baseline.pth --plot-file plots/loss_lenet.png
run lenet_modern   python train.py --model lenet_modern --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/lenet_modern.pth --plot-file plots/loss_lenet_modern.png
run alexnet_none   python train.py --model alexnet --norm none --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_none.pth --plot-file plots/loss_alexnet_none.png
run alexnet_lrn    python train.py --model alexnet --norm lrn --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_lrn.pth --plot-file plots/loss_alexnet_lrn.png
run alexnet_bn     python train.py --model alexnet --norm bn --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/alexnet_bn.pth --plot-file plots/loss_alexnet_bn.png
run alexnet_no_aug python train.py --model alexnet --norm bn --dataset cifar10 --epochs 15 --no-augment --save-model checkpoints/alexnet_no_aug.pth --plot-file plots/loss_no_aug.png
run alexnet_with_aug python train.py --model alexnet --norm bn --dataset cifar10 --epochs 15 --save-model checkpoints/alexnet_with_aug.pth --plot-file plots/loss_with_aug.png
run zoo_alexnet    python train.py --model zoo_alexnet --dataset cifar10 --epochs 10 --no-augment --save-model checkpoints/zoo_alexnet.pth --plot-file plots/loss_zoo_alexnet.png

python compare_models.py --dataset cifar10 2>&1 | tee logs/compare_models_cifar10.txt
python compare_models.py --dataset tiny_imagenet 2>&1 | tee logs/compare_models_tiny_imagenet.txt
python demo_transforms.py 2>&1 | tee logs/demo_transforms.txt
python -m utils.visualization --plot-ablation
sync_results

run tiny_alexnet_bn_aug python train.py --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --save-model checkpoints/alexnet_tiny_imagenet.pth --plot-file plots/loss_tiny_imagenet.png

for seed in 1 2; do
    for norm in none lrn bn; do
        run alexnet_${norm}_seed${seed} python train.py --model alexnet --norm $norm --dataset cifar10 --epochs 10 --no-augment --seed $seed --plot-file plots/loss_alexnet_${norm}_seed${seed}.png
    done
done

run tiny_lenet        python train.py --model lenet --dataset tiny_imagenet --epochs 10 --no-augment --plot-file plots/loss_tiny_lenet.png
run tiny_lenet_modern python train.py --model lenet_modern --dataset tiny_imagenet --epochs 10 --no-augment --plot-file plots/loss_tiny_lenet_modern.png
run tiny_alexnet_none python train.py --model alexnet --norm none --dataset tiny_imagenet --epochs 10 --no-augment --plot-file plots/loss_tiny_alexnet_none.png
run tiny_alexnet_lrn  python train.py --model alexnet --norm lrn --dataset tiny_imagenet --epochs 10 --no-augment --plot-file plots/loss_tiny_alexnet_lrn.png
run tiny_alexnet_bn   python train.py --model alexnet --norm bn --dataset tiny_imagenet --epochs 10 --no-augment --plot-file plots/loss_tiny_alexnet_bn.png
run tiny_zoo_alexnet  python train.py --model zoo_alexnet --dataset tiny_imagenet --epochs 10 --no-augment --plot-file plots/loss_tiny_zoo_alexnet.png

python -m utils.visualization --plot-ablation
sync_results
echo "ALL DONE"
