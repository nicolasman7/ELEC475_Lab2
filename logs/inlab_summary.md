# ELEC 475 Lab 2 — In-Lab Summary (LeNet-5 vs. Modernized LeNet)

Setup: CIFAR-10 resized to 64×64, 5 epochs, batch 128, Adam (lr 1e-3, wd 1e-4), seed 42, augmentation flag on
(the augmentation TODO is still empty, so no augmentation was actually applied). Device: **CPU** (Apple Silicon;
`train.py` only selects CUDA). Logs: `logs/train_lenet.txt`, `logs/train_lenet_modern.txt`, `logs/arch_*.txt`.

## 1. Comparison table

| Metric (epoch 5 unless noted) | LeNet-5 (baseline) | Modern LeNet | Δ |
|---|---|---|---|
| Parameters | 338,486 | 446,006 | +31.8% |
| MACs | 3.59 M | 4.59 M | +28.1% |
| FLOPs (2 × MACs) | 7.17 M | 9.19 M | +28.1% |
| Final train loss | 1.1879 | 0.8888 | −0.299 |
| Final val loss | 1.3076 | 1.0118 | −0.296 |
| Final train acc | 57.82% | 68.78% | +10.96 pts |
| Final val acc | 53.16% | 64.51% | +11.35 pts |
| Best val acc (epoch) | 53.16% (5) | 64.73% (4) | +11.57 pts |
| Final val Macro F1 | 53.19% | 64.00% | +10.81 pts |
| Mean epoch time | 54.44 s | 61.12 s | +12.3% |
| Throughput | ~918 samples/s | ~818 samples/s | −10.9% |
| Total training time | 272.3 s | 305.6 s | +12.3% |

Saved checkpoints hold the **best-val-acc** epoch: epoch 5 for the baseline, epoch 4 for the modern model.

Per-epoch validation loss:

| Epoch | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| LeNet-5 | 1.5042 | 1.3984 | 1.4068 | 1.3404 | 1.3076 |
| Modern | 1.3164 | 1.1379 | 1.0655 | 1.0166 | 1.0118 |

## 2. Analytical questions

**Q1. Did the modernized LeNet beat the baseline, and why?**
Yes, clearly. It reached 64.51% val accuracy / 64.00% Macro F1 against 53.16% / 53.19% (+11 pts), and after a
single epoch (52.06%) it already matched the baseline's 5-epoch result. Reasons:
- **ReLU vs Tanh:** ReLU doesn't saturate, so gradients flow through every active unit at full strength and the network
  learns faster in the same 5 epochs (train loss 0.89 vs 1.19).
- **Overlapping 3×3/s2 max pooling vs 2×2 avg pooling:** max pooling keeps the strongest activation (edges, textures)
  instead of blurring it out. The overlap gives some translation tolerance and mild regularization.
- **Padding = 2:** the convolutions no longer shrink the borders, so edge information is kept and FC1 sees a larger
  16×15×15 map.
- Caveat: these three changes (plus 32% more parameters, mostly in FC1) were made together, so this run can't
  attribute the gain to any one of them. Isolating them would need an ablation that changes one thing at a time.

**Q2. FLOP / speed trade-off.**
The modern model costs +28% FLOPs (7.17 → 9.19 M) but only +12.3% wall-clock per epoch (54.4 → 61.1 s, throughput
918 → 818 samples/s). In return it gains about 11 points of accuracy and F1, which is a very good trade. Time grows
less than FLOPs because each epoch also includes work that doesn't scale with model FLOPs: PIL decoding,
resizing 32→64, ToTensor and Normalize in 2 DataLoader workers, Python/optimizer overhead, and the validation pass.
The extra FLOPs come from padding: the conv layers run on 64×64 and 31×31 maps instead of 60×60 and 26×26
(conv MACs 3.24 M → 4.15 M), and FC1 grows from 2704 to 3600 inputs.

**Q3. Has val loss plateaued? Would early stopping halt?**
- *Baseline:* not plateaued. Val loss is still falling (1.3404 → 1.3076 in epoch 5). There was one non-improving
  epoch (epoch 3: 1.4068 > 1.3984), but epoch 4 reset the counter.
- *Modern:* it is starting to plateau. Val loss fell only 0.0048 in epoch 5 (1.0166 → 1.0118), while train loss
  fell 0.081, and val accuracy dipped (64.73 → 64.51). Train acc (68.8%) now exceeds val acc (64.5%), which is
  the beginning of overfitting.
- *Early stopping* (`--early-stopping`, defaults: monitor val_loss, patience 5, min_delta 1e-4) would **not** halt
  either run within 5 epochs. Every modern epoch improved by more than 1e-4, and the baseline had only 1 bad
  epoch against a patience of 5. Monitoring `val_acc` with patience 1 would have stopped the modern model after
  epoch 5 and restored epoch 4. Over a longer run (e.g. 15 epochs), the modern model's val loss is likely to turn
  upward and trigger the stop.

## 3. Oral-defence cheat sheet

Output size: **O = ⌊(W − K + 2P) / S⌋ + 1**

**LeNet-5 → 2704**

| Layer | Calc | Output |
|---|---|---|
| Input | — | 3 × 64 × 64 |
| Conv1 k5 p0 s1 | (64 − 5 + 0)/1 + 1 = 60 | 6 × 60 × 60 |
| AvgPool 2/2 | (60 − 2)/2 + 1 = 30 | 6 × 30 × 30 |
| Conv2 k5 p0 s1 | (30 − 5 + 0)/1 + 1 = 26 | 16 × 26 × 26 |
| AvgPool 2/2 | (26 − 2)/2 + 1 = 13 | 16 × 13 × 13 |
| Flatten | 16 × 13 × 13 = 16 × 169 | **2704** |

**Modern LeNet → 3600**

| Layer | Calc | Output |
|---|---|---|
| Input | — | 3 × 64 × 64 |
| Conv1 k5 p2 s1 | (64 − 5 + 4)/1 + 1 = 64 ("same" padding) | 6 × 64 × 64 |
| MaxPool 3/s2 | ⌊(64 − 3)/2⌋ + 1 = ⌊30.5⌋ + 1 = 31 | 6 × 31 × 31 |
| Conv2 k5 p2 s1 | (31 − 5 + 4)/1 + 1 = 31 | 16 × 31 × 31 |
| MaxPool 3/s2 | ⌊(31 − 3)/2⌋ + 1 = 14 + 1 = 15 | 16 × 15 × 15 |
| Flatten | 16 × 15 × 15 = 16 × 225 | **3600** |

The pools "overlap" because kernel 3 > stride 2, so adjacent windows share one row/column.

**Parameter check:** a conv layer has (K·K·C_in + 1)·C_out parameters; a linear layer has (in + 1)·out.
- Baseline: 456 + 2,416 + 324,600 + 10,164 + 850 = **338,486**
- Modern: 456 + 2,416 + **432,120** + 10,164 + 850 = **446,006**. The whole difference is in FC1 (3600 vs 2704 inputs).

**MAC check (baseline):** conv1 6·60·60·75 = 1.62 M, conv2 16·26·26·150 = 1.62 M, FCs 0.34 M → **3.58 M MACs ≈ 7.2 MFLOPs**.

**Why ReLU converges faster than Tanh**
1. **No saturation / vanishing gradient:** tanh′(x) = 1 − tanh²(x) ≤ 1 and goes to 0 for |x| ≳ 2, so saturated units pass
   almost no gradient. Multiplying these factors layer by layer makes the gradient shrink. ReLU′(x) = 1 for all x > 0,
   so gradients pass through active units unchanged.
2. **Cheap:** max(0, x) is a comparison, with no exponentials, in both the forward and backward pass.
3. **Sparse activations:** about half the units output exactly 0, which gives simpler, more separable representations.
4. **Evidence:** Krizhevsky et al. (AlexNet, 2012) found a 4-layer CNN with ReLU reached 25% training error on
   CIFAR-10 about 6× faster than the same network with tanh. In this lab, modern LeNet epoch-1 val acc was 52.1% vs 46.3%.
5. **Caveat to mention:** ReLU can "die" (it stays at 0 forever if its input is always negative). Leaky ReLU, GELU,
   and similar activations fix this.
