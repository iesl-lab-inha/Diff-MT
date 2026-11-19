# Diff-MT: Differential Multimodal Transformer

Reference implementation accompanying the paper **Multimodal Biosignals for Affective State Recognition Using Differential Multimodal Transformer**.

Muhammad Saleem and Deok-Hwan Kim, *IEEE Access*, vol. 13, pp. 202395–202407, 2025.

[Publication (DOI)](https://doi.org/10.1109/ACCESS.2025.3637310)

## Overview

Diff-MT processes heterogeneous physiological signals for affective-state and cognitive-load classification. The architecture comprises:

- **DMFE:** modality-specific Conv1D alignment, overlapping windows, multi-scale temporal differencing and fusion with the original features.
- **GCE:** sinusoidal positional encoding, global cross-modal multi-head attention, residual connections and layer normalization.
- **DWF:** separate self-attention pooling for the differential and global features, followed by weighted fusion and classification.

The classifier is optimized with multiclass focal loss.

## Implementation scope

This is a standalone reference reconstruction from Sections III–IV, Algorithm 1 and Figures 2–3 of the publication, not the archived source used to produce the published experiments. The article does not fully specify the alignment kernels, window and stride schedules, feature dimensions, attention depth, focal-loss parameters, exact scheduler, label mapping or split indices. These details are configurable where practical; defaults are implementation choices, not recovered experimental settings. In particular, DMFE uses pooled-window residuals and lag differences with overlap-add; GCE uses a shared attention layer over all modality tokens; DWF uses a trainable scalar and mean pooling after self-attention. The component-removal and fusion baselines are controlled reference variants. No original training checkpoints or dataset preprocessing pipelines are included.

## Repository files

| File | Contents |
|---|---|
| `Diff_MT_model.py` | DMFE, GCE, DWF, classifier and focal loss |
| `train.py` | Prepared-array loader, cross-validation, training and evaluation |
| `Diff_MT_ablation_study.py` | Full model and three component ablations |
| `smoke_test.py` | Synthetic-input forward and backward checks |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Local data, outputs and environments |

## Installation

Python 3.10 or later is recommended. Install a compatible PyTorch build for your CPU or CUDA environment using the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
python smoke_test.py
```

## Datasets and input format

Obtain each dataset from its original provider, subject to the applicable access and usage terms. This repository does not distribute participant records.

| Dataset | Input keys | Classification target |
|---|---|---|
| WESAD | GSR, BVP, EMG, ECG, RESP | Stress, amusement, neutral |
| DEAP | EEG, EMG, EOG, GSR | Binary valence or binary arousal |
| MOCAS | EEG_POW, EEG, BVP, EAR, GSR | Low, medium, high cognitive load |
| CogLoad | GSR, SKT, HR, ACC, IBI | Low, medium, high cognitive load |

Prepare synchronized samples in a NumPy `.npz` file. Each modality is a float array with shape `[samples, channels, time]` (or `[samples, time]` for a single channel). Different modalities may have different channel counts and temporal lengths, but their sample order must match. Each modality must have a fixed shape across samples. The `labels` key contains integer class IDs starting at zero, with no gaps. Use a separate label key for each DEAP target (e.g. `valence`, `arousal`). An optional `groups` array stores subject or trial IDs for group-disjoint evaluation. Any preprocessing requiring fitted statistics must be fitted using training data only; the input loader does not perform raw-dataset segmentation, normalization or label mapping.

```python
import numpy as np
# Arrays below are placeholders for prepared synchronized data.
# np.savez('data/wesad.npz', GSR=gsr, BVP=bvp, EMG=emg,
#          ECG=ecg, RESP=resp, labels=labels, groups=subject_ids)
```

## Training and evaluation

```bash
python train.py --dataset wesad --data data/wesad.npz --output results/wesad
python train.py --dataset deap --data data/deap.npz --label-key valence --output results/deap_valence
python train.py --dataset deap --data data/deap.npz --label-key arousal --output results/deap_arousal
python train.py --dataset mocas --data data/mocas.npz --output results/mocas
python train.py --dataset cogload --data data/cogload.npz --output results/cogload
```

The paper reports randomized 10-fold validation, Adam with learning rate 0.0002, weight decay 0.01, betas (0.9, 0.98), dropout 0.1, 100–120 epochs, and batch sizes 1024 for DEAP/MOCAS and 512 for WESAD/CogLoad. This implementation uses stratified 10-fold validation with an inner validation split, cosine annealing, 100 epochs, and training-fold inverse-frequency class weights with focal gamma 2. These are not asserted to reproduce the exact published protocol. Adjust `--epochs`, `--batch-size`, `--gamma`, `--class-weights`, and model parameters as needed. Use `--grouped` with the `groups` array for group-disjoint validation; this is a separate protocol from the reported randomized sample-level evaluation. Large batches may exceed device memory; reduce the batch size for your hardware.

Outputs include fold indices, model checkpoints, test predictions, per-fold accuracy and macro F1, mean and standard deviation, and the run configuration. No published score is used by the training or evaluation code.

## Ablation and fusion comparisons

```bash
python Diff_MT_ablation_study.py --dataset wesad --data data/wesad.npz --output results/ablation
python train.py --dataset wesad --data data/wesad.npz --variants early full late --output results/fusion
python train.py --dataset wesad --data data/wesad.npz --modalities ECG --output results/ecg_only
```

`no_dmfe` retains convolutional alignment but removes differential extraction; `no_gce` bypasses cross-attention; `no_dwf` uses mean-pooled global features instead of dual-stream attention fusion. Early fusion concatenates aligned features; late fusion averages independently trained unimodal posterior probabilities. All variants within a run use identical outer and inner splits.

## Published results (reference only)

| Dataset / target | Accuracy (%) | F1 (%) |
|---|---:|---:|
| DEAP valence | 94.23 | 94.89 |
| DEAP arousal | 93.91 | 94.92 |
| WESAD | 94.63 | 93.82 |
| CogLoad | 92.08 | 92.91 |
| MOCAS | 92.04 | 91.89 |

Values are from the abstract and Section V. Table 6 instead reports WESAD 94.28/93.98 and CogLoad F1 92.92. The values are retained as reported in their respective locations; no reconciliation is assumed.

### Ablation results (Table 7)

| Configuration | Accuracy (%) | F1 (%) |
|---|---:|---:|
| Without DMFE | 90.08 | 89.68 |
| Without GCE | 86.19 | 86.32 |
| Without DWF | 91.32 | 91.17 |
| Full model | 94.28 | 93.98 |

These are published reference values, not measurements from the present implementation. Benchmark reproduction requires the original data preparation, split indices and experimental settings.

## References and attribution

The article identifies [Husformer](https://github.com/SMARTlab-purdue/Husformer) (reference 43) and *Differential Transformer* (reference 45) as antecedents in its experimental implementation. This standalone reconstruction does not vendor source files from those repositories. Consult their respective licenses before incorporating upstream code.

## Citation

```bibtex
@article{saleem2025diffmt,
  author  = {Saleem, Muhammad and Kim, Deok-Hwan},
  title   = {Multimodal Biosignals for Affective State Recognition Using Differential Multimodal Transformer},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {202395--202407},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3637310}
}
```

## License

The article is published under CC BY 4.0. A separate code license should be selected by the repository maintainers after obtaining any necessary institutional authorization. The article license does not automatically establish a license for the source code.
