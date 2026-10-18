# FiBRE and FiBREncoder

FiBRE models image formation through a multi-core fiber. FiBREncoder learns image reconstruction from training pairs generated in silico using a calibrated FiBRE forward model.

## Workflow

1. Calibrate FiBRE parameters against paired calibration images using `FiBRE_forward/calib.py`.
2. Evaluate simulation fidelity against measured fiber images using `FiBRE_forward/test.py`.
3. Use FiBRE with calibrated parameters to generate reconstruction training pairs offline. The forward inference principle is the same as in `FiBRE_forward/test.py`; a separate data-generation script is not included.
4. Train a single-input or nine-input reconstruction network using `FiBREncoder_in_silico/train.py` or `train_9_input.py`.
5. Evaluate reconstruction using `test.py` or `test_9_input.py`.

## Environment

Target environment: **Python 3.10**, **PyTorch 2.0.1**, and **torchvision 0.15.2**.

See [requirements.md](requirements.md) for the complete dependency list and installation commands, including the Mamba dependencies for both reconstruction networks.

## Repository layout

```text
FiBRE_forward/
  PropagaModels_exp_v2_2.py
  calib.py
  test.py
  corePos_1440_25_light.mat
  lib/
  models/
  test_images/
    HR_Test/05_PaperTissue_X0/
    MCF_Test/05_PaperTissue_X0/
FiBREncoder_in_silico/
  train.py
  train_9_input.py
  test.py
  test_9_input.py
  lib/
  models/
  test_data/
    HR_Test/05_PaperTissue/
    expt_MCF_Test/05_PaperTissue/X0/ ... X8/
```

All relative data, output, and checkpoint paths resolve against the directory containing the corresponding script, independent of the working directory. Internal imports use relative package imports. Direct script execution and `python -m` from the repository root are supported.

The included data contain one paired paper-tissue example and nine corresponding fiber inputs, each 1440 × 1440 pixels. Calibration and reconstruction training sets are not included. Within each paired dataset, images must be in matching sorted order, with equal image counts. Nine-input directories must contain aligned acquisitions of the same sample in `X0`–`X8`.

## Model weights and test data

Model weights and test data are available on Zenodo: [10.5281/zenodo.23457475](https://doi.org/10.5281/zenodo.23457475). Model weights are not included in this repository. FiBREncoder training and testing use U-Net–Mamba reconstruction models. Select a released checkpoint matching the input count and architecture.

Extract the test data into `FiBRE_forward/test_images/` and `FiBREncoder_in_silico/test_data/`, preserving the directory structure shown above.

Place each checkpoint beside the corresponding script. The filenames below are local defaults expected by the scripts; use `--weights` to select a downloaded or locally trained checkpoint with a different filename:

| Model | Default local checkpoint | Notes |
| --- | --- | --- |
| FiBRE, `PropagaModel_MC`, `base_sc_g` | `FiBRE_forward/fibre_calibrated.pth.tar` | Requires calibrated forward-model parameters. |
| Single-input FiBREncoder, `Unet_ULMamba` | `FiBREncoder_in_silico/fibrencoder_1_input.pth.tar` | Requires a matching single-input U-Net–Mamba checkpoint. |
| Nine-input FiBREncoder, `Unet_ULMamba_multi_inputs` | `FiBREncoder_in_silico/fibrencoder_9_input.pth.tar` | U-Net–Mamba reconstruction weights are provided on Zenodo; select a checkpoint matching this architecture. |

Checkpoints must contain a `state_dict` matching the selected architecture and, for FiBRE, the core geometry and mode. Test scripts accept `--weights` to select another filename. Calibration and reconstruction training start from scratch by default. FiBRE calibration uses `--precalib true --dir_precalib ./checkpoint.pth.tar` to load previously calibrated weights; FiBREncoder uses `--pretrain true --dir_pretrain ./checkpoint.pth.tar` to load initial model weights. This option does not restore the optimizer state.

## FiBRE parameter calibration

Prepare the following directories inside `FiBRE_forward/`, or override their paths with the corresponding command-line arguments:

```text
calibration_data/
  HR_Calib/
  MCF_Calib/
  HR_Valid/small/
  MCF_Valid/small/
```

HR images are inputs; measured MCF images are calibration targets. Override calibration paths with `--dir_XCalib` and `--dir_ZCalib`. The default core geometry is `corePos_1440_25_light.mat`. The default model is `PropagaModels_exp_v2_2.PropagaModel_MC` with mode `base_sc_g`.

```bash
python FiBRE_forward/calib.py
python FiBRE_forward/test.py
```

Calibration checkpoints are written to `FiBRE_forward/saved_models/`. Pass a selected checkpoint to testing, for example:

```bash
python FiBRE_forward/test.py --weights saved_models/net_epoch_100.pth.tar
```

## FiBREncoder training

Generate paired training data offline with calibrated FiBRE. Place them inside `FiBREncoder_in_silico/`:

```text
training_data/
  HR_Train_80k/
  HR_Valid/
  sim_MCF_Train_80k/
    X0/ ... X8/
  sim_MCF_Valid/
    X0/ ... X8/
```

Single-input `train.py` trains U-Net–Mamba from `X0`; nine-input `train_9_input.py` trains its multi-input variant from all nine input directories. `HR_Train_80k` is the default folder name; paths may be overridden for datasets of other sizes. Nine-input training additionally evaluates the supplied experimental paper-tissue example; this example is used for validation/model selection and is not an independent held-out test set.

```bash
python FiBREncoder_in_silico/train.py --num_GPUs 1
python FiBREncoder_in_silico/train_9_input.py --num_GPUs 1 --save_path ./saved_models_9/
```

Both single-input and nine-input models require the Mamba CUDA dependencies and a compatible NVIDIA GPU. Checkpoints are written to `saved_models/` by default; use different `--save_path` directories for separate runs.

## Reconstruction and simulation testing

```bash
python FiBRE_forward/test.py
python FiBREncoder_in_silico/test_9_input.py
```

For single-input reconstruction, use a matching single-input U-Net–Mamba checkpoint:

```bash
python FiBREncoder_in_silico/test.py --weights saved_models/net_epoch_10.pth.tar
```

The defaults point to the supplied test images. Each script saves reconstructed/simulated grayscale PNG images and `metrics.csv` containing per-image PSNR, SSIM, MAE, and inference time. Predictions are clipped to [0, 1] for evaluation. No external custom-colormap module is required.

Use `--inputs`, `--targets`, `--weights`, `--output`, `--device`, and `--num_workers` to override testing settings. For nine-input testing, `--inputs` points to the parent directory containing `X0`–`X8`. FiBRE testing also accepts `--core_data` and `--mode`.
