from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    __package__ = SCRIPT_DIR.name

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from .lib.utils import compute_psnr, compute_ssim
# from custom_colormaps import get_colormap
from scipy.io import loadmat
from .PropagaModels_exp_v2_2 import PropagaModel_MC
from .lib.imageDatastore import imageDatastore

def main():
    parser = argparse.ArgumentParser(description="Evaluate FiBRE simulation fidelity")
    parser.add_argument("--inputs", default="test_images/HR_Test/05_PaperTissue_X0")
    parser.add_argument("--targets", default="test_images/MCF_Test/05_PaperTissue_X0")
    # Download model weights and place them beside this script.
    parser.add_argument("--weights", default="./fibre_calibrated.pth.tar")
    parser.add_argument("--output", default="./simulation_results")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--core_data", default="corePos_1440_25_light.mat")
    parser.add_argument("--mode", default="base_sc_g", choices=PropagaModel_MC.VALID_MODES)
    args = parser.parse_args()
    weights = SCRIPT_DIR / args.weights
    if not weights.is_file():
        raise FileNotFoundError(f"Download model weights before testing: {weights}")
    # Locate paired test images and the output directory.
    inputs, targets = SCRIPT_DIR / args.inputs, SCRIPT_DIR / args.targets
    output = SCRIPT_DIR / args.output
    output.mkdir(parents=True, exist_ok=True)
    core = loadmat(SCRIPT_DIR / args.core_data)
    model = PropagaModel_MC(core['corePos'] - 1, core['core_global'],
                           core['coreMasksOut_local'].astype(bool), mode=args.mode,
                           z=0, inScale=3000, pad_size=30, device=args.device)
    # Load parameters; remove the prefix used by DataParallel checkpoints.
    checkpoint = torch.load(weights, map_location=args.device)
    state = checkpoint["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(args.device).eval()
    dataset = imageDatastore(str(targets), str(inputs), n_colors=1)
    if len(dataset) == 0:
        raise ValueError("The test dataset is empty.")
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    rows = []
    # Run inference without gradients and measure model execution time.
    with torch.no_grad():
        for index, tensors in enumerate(loader):
            *images, target = tensors
            images = [image.to(args.device) for image in images]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            prediction = model(*images)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter() - start
            # Compare each prediction with its target and save the predicted image.
            reference = target[0, 0].numpy().clip(0, 1)
            result = prediction[0, 0].cpu().numpy().clip(0, 1)
            if result.shape != reference.shape:
                raise ValueError(f"Prediction and target shapes differ: {result.shape}, {reference.shape}")
            rows.append([compute_psnr(reference, result), compute_ssim(reference, result),
                         float(np.abs(reference - result).mean()), elapsed])
            Image.fromarray((result * 255).round().astype(np.uint8)).save(output / f"prediction_{index:04d}.png")
    # Export per-image fidelity metrics and report their averages.
    rows = np.asarray(rows)
    np.savetxt(output / "metrics.csv", rows, delimiter=",", header="psnr,ssim,mae,seconds", comments="")
    print(f"Images: {len(rows)} | PSNR: {rows[:, 0].mean():.4f} | SSIM: {rows[:, 1].mean():.4f} | MAE: {rows[:, 2].mean():.6f}")
    print(f"Results saved to {output}")

if __name__ == "__main__":
    main()
