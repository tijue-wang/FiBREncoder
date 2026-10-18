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
from .models.Unet_ULMamba import Unet_ULMamba_multi_inputs
from .lib.imageDatastore_multi_inputs import imageDatastore_9_inputs

def main():
    parser = argparse.ArgumentParser(description="Evaluate FiBREncoder reconstruction")
    parser.add_argument("--inputs", default="test_data/expt_MCF_Test/05_PaperTissue")
    parser.add_argument("--targets", default="test_data/HR_Test/05_PaperTissue")
    # Download model weights and place them beside this script.
    parser.add_argument("--weights", default="./fibrencoder_9_input.pth.tar")
    parser.add_argument("--output", default="./reconstruction_results_9")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()
    weights = SCRIPT_DIR / args.weights
    if not weights.is_file():
        raise FileNotFoundError(f"Download model weights before testing: {weights}")
    # Locate paired test images and the output directory.
    inputs, targets = SCRIPT_DIR / args.inputs, SCRIPT_DIR / args.targets
    output = SCRIPT_DIR / args.output
    output.mkdir(parents=True, exist_ok=True)
    model = Unet_ULMamba_multi_inputs(num_inputs=9, c_unet=[32, 64, 128, 256])
    # Load parameters; remove the prefix used by DataParallel checkpoints.
    checkpoint = torch.load(weights, map_location=args.device)
    state = checkpoint["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(args.device).eval()
    # Each sample contains X0-X8 followed by its HR target.
    dataset = imageDatastore_9_inputs(str(targets), *(str(inputs / f"X{i}") for i in range(9)))
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
