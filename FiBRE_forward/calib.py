from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    __package__ = SCRIPT_DIR.name

def main():

    import datetime
    import argparse
    import sys
    from tqdm import tqdm
    from scipy.io import loadmat
    from matplotlib import pyplot as plt

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .lib.imageDatastore import imageDatastore
    from .lib.utils import (
        setup_seed,
        tensor2np,
        load_checkpoint,
        save_checkpoint,
        adjust_learning_rate,
        compute_psnr,
        compute_ssim,
        Timer,
        time_text,
        save_kernels_to_h5,
        save_valid_preds_h5,
        )

    path = "calibration_data/"

    # Configure optimization, data paths, and checkpoint options.
    parser = argparse.ArgumentParser(description="net")
    parser.add_argument("--num_epochs",     default=100,  type=int)
    parser.add_argument("--batch_size",     default=24,   type=int)
    parser.add_argument("--val_batch_size", default=24,   type=int)
    parser.add_argument("--lr",             default=0.5e-4, type=float)
    parser.add_argument("--lr_DropPeriod",  default=25,   type=int)
    parser.add_argument("--lr_DropFactor",  default=0.5,  type=float)
    parser.add_argument("--save_everyEpoch",default=5,  type=int,)
    parser.add_argument("--save_path",      default="./saved_models/", type=str,)
    parser.add_argument("--save_model_arch",default=False, type=lambda value: value.lower() in ("true", "1", "yes"), help='save model architecture')
    parser.add_argument("--num_workers",    default=12,    type=int)
    parser.add_argument('--dir_dataset',    default=path, help = 'dataset path')
    parser.add_argument("--dir_ZCalib",     default= path + "MCF_Calib/", type=str)
    parser.add_argument("--dir_XCalib",     default= path + "HR_Calib/", type=str)
    parser.add_argument("--dir_ZValid",     default= path + "MCF_Valid/small/", type=str)
    parser.add_argument("--dir_XValid",     default= path + "HR_Valid/small/", type=str)
    parser.add_argument("--n_colors",       default=1,    type=int, help="number of color channels to use")
    parser.add_argument("--precalib",       default=False, type=lambda value: value.lower() in ("true", "1", "yes"), help="load previously calibrated model")
    parser.add_argument("--dir_precalib",   default="./fibre_calibrated.pth.tar",   type=str, help="path to checkpoint")
    parser.add_argument("--start-epoch",    default=1,    type=int, help="manual epoch number")
    parser.add_argument("--SAVE_KERNELS",    default=False, type=lambda value: value.lower() in ("true", "1", "yes"),)
    parser.add_argument("--SAVE_KERNELS_EVERY_EPOCH", default=1,  type=int,)
    parser.add_argument("--SAVE_KERNELS_DIR",default="./00_saved_model_tmp/kernels/",  type=str,)
    parser.add_argument("--SAVE_VALID",     default=False, type=lambda value: value.lower() in ("true", "1", "yes"),)
    parser.add_argument("--SAVE_VALID_EVERY_EPOCH", default=1,  type=int,)
    parser.add_argument("--SAVE_VALID_DIR", default="./00_saved_model_tmp/saved_valid_preds/",  type=str,)

    parser.add_argument("--rgb_range",      default=1,    type=int, help="maxium value of RGB")
    parser.add_argument("--seed",           default=1,    type=int)
    parser.add_argument("--ext",            default='.png', type=str)
    args = parser.parse_args()
    # Resolve relative paths against this script directory.
    for name, value in vars(args).items():
        if isinstance(value, str) and (name.startswith("dir_") or name in ("save_path", "SAVE_KERNELS_DIR", "SAVE_VALID_DIR")):
            setattr(args, name, str(SCRIPT_DIR / value))
    args.save_path = str(Path(args.save_path)) + "/"
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    # Download model weights before enabling initialization from calibrated weights.

    setup_seed(args.seed)

    # Load paired HR and measured fiber images for parameter calibration.
    dsCalib = imageDatastore(args.dir_ZCalib, args.dir_XCalib)
    dsValid = imageDatastore(args.dir_ZValid, args.dir_XValid)

    calib_loader = DataLoader(dataset=dsCalib, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(dataset=dsValid, num_workers=args.num_workers, batch_size=args.val_batch_size,
                              shuffle=False, pin_memory=True, drop_last=False)

    # Load fiber geometry and masks used by the forward model.
    coreData = loadmat(SCRIPT_DIR / 'corePos_1440_25_light.mat')
    inScale = 3000

    corePos = coreData['corePos'] - 1
    core_global = coreData['core_global']
    coreMasks_local = coreData['coreMasksOut_local'].astype(bool)
    num_core = corePos.shape[0]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    from .PropagaModels_exp_v2_2 import PropagaModel_MC
    model = PropagaModel_MC(corePos, core_global, coreMasks_local, mode="base_sc_g", z=0, inScale=inScale, pad_size=30, device=device)

    import copy
    model_arch = copy.deepcopy(model)
    model = model.to(device)

    if args.precalib:
        checkpoint = torch.load(args.dir_precalib, map_location=device)
        load_checkpoint(checkpoint, model)

    l1_loss = nn.L1Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Update model parameters from paired batches using the reconstruction loss.
    def calib(epoch):
        model.train()
        adjust_learning_rate(optimizer, epoch-1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)
        loss = 0.0

        print("\n------- Epoch {}/{} \t\t lr = {}".format(epoch, args.num_epochs, optimizer.param_groups[0]['lr']))

        tqdm_iterator = tqdm(calib_loader, desc="In calibration", leave=True)

        for ii, (dlX, dlZ) in enumerate(tqdm_iterator):
            dlX, dlZ = dlX.to(device), dlZ.to(device)

            dlY = model(dlX)

            loss_i = l1_loss(dlY, dlZ)

            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()

            for iCore in range(num_core):
                module = model.conv1_w[iCore]
                values = module.data
                module.data = values.clamp(0.0, 1.0)

                module = model.conv2_w[iCore]
                values = module.data
                module.data = values.clamp(0.0, 1.0)

            loss_i = loss_i.item()
            tqdm_iterator.set_postfix({'loss_i':loss_i})

            loss += loss_i

        loss /= len(calib_loader)
        print("\nCalib loss: {:.8f}".format(loss))
        return loss

    # Evaluate held-out pairs without updating model parameters.
    def valid(epoch=None, save_pred=False, save_dir=None):
        model.eval()
        loss = 0.0
        peaksnr, ssimval = 0, 0
        count = 0

        pred_list = []

        for (dlVX, dlVZ) in valid_loader:
            dlVX, dlVZ = dlVX.to(device), dlVZ.to(device)

            with torch.no_grad():
                dlVY = model(dlVX)

                loss += l1_loss(dlVY, dlVZ).detach().cpu().item()

            batch_val = dlVZ.shape[0]
            count += batch_val
            p, s = 0, 0
            for jj in range(batch_val):
                VY = tensor2np(dlVY.detach()[jj])
                VZ = tensor2np(dlVZ.detach()[jj])

                peaksnr += compute_psnr(VY, VZ)
                ssimval += compute_ssim(VY, VZ)

                if save_pred:
                    pred_list.append(VY)

        loss /= len(valid_loader)
        peaksnr /= count
        ssimval /= count
        print("Valid loss: {:.8f} \t\t PSNR: {:.4f} \t\t SSIM: {:.4f}".format(loss, peaksnr, ssimval))

        if save_pred:
            save_valid_preds_h5(pred_list, epoch=epoch, save_dir=save_dir, psnr=peaksnr, ssim=ssimval, loss=loss)

        return loss

    print("===> Calibration")

    if args.precalib:
        loss_stats = {'calib':checkpoint['loss_stats']['calib'], 'valid':checkpoint['loss_stats']['valid']}
    else:
        loss_stats = {'calib': [], 'valid': []}

    best_validLoss = 1_000_000

    code_start = datetime.datetime.now()
    timer = Timer()

    # Run optimization and validation, and save checkpoints at the configured interval.
    for epoch in range(args.start_epoch, args.num_epochs+1):
        t_epochStart = timer.t()

        if args.SAVE_VALID and (epoch == 1):
            loss_valid = valid(epoch=epoch-1, save_pred=True, save_dir=args.SAVE_VALID_DIR)
        if args.SAVE_KERNELS and (epoch == 1):
            save_kernels_to_h5(model, epoch-1, save_dir=args.SAVE_KERNELS_DIR, prefix='kernels')

        loss_calib = calib(epoch)

        if args.SAVE_VALID and (epoch % args.SAVE_VALID_EVERY_EPOCH == 0):
            loss_valid = valid(epoch=epoch, save_pred=True, save_dir=args.SAVE_VALID_DIR)
        else:
            loss_valid = valid()

        loss_stats['calib'].append(loss_calib)
        loss_stats['valid'].append(loss_valid)

        # Keep a separate copy of the parameters with the lowest validation loss.
        if loss_valid < best_validLoss:
            best_validLoss = loss_valid
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % args.save_everyEpoch == 0:
            save_checkpoint(model, model_arch, optimizer, epoch, loss_stats, args)

            model_best = copy.deepcopy(model)
            model_best.load_state_dict(best_state_dict)
            s = f'_best_' + str(best_epoch)
            save_checkpoint(model_best, model_arch, optimizer, epoch, loss_stats, args, text=s)

        if args.SAVE_KERNELS and (epoch % args.SAVE_KERNELS_EVERY_EPOCH == 0):
            save_kernels_to_h5(model, epoch, save_dir=args.SAVE_KERNELS_DIR, prefix='kernels')

        t_epochEnd = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.num_epochs - args.start_epoch + 1)
        t_epoch = time_text(t_epochEnd - t_epochStart)
        t_elapsed, t_all = time_text(t_epochEnd), time_text(t_epochEnd / prog)
        print('Time cost: {}, {}/{}'.format(t_epoch, t_elapsed, t_all))

    code_end = datetime.datetime.now()
    print('Total cost times: %s' % str(code_end - code_start).split('.', 2)[0])

    plt.figure('Calibration progress',figsize=(10,5))
    plt.title("Calibration progress")
    plt.plot(loss_stats['calib'], label="Calib")
    plt.plot(loss_stats['valid'], label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
