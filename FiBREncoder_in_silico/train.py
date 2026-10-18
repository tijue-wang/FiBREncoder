from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    __package__ = SCRIPT_DIR.name

def main():

    import datetime
    import argparse
    import copy
    from tqdm import tqdm
    from matplotlib import pyplot as plt

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from .lib.imageDatastore_augment import imageDatastore_augment
    from .lib.utils import setup_seed, tensor2np, load_checkpoint, save_checkpoint, adjust_learning_rate, compute_psnr, compute_ssim, Timer, time_text

    path = "training_data/"

    # Configure optimization, data paths, and checkpoint options.
    parser = argparse.ArgumentParser(description="net")
    parser.add_argument("--num_epochs",     default=10,  type=int)
    parser.add_argument("--batch_size",     default=16,   type=int)
    parser.add_argument("--val_batch_size", default=16,   type=int)
    parser.add_argument("--lr",             default=1e-4, type=float)
    parser.add_argument("--lr_DropFactor",  default=0.5, type=float)
    parser.add_argument("--lr_DropPeriod",  default=5,  type=int)
    parser.add_argument("--save_everyEpoch",default=1,  type=int,)
    parser.add_argument("--save_path",      default="./saved_models/", type=str,)
    parser.add_argument('--valid_everyEpoch', default=1, type=int)
    parser.add_argument("--num_workers",    default=12,  type=int)
    parser.add_argument("--num_GPUs",       default=1,   type=int)
    parser.add_argument("--dir_ZTrain",     default= path + "HR_Train_80k", type=str)
    parser.add_argument("--dir_XTrain",     default= path + "sim_MCF_Train_80k/X0", type=str)
    parser.add_argument("--dir_ZValid",     default= path + "HR_Valid", type=str)
    parser.add_argument("--dir_XValid",     default= path + "sim_MCF_Valid/X0", type=str)
    parser.add_argument("--n_colors",       default=1,    type=int, help="number of color channels to use")
    parser.add_argument("--pretrain",       default=False, type=lambda value: value.lower() in ("true", "1", "yes"), help="load pre-trained model")
    parser.add_argument("--dir_pretrain",   default="./fibrencoder_1_input.pth.tar",   type=str, help="path1 to checkpoint")
    parser.add_argument("--start-epoch",    default=1,    type=int, help="manual epoch number")

    parser.add_argument("--save_model_arch",default=False, type=lambda value: value.lower() in ("true", "1", "yes"), help='save model architecture')
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
    # Download model weights before enabling pretrained initialization.

    setup_seed(args.seed)

    # Load HR targets and single-input fiber images generated offline by FiBRE.
    dsTrain = imageDatastore_augment(args.dir_ZTrain, args.dir_XTrain,
                                        patches=True, patch_size=256,
                                     )

    dsValid = imageDatastore_augment(args.dir_ZValid, args.dir_XValid, n_colors=args.n_colors,
                                      patches=True, patch_size=256,
                                    )

    train_loader = DataLoader(dataset=dsTrain, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True, persistent_workers=False)
    valid_loader = DataLoader(dataset=dsValid, num_workers=args.num_workers, batch_size=args.val_batch_size,
                              shuffle=False, pin_memory=True, drop_last=False)

    if args.num_GPUs > torch.cuda.device_count():
        raise ValueError("Requested GPU count exceeds available CUDA devices; use --num_GPUs 0 for CPU.")
    if args.num_GPUs > 0:
        main_device = 0
        device = f"cuda:{main_device}"
        device_ids = list(range(args.num_GPUs)) if args.num_GPUs > 1 else None
    else:
        device = "cpu"

    from .models.model_UNET_EDSR import UNET_EDSR
    model = UNET_EDSR(in_channels=1, out_channels=1, features=[32,64,128,256], scale=1, n_resblocks=32, n_feats=256,
                      inplace=False)

    model_arch = copy.deepcopy(model)
    model = model.to(device)

    if args.pretrain:
        checkpoint = torch.load(args.dir_pretrain, map_location=device)
        load_checkpoint(checkpoint, model)

    l1_criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pespective = False
    if pespective:
        align_layer = model.align
        optimizer_align = torch.optim.Adam([
            {'params': align_layer.affine_delta, 'lr': args.lr,       'weight_decay': 0},
            {'params': align_layer.persp_delta,  'lr': args.lr * 0.2, 'weight_decay': 0},])
        optimizer = torch.optim.Adam([{'params': model.unet.parameters(),    'lr': args.lr},
                                      {'params': model.ulmamba.parameters(), 'lr': args.lr},])
        from .lib.utils import HomographyStageScheduler
        scheduler = HomographyStageScheduler(align_layer, total_epochs=args.num_epochs, stage_ratios=(0.4,0.4,0.2))

    model = nn.DataParallel(model, device_ids=device_ids, output_device=main_device).to(device) if args.num_GPUs > 1 else model.to(device)

    # Update model parameters from paired batches using the reconstruction loss.
    def train(epoch):
        model.train()
        loss = 0.0

        adjust_learning_rate(optimizer, epoch-1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)

        if pespective:
            if scheduler.step(epoch): print(f">>> {scheduler.stage_name}")

        print("\n------- Epoch {}/{} \t\t lr = {}".format(epoch, args.num_epochs, optimizer.param_groups[0]['lr']))

        tqdm_iterator = tqdm(train_loader, desc="In training", leave=True, ncols=80)

        for ii, (dlX, dlZ) in enumerate(tqdm_iterator):
            dlX, dlZ = dlX.to(device), dlZ.float().to(device)

            dlY = model(dlX)
            loss_iter = l1_criterion(dlY, dlZ)

            if pespective: optimizer_align.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            loss_iter.backward()
            if pespective:
                torch.nn.utils.clip_grad_norm_([align_layer.persp_delta], max_norm=0.01)
                optimizer_align.step()
            optimizer.step()

            tqdm_iterator.set_postfix(loss=loss_iter.item())

            loss += loss_iter.detach().cpu().item()

        loss /= len(train_loader)
        print("\nTrain loss: {:.4f}".format(loss))
        return loss

    # Evaluate held-out pairs without updating model parameters.
    def valid():
        model.eval()
        loss = 0.0
        peaksnr, ssimval = 0, 0
        count = 0

        for (dlVX, dlVZ) in valid_loader:
            dlVX = dlVX.to(device)
            dlVZ = dlVZ.to(device)

            with torch.no_grad():
                dlVY = model(dlVX)
                loss += l1_criterion(dlVY, dlVZ).detach().cpu().item()

            batch_val = dlVZ.shape[0]
            count += batch_val
            for jj in range(batch_val):
                VY = tensor2np(dlVY.detach()[jj])
                VZ = tensor2np(dlVZ.detach()[jj])

                peaksnr += compute_psnr(VY, VZ)
                ssimval += compute_ssim(VY, VZ)

        loss /= len(valid_loader)
        peaksnr /= count
        ssimval /= count
        print("Valid loss: {:.4f} \t\t PSNR: {:.4f} \t\t SSIM: {:.4f}".format(loss, peaksnr, ssimval))

        return loss

    print("===> Training")

    if args.pretrain:
        loss_stats = {k: v for k, v in checkpoint['loss_stats'].items()}
    else:
        loss_stats = {'train': [], 'valid': [], 'valid_expt': []}

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = args.start_epoch
    best_validLoss = 1_000_000

    code_start = datetime.datetime.now()
    timer = Timer()

    # Run optimization and validation, and save checkpoints at the configured interval.
    for epoch in range(args.start_epoch, args.num_epochs+1):
        t_epochStart = timer.t()

        loss_train = train(epoch)
        loss_stats['train'].append(loss_train)

        do_valid = (epoch % args.valid_everyEpoch == 0) or (epoch == args.num_epochs)

        if do_valid:
            loss_valid = valid()
            loss_stats['valid'].append(loss_valid)

            # Keep a separate copy of the parameters with the lowest validation loss.
            if loss_valid < best_validLoss:
                best_validLoss = loss_valid
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        if pespective:
            m = align_layer.get_matrix().detach().cpu().numpy()
            print(m)
            import numpy as np
            np.save(Path(args.save_path) / ('learned_perspective2d_matrix_ep' + str(epoch) + '.npy'), m)

        if epoch % args.save_everyEpoch == 0 or epoch == args.num_epochs:
            save_checkpoint(model, model_arch, optimizer, epoch, loss_stats, args)

            model_best = copy.deepcopy(model)
            model_best.load_state_dict(best_state_dict)
            s = f'_best_' + str(best_epoch)
            save_checkpoint(model_best, model_arch, optimizer, epoch, loss_stats, args, text=s)

        t_epochEnd = timer.t()
        prog = (epoch - args.start_epoch + 1) / (args.num_epochs - args.start_epoch + 1)
        t_epoch = time_text(t_epochEnd - t_epochStart)
        t_elapsed, t_all = time_text(t_epochEnd), time_text(t_epochEnd / prog)
        print('Time cost: {}, {}/{}'.format(t_epoch, t_elapsed, t_all))

    code_end = datetime.datetime.now()
    print('Total cost times: %s' % str(code_end - code_start).split('.', 2)[0])

    plt.figure('Training progress',figsize=(10,5))
    plt.title("Training progress")
    plt.plot(loss_stats['train'], label="Train")
    plt.plot(range(0, len(loss_stats['valid']) * args.valid_everyEpoch, args.valid_everyEpoch),
             loss_stats['valid'], label="Valid")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
