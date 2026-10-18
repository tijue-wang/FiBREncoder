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

    from .models.Unet_ULMamba import Unet_ULMamba_multi_inputs

    from .lib.imageDatastore_multi_inputs import imageDatastore_9_inputs
    from .lib.utils import setup_seed, tensor2np, load_checkpoint, save_checkpoint, adjust_learning_rate, compute_psnr, compute_ssim, Timer, time_text

    path = "training_data/"

    # Configure optimization, data paths, and checkpoint options.
    parser = argparse.ArgumentParser(description="net")
    parser.add_argument("--num_epochs",     default=10,  type=int)
    parser.add_argument("--batch_size",     default=16,   type=int)
    parser.add_argument("--val_batch_size", default=16,   type=int)
    parser.add_argument("--lr",             default=0.5e-4, type=float)
    parser.add_argument("--lr_DropFactor",  default=0.5,  type=float)
    parser.add_argument("--lr_DropPeriod",  default=5,   type=int)
    parser.add_argument("--save_everyEpoch",default=1,  type=int,)
    parser.add_argument("--save_path",      default="./saved_models/", type=str,)
    parser.add_argument("--save_model_arch",default=False, type=lambda value: value.lower() in ("true", "1", "yes"), help='save model architecture')
    parser.add_argument("--num_workers",    default=12,    type=int, help='number of subprocesses for data loading')
    parser.add_argument("--num_GPUs",       default=1,     type=int)
    parser.add_argument("--dir_ZTrain",     default= path + "HR_Train_80k", type=str)
    parser.add_argument("--dir_XTrain",     default= path + "sim_MCF_Train_80k", type=str)
    parser.add_argument("--dir_ZValid",     default= path + "HR_Valid", type=str)
    parser.add_argument("--dir_XValid",     default= path + "sim_MCF_Valid", type=str)
    parser.add_argument("--dir_ZValid2",    default= "test_data/HR_Test/05_PaperTissue", type=str)
    parser.add_argument("--dir_XValid2",    default= "test_data/expt_MCF_Test/05_PaperTissue", type=str)
    parser.add_argument("--n_colors",       default=1,    type=int, help="number of color channels to use")
    parser.add_argument("--pretrain",       default=False, type=lambda value: value.lower() in ("true", "1", "yes"), help="load pre-trained model")
    parser.add_argument("--dir_pretrain",   default="./fibrencoder_9_input.pth.tar",   type=str, help="path1 to checkpoint")
    parser.add_argument("--start-epoch",    default=1,    type=int, help="manual epoch number")

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

    # Load HR targets and the nine corresponding simulated fiber inputs.
    dir_XTrain0 = args.dir_XTrain + '/X0'
    dir_XTrain1 = args.dir_XTrain + '/X1'
    dir_XTrain2 = args.dir_XTrain + '/X2'
    dir_XTrain3 = args.dir_XTrain + '/X3'
    dir_XTrain4 = args.dir_XTrain + '/X4'
    dir_XTrain5 = args.dir_XTrain + '/X5'
    dir_XTrain6 = args.dir_XTrain + '/X6'
    dir_XTrain7 = args.dir_XTrain + '/X7'
    dir_XTrain8 = args.dir_XTrain + '/X8'
    dsTrain = imageDatastore_9_inputs(args.dir_ZTrain, dir_XTrain0, dir_XTrain1, dir_XTrain2, dir_XTrain3,
                                          dir_XTrain4, dir_XTrain5, dir_XTrain6, dir_XTrain7, dir_XTrain8,
                                          )
    dir_XValid0 = args.dir_XValid + '/X0'
    dir_XValid1 = args.dir_XValid + '/X1'
    dir_XValid2 = args.dir_XValid + '/X2'
    dir_XValid3 = args.dir_XValid + '/X3'
    dir_XValid4 = args.dir_XValid + '/X4'
    dir_XValid5 = args.dir_XValid + '/X5'
    dir_XValid6 = args.dir_XValid + '/X6'
    dir_XValid7 = args.dir_XValid + '/X7'
    dir_XValid8 = args.dir_XValid + '/X8'
    dsValid = imageDatastore_9_inputs(args.dir_ZValid, dir_XValid0, dir_XValid1, dir_XValid2, dir_XValid3,
                                          dir_XValid4, dir_XValid5, dir_XValid6, dir_XValid7, dir_XValid8)
    if hasattr(args, "dir_ZValid2"):
        dir_XValid20 = args.dir_XValid2 + '/X0'
        dir_XValid21 = args.dir_XValid2 + '/X1'
        dir_XValid22 = args.dir_XValid2 + '/X2'
        dir_XValid23 = args.dir_XValid2 + '/X3'
        dir_XValid24 = args.dir_XValid2 + '/X4'
        dir_XValid25 = args.dir_XValid2 + '/X5'
        dir_XValid26 = args.dir_XValid2 + '/X6'
        dir_XValid27 = args.dir_XValid2 + '/X7'
        dir_XValid28 = args.dir_XValid2 + '/X8'
        dsValid2 = imageDatastore_9_inputs(args.dir_ZValid2, dir_XValid20, dir_XValid21, dir_XValid22, dir_XValid23,
                                              dir_XValid24, dir_XValid25, dir_XValid26, dir_XValid27, dir_XValid28)

    train_loader = DataLoader(dataset=dsTrain, num_workers=args.num_workers, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, drop_last=True, persistent_workers=False)
    valid_loader = DataLoader(dataset=dsValid, num_workers=args.num_workers, batch_size=args.val_batch_size,
                              shuffle=False, pin_memory=True, drop_last=False)
    if hasattr(args, "dir_ZValid2"):
        valid_loader2 = DataLoader(dataset=dsValid2, num_workers=args.num_workers, batch_size=args.val_batch_size,
                                  shuffle=False, pin_memory=True, drop_last=False)

    if args.num_GPUs > torch.cuda.device_count():
        raise ValueError("Requested GPU count exceeds available CUDA devices; use --num_GPUs 0 for CPU.")
    if args.num_GPUs > 0:
        main_device = 0
        device = f"cuda:{main_device}"
        device_ids = list(range(args.num_GPUs)) if args.num_GPUs > 1 else None
    else:
        device = "cpu"

    model = Unet_ULMamba_multi_inputs(num_inputs=9, c_unet=[32,64,128,256])

    model_arch = copy.deepcopy(model) if args.save_model_arch else None
    model = nn.DataParallel(model, device_ids=device_ids, output_device=main_device).to(device) if args.num_GPUs > 1 else model.to(device)

    if args.pretrain:
        checkpoint = torch.load(args.dir_pretrain, map_location=device)
        load_checkpoint(checkpoint, model)

    l1_loss = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Update model parameters from paired batches using the reconstruction loss.
    def train(epoch):
        model.train()
        loss = 0.0

        adjust_learning_rate(optimizer, epoch-1, args.lr, args.lr_DropFactor, args.lr_DropPeriod)

        print("\n------- Epoch {}/{} \t\t lr = {}".format(epoch, args.num_epochs, optimizer.param_groups[0]['lr']))

        tqdm_iterator = tqdm(train_loader, desc="In training", leave=True)

        for ii, tensors in enumerate(tqdm_iterator):
            dlX0,dlX1,dlX2,dlX3,dlX4,dlX5,dlX6,dlX7,dlX8,dlZ = [tensor.to(device, non_blocking=True) for tensor in tensors]

            dlY = model(dlX0,dlX1,dlX2,dlX3,dlX4,dlX5,dlX6,dlX7,dlX8)
            loss_i = l1_loss(dlY, dlZ)

            optimizer.zero_grad(set_to_none=True)
            loss_i.backward()
            optimizer.step()

            tqdm_iterator.set_postfix(loss=loss_i.item())

            loss += loss_i.detach().cpu().item()

        loss /= len(train_loader)
        print("\nTrain loss: {:.4f}".format(loss))
        return loss

    # Evaluate held-out pairs without updating model parameters.
    def valid(valid_loader):
        model.eval()
        loss, loss_l1 = 0.0, 0.0
        peaksnr, ssimval = 0, 0
        count = 0

        for tensors in valid_loader:
            dlVX0,dlVX1,dlVX2,dlVX3,dlXV4,dlXV5,dlXV6,dlXV7,dlXV8,dlVZ = [tensor.to(device, non_blocking=True) for tensor in tensors]

            with torch.no_grad():
                dlVY = model(dlVX0,dlVX1,dlVX2,dlVX3,dlXV4,dlXV5,dlXV6,dlXV7,dlXV8)

                loss += l1_loss(dlVY, dlVZ).detach().cpu().item()

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

    best_validLoss = 1_000_000

    code_start = datetime.datetime.now()
    timer = Timer()

    # Run optimization and validation, and save checkpoints at the configured interval.
    for epoch in range(args.start_epoch, args.num_epochs+1):
        t_epochStart = timer.t()

        loss_train = train(epoch)
        loss_stats['train'].append(loss_train)

        loss_valid  = valid(valid_loader)
        loss_stats['valid'].append(loss_valid)
        if hasattr(args, "dir_ZValid2"):
            loss_valid2 = valid(valid_loader2)
            loss_stats['valid_expt'].append(loss_valid2)

        loss_val = loss_valid2 if hasattr(args, "dir_ZValid2") else loss_valid
        # Keep a separate copy of the parameters with the lowest validation loss.
        if loss_val < best_validLoss:
            best_validLoss = loss_val
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % args.save_everyEpoch == 0:
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
    plt.plot(loss_stats['valid'], label="Valid")
    plt.plot(loss_stats['valid_expt'], label="Valid expt") if hasattr(args, "dir_ZValid2") else None
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
