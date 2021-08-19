import os

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from tensorfn import load_arg_config, load_wandb
from score.F_beta import calculate_f_beta_score,plot_pr_curve,calculate_f_beta_score_animeface
from tensorfn import distributed as dist
from tensorfn.optim import lr_scheduler
from tqdm import tqdm
from pathlib import Path
import time
import logging
from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from dataset import MultiResolutionDataset
from config import DiffusionConfig


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def train(conf, results_dir, loader, test_loader,model, ema, diffusion, optimizer, scheduler, device, wandb):
    loader = sample_data(loader)
    best_fid = 1000
    best_fid_index =0
    pbar = range(conf.training.n_iter + 1)

    if dist.is_primary():
        pbar = tqdm(pbar, dynamic_ncols=True)

    for i in pbar:
        epoch, img = next(loader)
        img = img.to(device)
        time = torch.randint(
            0, conf.diffusion.beta_schedule.n_timestep, (img.shape[0],), device=device
        )
        loss = diffusion.p_loss(model, img, time)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"epoch: {epoch}; loss: {loss.item():.4f}; lr: {lr:.5f}"
            )

            if i % conf.evaluate.log_every == 0:
                wandb.log({"epoch": epoch, "loss": loss.item(), "lr": lr}, step=i)

            if i % conf.evaluate.save_every == 0:
                img=diffusion.p_sample_loop(ema, img.shape, 'cuda')
                save_image((img.detach().cpu()+1)/2, os.path.join(results_dir,'generated_img.png'), nrow=8)
                fid_cache = '/home/congen/code/AGE-exp/datasets/tf_fid_stats_cifar10_32.npz'
                n_generate = 1000
                num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
                is_scores, fid_score, precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(diffusion,test_loader,
                                                                    ema, n_generate, num_run4PR, num_cluster4PR,
                                                                    beta4PR, num_split, fid_cache,device)
                wandb.define_metric("FID", summary="min")
                wandb.define_metric("IS", summary="max")
                wandb.define_metric("F8", summary="max")
                wandb.define_metric("F1/8", summary="max")
                wandb.log({"FID": fid_score.item(), "IS": is_scores.item(),
                           "F8": f_beta.item(), "F1/8": f_beta_inv.item()}, step=i)
                if i % 50000 ==0:
                    if conf.distributed:
                        model_module = model.module

                    else:
                        model_module = model

                    torch.save(
                        {
                            "model": model_module.state_dict(),
                            "ema": ema.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "conf": conf,
                        },
                        f"{results_dir}/diffusion_{i}.pt",
                    )
                if (fid_score < best_fid):
                    wandb.run.summary["best_fid"] = fid_score
                    wandb.run.summary["best_fid_step"] = i
                    best_fid = fid_score
                    best_fid_index = i
                    if conf.distributed:
                        model_module = model.module

                    else:
                        model_module = model

                    torch.save(
                        {
                            "model": model_module.state_dict(),
                            "ema": ema.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "conf": conf,
                        },
                        f"{results_dir}/diffusion_best.pt",
                    )
                wandb.log({"best_FID": best_fid.item(),"best_FID_index": best_fid_index}, step=i)


def main(conf):
    wandb = None
    if dist.is_primary() and conf.evaluate.wandb:
        wandb = load_wandb()
        wandb.init(project="denoising diffusion2")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    beta_schedule = "linear"
    results_folder = './results_{}/{}'.format(conf.dataset.name, int(time.time()))
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True,exist_ok=True)
    conf.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [   transforms.Resize(conf.dataset.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.Resize(conf.dataset.resolution),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
         ]
    )

    train_set = MultiResolutionDataset(
        conf.dataset.path, transform, train=True,dataset=conf.dataset.name,resolution=conf.dataset.resolution
    )
    test_set = MultiResolutionDataset(
        conf.dataset.path, test_transform, train=False,dataset=conf.dataset.name,resolution=conf.dataset.resolution)
    train_sampler = dist.data_sampler(
        train_set, shuffle=True, distributed=conf.distributed
    )
    test_sampler = dist.data_sampler(
        test_set, shuffle=False, distributed=conf.distributed
    )
    train_loader = conf.training.dataloader.make(train_set, sampler=train_sampler)
    test_loader = data.DataLoader(test_set, sampler=test_sampler,batch_size=conf.training.dataloader.batch_size,
                                  shuffle=False, pin_memory=True,drop_last=False)

    model = conf.model.make()
    model = model.to(device)
    ema = conf.model.make()
    ema = ema.to(device)

    if conf.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = conf.training.optimizer.make(model.parameters())
    scheduler = conf.training.scheduler.make(optimizer)

    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)

    train(
        conf, results_folder,train_loader,test_loader, model, ema, diffusion, optimizer, scheduler, device, wandb
    )

def test(conf):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    beta_schedule = "linear"
    results_folder = './results_{}/eval_{}'.format(conf.dataset.name, int(time.time()))
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    conf.distributed = dist.get_world_size() > 1

    test_transform = transforms.Compose(
        [transforms.Resize(conf.dataset.resolution),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
         ]
    )

    test_set = MultiResolutionDataset(
        conf.dataset.path, test_transform, train=False, dataset=conf.dataset.name, resolution=conf.dataset.resolution)

    test_sampler = dist.data_sampler(
        test_set, shuffle=False, distributed=conf.distributed
    )

    test_loader = data.DataLoader(test_set, sampler=test_sampler, batch_size=conf.training.dataloader.batch_size,
                                  shuffle=False, pin_memory=True, drop_last=False)
    ckpt = torch.load('./results_cifar10/1629313170/diffusion_best.pt')
    model = conf.model.make()
    model.load_state_dict(ckpt['ema'])
    model = model.to('cuda')
    model.eval()
    # if conf.distributed:
    #     model = nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[dist.get_local_rank()],
    #         output_device=dist.get_local_rank(),
    #     )


    betas = conf.diffusion.beta_schedule.make()
    diffusion = GaussianDiffusion(betas).to(device)
    img = diffusion.p_sample_loop(model, [64, 3, conf.dataset.resolution, conf.dataset.resolution], 'cuda')
    save_image((img.detach().cpu() + 1) / 2, os.path.join(results_folder, 'generated_img.png'), nrow=8)
    fid_cache = '/home/congen/code/AGE-exp/datasets/tf_fid_stats_cifar10_32.npz'
    n_generate = len(test_loader.dataset)
    num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
    is_scores, fid_score, precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(diffusion,
                                                                                         test_loader,
                                                                                         model, n_generate,
                                                                                         num_run4PR,
                                                                                         num_cluster4PR,
                                                                                         beta4PR, num_split,
                                                                                         fid_cache, device)
    print('IS', is_scores)
    print('FID', fid_score)
    print('F8', f_beta)
    print('F1/8', f_beta_inv)



"""
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        python train.py --conf ./config/diffusion_cifar10.conf


    :return:
    """
if __name__ == "__main__":
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
