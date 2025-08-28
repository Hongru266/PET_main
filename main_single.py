import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import wandb

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)  # 减少 batch size
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--mask_loss_coef', default=1.0, type=float,
                        help="Mask loss coefficient")
    parser.add_argument('--bce_loss_coef', default=1.0, type=float,
                        help="Binary Cross Entropy loss coefficient")
    parser.add_argument('--smoothl1_loss_coef', default=1.0, type=float,
                        help="Dice loss coefficient")
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--syn_bn', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--force_single_gpu', action='store_true', help='Force single GPU training')
    
    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', default='PET-crowd-counting', type=str, help='wandb project name')
    parser.add_argument('--wandb_entity', default=None, type=str, help='wandb entity/team name')
    parser.add_argument('--wandb_name', default=None, type=str, help='wandb run name')
    
    return parser


def main(args):
    # 强制禁用分布式训练
    if args.force_single_gpu or args.world_size == 1:
        print('Force single GPU mode')
        args.distributed = False
        args.rank = 0
        args.gpu = 0
        args.world_size = 1
    else:
        utils.init_distributed_mode(args)
    
    print(args)
    device = torch.device(args.device)
    
    # 检查 CUDA 环境
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"PyTorch version: {torch.__version__}")
        
        # 禁用 cuDNN 以避免 CUDNN_STATUS_EXECUTION_FAILED 错误
        torch.backends.cudnn.enabled = False  # 完全禁用 cuDNN
        torch.backends.cudnn.benchmark = False  # 禁用 benchmark 模式
        torch.backends.cudnn.deterministic = True  # 启用确定性模式
        print("cuDNN has been disabled")
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        
        # 检查 GPU 内存
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"GPU memory: {gpu_memory:.2f} GB")
    else:
        print("CUDA is not available")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # initialize wandb
    if args.use_wandb and utils.is_main_process():
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            resume="allow"
        )

    # build model
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # build optimizer
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    

    # 初始化训练状态
    best_mae, best_rmse, best_epoch = 1e8, 1e8, 0
    start_epoch = 0
    
    # 检查是否有命令行指定的resume路径
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_mae' in checkpoint:
                best_mae = checkpoint['best_mae']
            if 'best_rmse' in checkpoint:
                best_rmse = checkpoint['best_rmse']
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
        print(f"从命令行指定的checkpoint恢复训练，从第 {start_epoch} 个 epoch 继续")

    # 检查自动checkpoint目录

    # 检查自动checkpoint目录

    # 检查自动checkpoint目录
    ckpt_dir_name = f"{args.output_dir}_{args.lr}_{args.batch_size}_"
    ckpt_dir_name += f"{args.bce_loss_coef}_{args.smoothl1_loss_coef}_0828_try1"
    args.ckpt_dir = os.path.join("checkpoints", args.dataset_file, ckpt_dir_name)
    # 如果没有命令行指定的resume路径，则尝试从自动保存目录恢复
    if not args.resume and os.path.exists(args.ckpt_dir):
        ckpt_path = os.path.join(args.ckpt_dir, "checkpoint.pth")
        if os.path.isfile(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                if 'best_mae' in checkpoint:
                    best_mae = checkpoint['best_mae']
                if 'best_rmse' in checkpoint:
                    best_rmse = checkpoint['best_rmse']
                if 'best_epoch' in checkpoint:
                    best_epoch = checkpoint['best_epoch']
                print(f"自动恢复训练，从第 {start_epoch} 个 epoch 继续")
                print(f"之前最佳 MAE: {best_mae}, 最佳 RMSE: {best_rmse}, 最佳 epoch: {best_epoch}")
            except Exception as e:
                print(f"加载checkpoint失败: {e}")
                print("开始新的训练")
                start_epoch = 0
        else:
            print("checkpoint文件不存在，开始新的训练")
    
    # 确保checkpoint目录存在
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print(f"创建checkpoint目录: {args.ckpt_dir}")
    # os.makedirs(args.ckpt_dir, exist_ok=True)

    # output directory and log 
    if utils.is_main_process():
        # output_dir = os.path.join("./outputs", args.dataset_file, args.output_dir)
        # os.makedirs(output_dir, exist_ok=True)
        # output_dir = Path(output_dir)
        output_dir = Path(args.ckpt_dir)
        
        # 创建多个日志文件
        run_log_name = output_dir / 'run_log.txt'
        with open(run_log_name, "a") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}".format(args))
            log_file.write("parameters: {}".format(n_parameters))

    # training
    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        t1 = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, silent=True)
        t2 = time.time()
        train_time = time.time() - start_time

        # 安全的内存清理 - 只清理未引用的临时张量和内存碎片
        # 不会影响模型参数、优化器状态或任何重要数据
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        if utils.is_main_process():
            # 详细的训练日志
            with open(run_log_name, "a") as log_file:
                log_file.write('\n[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        if utils.is_main_process():
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_mae': best_mae,
                    'best_rmse':best_rmse,
                    'best_epoch': best_epoch,
                }, checkpoint_path)
        save_checkpoint_time = time.time() - start_time - train_time
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # log to wandb
        if args.use_wandb and utils.is_main_process():
            wandb.log({
                **{f'train/{k}': v for k, v in train_stats.items()},
                'train/epoch': epoch,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'train/learning_rate_backbone': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
            }, step=epoch)

        # write log
        if utils.is_main_process():
            with open(run_log_name, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        if epoch % args.eval_freq == 0 and epoch > 0:
            t1 = time.time()
            test_stats = evaluate(model, data_loader_val, device, epoch, None)
            t2 = time.time()
            
            # 评估后也进行内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # output results
            mae, mse = test_stats['mae'], test_stats['mse']
            if mae < best_mae:
                best_epoch = epoch
                best_mae = mae
                best_rmse = mse
                
            # log to wandb
            if args.use_wandb and utils.is_main_process():
                wandb.log({
                    'eval/mae': mae,
                    'eval/mse': mse,
                    'eval/rmse': np.sqrt(mse),
                    # 'eval/best_mae': best_mae,
                    # 'eval/best_rmse': best_rmse,
                    # 'eval/best_epoch': best_epoch,
                    # 'eval/evaluation_time': t2 - t1
                }, step=epoch)
                
            print("\n==========================")
            print("\nepoch:", epoch, "mae:", mae, "mse:", mse, "\n\nbest mae:", best_mae, "best epoch:", best_epoch)
            print("==========================\n")
            if utils.is_main_process():
                with open(run_log_name, "a") as log_file:
                    log_file.write("\nepoch:{}, mae:{}, mse:{}, time{}, \n\nbest mae:{}, best epoch: {}\n\n".format(
                                                epoch, mae, mse, t2 - t1, best_mae, best_epoch))
                                                
            # save best checkpoint
            if mae == best_mae and utils.is_main_process():
                src_path = output_dir / 'checkpoint.pth'
                dst_path = output_dir / 'best_checkpoint.pth'
                shutil.copyfile(src_path, dst_path)
        evaluate_time = time.time() - start_time - train_time - save_checkpoint_time
        print(f'train_time :{train_time:.2f}s, save_time: {save_checkpoint_time:.2f}s, evaluate_time: {evaluate_time:.2f}s')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # finish wandb run
    if args.use_wandb and utils.is_main_process():
        wandb.log({
            'training/total_time_seconds': total_time,
            'training/final_best_mae': best_mae,
            'training/final_best_rmse': best_rmse,
            'training/final_best_epoch': best_epoch
        })
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
