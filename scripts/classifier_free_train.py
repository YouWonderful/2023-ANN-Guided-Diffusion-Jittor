import os
import argparse
import numpy as np
from tqdm import tqdm
import jittor as jt
import jittor.optim as optim
import itertools
from jittor.misc import save_image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表
from guided_diffusion.embedding import ConditionalEmbedding
from guided_diffusion.image_datasets import load_data_cifar, transback
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_classifier_free_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

jt.flags.use_cuda = 1

from jittor.optim import LRScheduler

class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler = None, last_epoch = None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = last_epoch
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
    def state_dict(self):
        warmdict = {key:value for key, value in self.__dict__.items() if (key != 'optimizer' and key != 'after_scheduler')}
        cosdict = {key:value for key, value in self.after_scheduler.__dict__.items() if key != 'optimizer'}
        return {'warmup':warmdict, 'afterscheduler':cosdict}
    def load_state_dict(self, state_dict: dict):
        self.after_scheduler.__dict__.update(state_dict['afterscheduler'])
        self.__dict__.update(state_dict['warmup'])

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def train(args:argparse.Namespace):
    # initialize settings
    # load data
    print("creating data loader...")
    data = load_data_cifar(
        args.batchsize
    )
    # initialize models
    print("creating model and diffusion...")
    unet, diffusion = create_classifier_free_diffusion(
        args
    )
    
    cemblayer = ConditionalEmbedding(10, args.cdim, args.cdim)
    # load checkpoint
    if args.checkpoint >= 0:
        print("loading checkpoint...")
        lastepc = args.checkpoint
        # load checkpoints
        checkpoint = jt.load(os.path.join(args.moddir, f'ckpt_{lastepc}_checkpoint.pkl'))
        unet.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    
    # optimizer settings
    optimizer = optim.AdamW(
                    list(itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters()
                    )),
                    lr = args.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = jt.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = args.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = args.multiplier,
                            warm_epoch = args.epoch // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    for epc in range(lastepc, args.epoch):
        # turn into train mode
        diffusion.model.train()
        cemblayer.train()
        # batch iterations
        with tqdm(total=50000 // args.batchsize, dynamic_ncols=True) as tqdmDataLoader:
            for img, lab in data:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img
                lab = lab
                cemb = cemblayer(lab)
                cemb[np.where(np.random.rand(b)<args.threshold)] = 0
                loss = diffusion.trainloss(x_0, cemb = cemb)
                optimizer.backward(loss)
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['defaults']["lr"]
                    }
                )
                tqdmDataLoader.update(1)
        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % args.interval == 0:
            print('saving checkpoint...')
            # save checkpoints
            checkpoint = {
                                'net':diffusion.model.state_dict(),
                                'cemblayer':cemblayer.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            if not os.path.exists(args.moddir):
                os.mkdir(args.moddir)
            jt.save({'last_epoch':epc+1}, os.path.join(args.moddir,'last_epoch.pkl'))
            jt.save(checkpoint, os.path.join(args.moddir, f'ckpt_{epc+1}_checkpoint.pkl'))
            
            print('sampling...')
            diffusion.model.eval()
            cemblayer.eval()
            # generating samples
            # The model generate 80 pictures(8 per row) each time
            # pictures of same row belong to the same class
            all_samples = []
            batch_size = args.genbatch
            with jt.no_grad():
                lab = jt.ones(args.clsnum, batch_size // args.clsnum) \
                * jt.arange(start = 0, end = args.clsnum).reshape(-1, 1)
                lab = lab.reshape(-1, 1).squeeze()
                lab = lab
                cemb = cemblayer(lab)
                genshape = (batch_size , 3, 32, 32)
                if args.ddim:
                    generated = diffusion.ddim_sample(genshape, args.num_steps, args.eta, args.select, cemb = cemb)
                else:
                    generated = diffusion.sample(genshape, cemb = cemb)
                img = transback(generated)
                img = img.reshape(args.clsnum, batch_size // args.clsnum, 3, 32, 32).contiguous()
                all_samples.extend([img])
                samples = jt.concat(all_samples, dim = 1).reshape(args.genbatch, 3, 32, 32)
                if not os.path.exists(args.samdir):
                    os.mkdir(args.samdir)
                save_image(samples, os.path.join(args.samdir, f'generated_{epc+1}_pict.png'), nrow = args.genbatch // args.clsnum)

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=16,help='batch size per device for training Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=1,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='models',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--checkpoint',default=-1,type=int,help='model\'s checkpoint epoch')
    # unet_setting
    parser.add_argument('--image_size',default=32,type=int)
    parser.add_argument('--num_channels',default=64,type=int)
    parser.add_argument('--num_res_blocks',default=2,type=int)
    parser.add_argument('--channel_mult',default="1,2,4,8",type=str)
    parser.add_argument('--learn_sigma',default=False,type=bool)
    parser.add_argument('--class_cond',default=False,type=bool)
    parser.add_argument('--use_checkpoint',default=False,type=bool)
    parser.add_argument('--attention_resolutions',default="16,8",type=str)
    parser.add_argument('--num_heads',default=4,type=int)
    parser.add_argument('--num_head_channels',default=-1,type=int)
    parser.add_argument('--num_heads_upsample',default=-1,type=int)
    parser.add_argument('--use_scale_shift_norm',default=True,type=bool)
    parser.add_argument('--dropout',default=0.0,type=float)
    parser.add_argument('--resblock_updown',default=False,type=bool)
    parser.add_argument('--use_fp16',default=False,type=bool)
    parser.add_argument('--use_new_attention_order',default=False,type=bool)
    parser.add_argument('--use_classifier_free_diffusion',default=True,type=bool)
    # NOTICE: this param is only used when use_classifier_free_diffusion = True
    parser.add_argument('--class_num',default=10,type=int)
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()