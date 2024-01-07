import os
import jittor as jt
import argparse
import numpy as np
from math import ceil
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

@jt.no_grad()
def sample(args:argparse.Namespace):
    # initialize settings
    # load models
    print("creating model and diffusion...")
    unet, diffusion = create_classifier_free_diffusion(
        args
    )
    checkpoint = jt.load(os.path.join(args.moddir, f'ckpt_{args.epoch}_checkpoint.pkl'))
    unet.load_state_dict(checkpoint['net'])
    # net.load_state_dict(torch.load(os.path.join(args.moddir, f'2nd_ckpt_{args.epoch}_diffusion.pt')))
    cemblayer = ConditionalEmbedding(10, args.cdim, args.cdim)
    cemblayer.load_state_dict(checkpoint['cemblayer'])
    # cemblayer.load_state_dict(torch.load(os.path.join(args.moddir, f'2nd_ckpt_{args.epoch}_cemblayer.pt')))
    # eval mode
    diffusion.model.eval()
    cemblayer.eval()
    if args.fid:
        numloop = ceil(args.genum  / args.genbatch)
    else:
        numloop = 1
    batch_size = args.genbatch
    # label settings
    if args.label == 'range':
        lab = jt.ones(args.clsnum, batch_size // args.clsnum).long() \
            * jt.arange(start = 0, end = args.clsnum).reshape(-1,1)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab
    else:
        lab = jt.randint(low = 0, high = args.clsnum, size = (batch_size,))
    # get label embeddings
    # if local_rank == 0:
    #     print(lab)
    #     print(f'numloop:{numloop}')
    cemb = cemblayer(lab)
    genshape = (batch_size, 3, 32, 32)
    all_samples = []
    print(f'numloop:{numloop}')
    for _ in range(numloop):
        if args.ddim:
            generated = diffusion.ddim_sample(genshape, args.num_steps, args.eta, args.select, cemb = cemb)
        else:
            generated = diffusion.sample(genshape, cemb = cemb)
        # transform samples into images
        img = transback(generated)
        img = img.reshape(args.clsnum, batch_size // args.clsnum, 3, 32, 32).contiguous()
        all_samples.extend([img.cpu()])
    samples = jt.concat(all_samples, dim = 1).reshape(args.genbatch * numloop, 3, 32, 32)
    # save images
    if args.fid:
        samples = (samples * 255).clamp(0, 255).unary(jt.uint8)
        samples = samples.permute(0, 2, 3, 1).numpy()[:args.genum]
        print(samples.shape)
        np.savez(os.path.join(args.samdir, f'sample_{samples.shape[0]}_diffusion_{args.epoch}_{args.w}.npz'),samples)
    else:
        save_image(samples, os.path.join(args.samdir, f'sample_{args.epoch}_pict_{args.w}.png'), nrow = args.genbatch // args.clsnum)

def main():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--dtype',default=jt.float32)
    parser.add_argument('--w',type=float,default=3.0,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=1.0,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1000,help='epochs for loading models')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--label',type=str,default='range',help='labels of generated images')
    parser.add_argument('--moddir',type=str,default='models',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0,help='dropout rate for model')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--fid',type=bool,default=False,help='generate samples used for quantative evaluation')
    parser.add_argument('--genum',type=int,default=5600,help='num of generated samples')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=bool,default=False,help='whether to use ddim')
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
    sample(args)

if __name__ == '__main__':
    main()
