"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import jittor as jt

from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        jt.load(args.model_path)
    )
    model.to("cuda")
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to("cuda")
        model_kwargs = {k: v.to("cuda") for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean()
        all_bpd.append(total_bpd.item())
        num_complete += batch.shape[0]

        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")

    for name, terms in all_metrics.items():
        out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        np.savez(out_path, np.mean(np.stack(terms), axis=0))

    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()