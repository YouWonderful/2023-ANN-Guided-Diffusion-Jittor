# 2023-ANN-Diffussion-Jittor

2023ANN大实验

## WorkList

### 2023.11.22

Diffusion 论文原仓库代码: https://github.com/openai/guided-diffusion/tree/main

已转移到该仓库，下一步将pytorch转为jittor。

### Files needed to be converted

**./evaluations/**

```assembly
✔(You not verified) evaluator.py
```

**./Guided Diffusion/**

```assembly
✖✖✖(用不到) dist_util.py
✔(Qiu not verified.) fp16_util.py
✔(Qiu not verified.) gaussian_diffusion.py
✔(Qiu not verified.) image_datasets.py
✔(No need to change) logger.py
✔(Qiu not verified.) losses.py
✔(Qiu not verified.) nn.py
✔(Qiu not verified.) resample.py
✔(Qiu not verified.) respace.py
✔(本来就不需要改) script_util.py
✔(You not verified) train_util.py
✔(Qiu not verified) unet.py
```

**./scripts**

```assembly
✔(You not verified) classifier_sample.py
✔(You not verified) classifier_train.py
✔(You not verified) image_nll.py
✔(You not verified) image_sample.py
✔(You not verified) image_train.py
✔(You not verified) super_res_sample.py
✔(You not verified) super_res_train.py
```

#### Train command
```sh
# set global setting
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 12 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 100 --timestep_respacing 250"
# train classifier
python scripts/classifier_train.py --data_dir ./datasets/train $TRAIN_FLAGS $CLASSIFIER_FLAGS
# sample
python ./scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```
python3 ./scripts/classifier_sample.py --batch_size 4 --num_samples 1000 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --classifier_scale 1.0 --classifier_path models/256x256_classifier.pkl --model_path models/256x256_diffusion.pkl --use_fp16 True

python3 ./scripts/classifier_sample.py --batch_size 4 --num_samples 2000 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --classifier_scale 1.0 --classifier_path ../2023-ann-diffussion-jittor/models/256x256_classifier.pt --model_path ../2023-ann-diffussion-jittor/models/256x256_diffusion.pt --use_fp16 True

python evaluator.py VIRTUAL_imagenet256_labeled.npz ../sample/samples_50000x256x256x3.npz 

python evaluatorjt.py VIRTUAL_imagenet256_labeled.npz ../sample/samples_1000x256x256x3.npz 

python3 ../guided-diffusion-main2/scripts/classifier_sample.py --batch_size 4 --num_samples 50 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt --use_fp16 True

Pytorch Version
python3 ./scripts/sample.py --batch_size 20 --num_samples 1000 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --classifier_scale 1.0 --classifier_path ../2023-ann-diffussion-jittor/models/256x256_classifier.pt --model_path ../2023-ann-diffussion-jittor/models/256x256_diffusion.pt --use_fp16 True