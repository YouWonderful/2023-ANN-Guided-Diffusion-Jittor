# 2023-ANN-Diffussion-Jittor

2023ANN大实验

## 简介
该仓库是[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)的Jittor实现版本。并使用Jittor框架实现了[Classifier-Free Diffusion Guidance](https://openreview.net/forum?id=qw8AKxfYbI)

## Preparation
* 安装好jittor环境
* ImageNet 256x256: 预训练模型
    * 256x256 classifier: [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
    * 256x256 diffusion: [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
    * 将预训练模型下载到`models`文件夹下之后，需要将pt文件转成pkl文件以便jittor加载：`python transfer_model.py`
* ImageNet 256x256: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)，下载到`evaluations`文件夹下

## Command
```sh
# set global setting
TRAIN_FLAGS="--batch_size 1 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 4 --num_samples 1000 --timestep_respacing 250"
# train classifier
python scripts/classifier_train.py --data_dir ./datasets/train $TRAIN_FLAGS $CLASSIFIER_FLAGS
# train diffusion
python scripts/image_train.py --data_dir ./datasets/train $TRAIN_FLAGS $MODEL_FLAGS
# sample
python ./scripts/classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pkl --model_path models/256x256_diffusion.pkl $SAMPLE_FLAGS
# evaluation(第二个参数为需要 evaluate 的文件)
python ./evaluations/evaluator.py ./evaluations/VIRTUAL_imagenet256_labeled.npz ./sample/samples_10000x256x256x3.npz
# train classifier-free guidance model
python ./scripts/classifier_free_train.py
# sample from classifier-free guidance model
python ./scripts/classifier_free_sample.py
```