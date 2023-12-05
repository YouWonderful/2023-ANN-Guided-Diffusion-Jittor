# 2023-ANN-Diffussion-Jittor

2023ANN大实验

## WorkList

### 2023.11.22

Diffusion 论文原仓库代码: https://github.com/openai/guided-diffusion/tree/main

已转移到该仓库，下一步将pytorch转为jittor。

### Files needed to be converted

**./Guided Diffusion/**

```assembly
✖ dist_util.py
✔(Qiu not verified.) fp16_util.py
✔(Qiu not verified.) gaussian_diffusion.py
✔(Qiu not verified.) image_datasets.py
✔(No need to change) logger.py
✔(Qiu not verified.) losses.py
✔(Qiu not verified.) nn.py
✔(Qiu not verified.) resample.py
✔(Qiu not verified.) respace.py
✖ script_util.py
✖ train_util.py
✔(Qiu not verified) unet.py
```

**./scripts**

```assembly
✖ classifier_sample.py
✖ classifier_train.py
✖ image_nll.py
✖ image_sample.py
✖ image_train.py
✖ super_res_sample.py
✖ super_res_train.py
```

