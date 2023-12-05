# 2023-ANN-Diffussion-Jittor

2023ANN大实验

## WorkList

### 2023.11.22

Diffusion 论文原仓库代码: https://github.com/openai/guided-diffusion/tree/main

已转移到该仓库，下一步将pytorch转为jittor。

### Files needed to be converted

**./evaluations/**

```assembly
evaluator.py
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

