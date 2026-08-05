# LingBot-VA 模型参数与训练显存分析

本文汇总 LingBot-VA 模型页面标签、模型组成、各组件参数量，以及默认后训练方案的显存计算。分析对象为 `lingbot-va-posttrain-robotwin` 共享骨干版本。

## 1. 模型页面标签

- **11.48B**：模型仓库统计的总参数量，约 114.8 亿参数，不是文件大小。
- **Safetensors**：权重使用 `.safetensors` 格式。
- **Diffusers**：模型按 Diffusers 组件化目录和接口组织。
- **PyTorch**：模型使用 PyTorch 实现。
- **Apache-2.0**：模型采用 Apache 2.0 开源许可证。
- **24.38 GB**：模型仓库文件总大小，不等于推理或训练所需显存。

模型文件大小约 24.38 GB，主要因为 Transformer 和文本编码器以 BF16 保存，而 VAE 以 FP32 保存。

## 2. 整体模型参数量

LingBot-VA 推理流水线包含三个主要组件：

| 组件 | 参数量 | 占总参数量 | 后训练状态 |
|---|---:|---:|---|
| 视频—动作 Transformer | 5,088,872,670 | 约 44.35% | 可训练 |
| UMT5-XXL 文本编码器 | 5,680,910,336 | 约 49.51% | 冻结 |
| Wan2.2 VAE | 704,688,668 | 约 6.14% | 冻结 |
| **结构实际总计** | **11,474,471,674** | **100%** | 约 **11.474B** |

发布的 Transformer 权重中还保留了一个当前代码未使用的旧 `patch_embedding`，包含 592,896 个参数。因此模型仓库统计为：

```text
11,474,471,674 + 592,896 = 11,475,064,570
```

即约 **11.475B**，页面四舍五入后显示为 **11.48B**。

RoboTwin 后训练只改变参数值，不改变模型结构，因此 base 和 posttrain 版本的参数量相同。

## 3. 视频—动作 Transformer

核心结构定义位于 [`wan_va/modules/model.py`](wan_va/modules/model.py)，主要配置如下：

| 配置 | 数值 |
|---|---:|
| Transformer 层数 | 30 |
| 注意力头数 | 24 |
| 单头维度 | 128 |
| 隐藏维度 | 3,072 |
| FFN 维度 | 14,336 |
| 文本条件维度 | 4,096 |
| 视频 latent 通道 | 48 |
| 动作通道 | 30 |
| Patch size | 1 × 2 × 2 |

视频流与动作流共享同一组 30 层 Transformer Block，并非各自拥有一套 5B 骨干。

### 3.1 Transformer 子模块参数量

| 子模块 | 参数量 | 占 Transformer |
|---|---:|---:|
| 30 层 FFN | 2,642,933,760 | 约 51.94% |
| 30 层自注意力 | 1,133,015,040 | 约 22.26% |
| 30 层文本交叉注意力 | 1,133,015,040 | 约 22.26% |
| 视频条件嵌入 | 88,897,536 | 约 1.75% |
| 动作条件嵌入 | 88,897,536 | 约 1.75% |
| 视频输入 Patch 投影 | 592,896 | 约 0.012% |
| 动作输入投影 | 95,232 | 约 0.002% |
| 视频输出头 | 590,016 | 约 0.012% |
| 动作输出头 | 92,190 | 约 0.002% |
| Block 调制参数表 | 552,960 | 约 0.011% |
| Cross-attention LayerNorm | 184,320 | 约 0.004% |
| 最终调制参数表 | 6,144 | 小于 0.001% |
| **合计** | **5,088,872,670** | **100%** |

每个 Transformer Block 约包含：

| Block 内部组件 | 单层参数量 |
|---|---:|
| 自注意力 | 37,767,168 |
| 文本交叉注意力 | 37,767,168 |
| FFN | 88,097,792 |
| 调制表和归一化 | 24,576 |
| **单层合计** | **163,656,704** |

30 层 Block 合计为 4,909,701,120 个参数，是核心网络参数量的主体。

## 4. UMT5-XXL 文本编码器

| 子模块 | 参数量 |
|---|---:|
| FFN | 3,019,898,880 |
| Self-Attention | 1,610,661,888 |
| Token Embedding | 1,050,148,864 |
| 归一化参数 | 200,704 |
| **合计** | **5,680,910,336** |

文本编码器负责将自然语言指令编码为 4,096 维条件。后训练数据已经保存文本 embedding，因此训练程序不会加载或优化 UMT5。

## 5. Wan2.2 VAE

| 子模块 | 参数量 |
|---|---:|
| Decoder | 555,049,228 |
| Encoder | 149,627,776 |
| Quant Conv | 9,312 |
| Post-Quant Conv | 2,352 |
| **合计** | **704,688,668** |

主要配置如下：

- `base_dim = 160`
- `decoder_base_dim = 256`
- `dim_mult = [1, 2, 4, 4]`
- `z_dim = 48`
- 空间压缩率：16
- 时间压缩率：4

后训练数据已经保存视频 latent，因此训练程序同样不会加载或优化 VAE。

## 6. 后训练时实际参与优化的参数

后训练代码位于 [`wan_va/train.py`](wan_va/train.py)。程序只加载 Transformer，并将其全部参数设置为可训练：

```python
self.transformer.requires_grad_(True)
self.optimizer = torch.optim.AdamW(
    [p for p in self.transformer.parameters() if p.requires_grad],
    ...
)
```

因此显存计算应使用：

```text
P = 5,088,872,670
```

而不是完整流水线的 11.48B。

## 7. 默认训练配置

RoboTwin 后训练配置位于 [`wan_va/configs/va_robotwin_train_cfg.py`](wan_va/configs/va_robotwin_train_cfg.py)：

```text
每卡 micro-batch size       = 1
gradient accumulation       = 1
默认 GPU 数                 = 8
参数计算 dtype              = BF16
梯度归约 dtype              = FP32
优化器                      = AdamW
FSDP reshard_after_forward  = True
activation checkpointing    = 开启
```

FSDP 和 activation checkpointing 的实现位于 [`wan_va/distributed/fsdp.py`](wan_va/distributed/fsdp.py)。

默认全局 batch size 为：

```text
global_batch
= per_gpu_batch × GPU数量 × gradient_accumulation_steps
= 1 × 8 × 1
= 8
```

## 8. 参数、梯度和优化器状态显存

Transformer 首先以 FP32 加载。FSDP 在前向和反向时临时使用 BF16 完整参数，但在优化器步骤中保留 FP32 参数分片。

每个参数的长期驻留开销为：

| 项目 | 每参数字节数 |
|---|---:|
| FP32 参数 | 4 bytes |
| FP32 梯度 | 4 bytes |
| AdamW 一阶矩 | 4 bytes |
| AdamW 二阶矩 | 4 bytes |
| **合计** | **16 bytes/param** |

若使用 `N` 张 GPU 做 FSDP 全分片，则：

```text
M_state = 16 × P / N
```

代入 `P = 5,088,872,670`：

| GPU 数 | 参数分片 | 梯度分片 | AdamW 状态 | 固定状态合计/卡 |
|---:|---:|---:|---:|---:|
| 1 | 18.96 GiB | 18.96 GiB | 37.91 GiB | **75.83 GiB** |
| 2 | 9.48 GiB | 9.48 GiB | 18.96 GiB | **37.91 GiB** |
| 4 | 4.74 GiB | 4.74 GiB | 9.48 GiB | **18.96 GiB** |
| 8 | 2.37 GiB | 2.37 GiB | 4.74 GiB | **9.48 GiB** |
| 16 | 1.18 GiB | 1.18 GiB | 2.37 GiB | **4.74 GiB** |

以上只包含参数、梯度和优化器状态，不包含激活、通信 buffer 和 CUDA 运行时。

## 9. FSDP 临时显存

FSDP 会在执行子模块前临时 all-gather 当前模块的 BF16 权重，并在前向结束后重新切分。

- 最大 FFN 约 88.1M 参数，对应约 176 MB BF16 权重。
- 根 FSDP 单元管理约 179M 参数，对应约 358 MB BF16 权重。
- 反向预取、FP32 reduce-scatter 和 NCCL 还需要额外 buffer。

因此通常需要为 FSDP 通信和临时参数预留：

```text
约 0.5～1.5 GiB/卡
```

## 10. 激活显存

设：

- `F`：视频 latent 帧数；
- `Nv`：每帧视频 token 数；
- `Na = 16`：每帧动作 token 数；
- `B`：每卡 micro-batch size。

训练同时包含 noisy/condition 两套视频和动作 token，因此序列长度为：

```text
L = 2 × B × F × (Nv + Na)
```

若使用每帧 192 个视频 token、`B = 1`：

```text
L = 2 × F × (192 + 16) = 416F
```

| latent 帧数 F | 训练序列长度 L |
|---:|---:|
| 8 | 3,328 |
| 16 | 6,656 |
| 24 | 9,984 |
| 32 | 13,312 |

### 10.1 Activation checkpointing 下限

保存 30 层 Block 输入 hidden state 的理论下限为：

```text
M_hidden = 30 × L × 3072 × 2 bytes
```

当 `L ≈ 10,000` 时：

```text
M_hidden ≈ 1.72 GiB
```

时间调制投影还需要：

```text
M_time_proj = L × 6 × 3072 × 2 bytes
```

当 `L ≈ 10,000` 时约为 0.34 GiB。

加入反向重计算时的 Q/K/V、14,336 维 FFN 中间量、FP32 LayerNorm、FlexAttention 工作区和损失计算后，实际激活及算子工作区通常估计为：

```text
约 4～8 GiB/卡（L ≈ 10K，batch size = 1）
```

激活显存随 batch size 和序列长度近似线性增长。梯度累积不会保存多个 micro-batch 的激活，因此不会按累积步数倍增峰值显存。

## 11. 最终峰值显存估算

完整估算公式为：

```text
峰值显存
≈ 参数/梯度/优化器分片
 + 激活与反向重计算工作区
 + FSDP all-gather/reduce-scatter buffer
 + CUDA、NCCL、编译缓存及内存碎片
```

在 `batch size = 1`、序列长度约 10K、启用 activation checkpointing 的情况下：

| GPU 数 | 固定状态/卡 | 估计峰值/卡 | 建议显存 |
|---:|---:|---:|---:|
| 1 | 75.83 GiB | 约 84～92+ GiB | 96 GB 以上，或使用额外卸载/量化方案 |
| 2 | 37.91 GiB | 约 46～54 GiB | 64 GB |
| 4 | 18.96 GiB | 约 27～35 GiB | 40/48 GB |
| 8 | 9.48 GiB | 约 18～26 GiB | 32 GB 起，40/48 GB 更稳 |
| 16 | 4.74 GiB | 约 13～21 GiB | 24/32 GB |

这些是基于模型结构和训练实现的工程估算，不是实测的 `torch.cuda.max_memory_allocated()` 数值。实际峰值还会受以下因素影响：

- episode 和 latent 序列长度；
- 每帧视频 token 数；
- GPU 架构和 PyTorch/CUDA 版本；
- FlexAttention 编译缓存；
- FSDP 通信预取；
- CUDA allocator 碎片；
- 是否在首个 AdamW step 之后测量。

## 12. 结论

1. ModelScope 的 **11.48B** 是 Transformer、UMT5 和 VAE 的总规模。
2. 后训练只优化约 **5.089B Transformer**，UMT5 与 VAE 不进入训练显存。
3. FP32 参数、梯度和 AdamW 状态合计为 **16 bytes/param**，由 FSDP 在 GPU 间切分。
4. 默认 8 卡配置的固定模型状态约为 **9.48 GiB/卡**。
5. 加入激活、FSDP 通信和 CUDA 开销后，默认训练估计约 **18～26 GiB/卡**。
6. **8 × 32 GB** 可视为较合理的最低配置，**8 × 40/48 GB** 更稳健。
7. 单卡全参数 AdamW 训练的固定状态已达 75.83 GiB，计入激活后 80 GB GPU 通常不足。

## 参考文件

- [`wan_va/modules/model.py`](wan_va/modules/model.py)：Transformer 结构定义。
- [`wan_va/train.py`](wan_va/train.py)：训练流程、优化器和损失计算。
- [`wan_va/distributed/fsdp.py`](wan_va/distributed/fsdp.py)：FSDP 和 activation checkpointing。
- [`wan_va/configs/va_robotwin_train_cfg.py`](wan_va/configs/va_robotwin_train_cfg.py)：RoboTwin 后训练配置。
- [`wan_va/configs/va_robotwin_cfg.py`](wan_va/configs/va_robotwin_cfg.py)：输入尺寸和动作配置。
- [`README-CH.md`](README-CH.md)：官方使用和启动说明。
- [`LingBot_VA_paper.pdf`](LingBot_VA_paper.pdf)：LingBot-VA 论文。

