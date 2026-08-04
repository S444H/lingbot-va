<h1 align="center">LingBot-VA：面向机器人控制的因果世界建模</h1>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.21998"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=arxiv"></a>
  <a href="https://technology.robbyant.com/lingbot-va"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
  <a href="https://huggingface.co/collections/robbyant/lingbot-va"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange"></a>
  <a href="https://modelscope.cn/collections/Robbyant/LingBot-VA"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple"></a>
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
</p>

<p align="center">
  <img src="assets/teaser_v3.png" width="100%">
</p>



https://github.com/user-attachments/assets/cec7b7a6-953b-4fa4-8f1a-47efc1fce547




## 目录

- [最新动态](#-最新动态)
- [模型下载](#-模型下载)
- [快速开始](#️-快速开始)
  - [安装](#安装)
  - [`attn_mode` 配置](#️-重要attn_mode-配置)
  - [部署 LingBot-VA 进行推理](#部署-lingbot-va-进行推理)
    - [在 RoboTwin-2.0 上评测](#在-robotwin-20-上评测)
    - [在 LIBERO 上评测](#在-libero-上评测)
    - [运行图像到视频-动作生成](#运行图像到视频-动作生成)
  - [LingBot-VA 后训练](#lingbot-va-后训练)
    - [数据准备](#数据准备)
    - [自定义数据集准备](#自定义数据集准备)
    - [训练](#训练)
- [性能](#-性能)
  - [仿真评测](#仿真评测)
  - [真实世界部署](#真实世界部署)
- [许可证](#-许可证)
- [引用](#引用)
- [致谢](#-致谢)

---

## 💫 认识一下 **LingBot-VA**！我们构建了一个同时进行世界建模与动作生成的自回归扩散框架！🤖✨

**LingBot-VA** 主要聚焦于：

- **自回归视频-动作世界建模**：在单个交错序列中，从架构层面统一视觉动态预测和动作推断，同时保持二者在概念上的区别。
- **高效执行**：采用双流混合 Transformer（Mixture-of-Transformers，MoT）架构，并结合异步执行和 KV Cache。
- **长时程性能与泛化能力**：在样本效率、长时程任务成功率以及对新场景的泛化能力方面取得显著提升。

# 🚀 最新动态

- **[2026-04-24]** 已发布在 **LIBERO-LONG** 上后训练的模型权重！（**重要**：请确保 [`wan_va/configs/va_libero_cfg.py`](wan_va/configs/va_libero_cfg.py) 中的 `va_libero_cfg.action_snr_shift`、`va_libero_cfg.used_action_channel_ids` 和 `va_libero_cfg.norm_stat` 与仓库最新版本保持同步。）
- **[2026-04-08]** 已发布 **LIBERO** 数据集的后训练与推理代码！
- **[2026-02-17]** 已发布后训练代码和数据集！支持在自定义机器人操作数据集上微调 LingBot-VA。
- **[2026-01-29]** 已发布共享骨干网络版本的权重和代码！敬请期待我们的分离版本！





---



# 📦 模型下载

- **用于后训练的预训练检查点**

| 模型名称 | Hugging Face 仓库 | ModelScope 仓库 | 说明 |
| :--- | :--- | :--- | :--- |
| lingbot-va-base &nbsp; | [🤗 robbyant/lingbot-va-base &nbsp;](https://huggingface.co/robbyant/lingbot-va-base) | [🤖 Robbyant/lingbot-va-base &nbsp;](https://modelscope.cn/models/Robbyant/lingbot-va-base) | 采用共享骨干网络的 LingBot-VA |
| lingbot-va-posttrain-robotwin &nbsp; | [🤗 robbyant/lingbot-va-posttrain-robotwin &nbsp;](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) | [🤖 Robbyant/lingbot-va-posttrain-robotwin &nbsp;](https://modelscope.cn/models/Robbyant/lingbot-va-posttrain-robotwin) | 采用共享骨干网络的 LingBot-VA-Posttrain-Robotwin |
| lingbot-va-posttrain-libero-long &nbsp; | [🤗 robbyant/lingbot-va-posttrain-libero-long &nbsp;](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long) | [🤖 Robbyant/lingbot-va-posttrain-libero-long &nbsp;](https://modelscope.cn/models/Robbyant/lingbot-va-posttrain-libero-long) | 采用共享骨干网络的 LingBot-VA-Posttrain-LIBERO-LONG |

- **后训练数据集**

| 数据集名称 | Hugging Face 仓库 | ModelScope 仓库 | 说明 |
| :--- | :--- | :--- | :--- |
| robotwin-clean-and-aug-lerobot &nbsp; | [🤗 robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot) | [🤖 Robbyant/robotwin-clean-and-aug-lerobot](https://modelscope.cn/datasets/Robbyant/robotwin-clean-and-aug-lerobot) | 用于后训练的 LeRobot 格式 RoboTwin 清洗与增强数据集 |
| libero-long-lerobot &nbsp; | [🤗 robbyant/libero-long-lerobot](https://huggingface.co/datasets/robbyant/libero-long-lerobot) | [🤖 Robbyant/libero-long-lerobot](https://modelscope.cn/datasets/Robbyant/libero-long-lerobot) | LeRobot 格式的 LIBERO-Long 后训练数据集 |

---

# 🛠️ 快速开始

## 安装

**环境要求**

 • Python == 3.10.16
 • PyTorch == 2.9.0
 • CUDA 12.6


```bash
conda create -n LingBot-va-1.0 python=3.10.16
conda activate LingBot-va-1.0
```

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```
```bash
pip install flash-attn --no-build-isolation
```
```bash
pip install -e .
``` 



## ⚠️ 重要：`attn_mode` 配置

> **必须根据训练或推理场景修改 `attn_mode` 设置。**
> 由于 LingBot-VA 通过 `from_pretrained` 加载，该参数会从模型目录下的 **`transformer/config.json`** 中读取。
> 启动前需要**手动编辑**此文件。
>
> | 模式 | `attn_mode` 值 | 说明 |
> |---|---|---|
> | **训练** | `"flex"` | 训练时必须使用，**无法**用于推理。 |
> | **推理 / 评测** | `"torch"` 或 `"flashattn"` | 推理时必须使用；在评测时使用 `"flex"` 会导致错误。 |
>
> **修改方法：** 打开 `<your-model-path>/transformer/config.json`，找到 `"attn_mode"` 字段，并将其设置为对应的值。

---

## 部署 LingBot-VA 进行推理

LingBot-VA 既支持独立运行，也支持将模型环境与仿真环境分离的服务器—客户端架构。通过隔离依赖，该设计可以避免软件包冲突，并支持在 GPU、集群和其他设备上进行分布式推理。

<!-- ### 独立推理
```python
python inference.py
```
该命令会处理 `examples/0/` 中的示例数据，并将可视化结果保存至 `result/`。 -->

### 在 RoboTwin-2.0 上评测

**准备环境**

可以参考 RoboTwin-2.0 原仓库的官方说明：
[https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)


简要步骤如下：

1. 安装 Vulkan 依赖：

   ```bash
   sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
   ```

2. 克隆 RoboTwin 仓库：

   ```bash
   git clone https://github.com/RoboTwin-Platform/RoboTwin.git && cd RoboTwin && git checkout 2eeec322
   ```

3. 将 `script/requirements.txt` 修改为以下内容：

   ```txt
   transforms3d==0.4.2
   sapien==3.0.0b1
   scipy==1.10.1
   mplib==0.2.1
   gymnasium==0.29.1
   trimesh==4.4.3
   open3d==0.18.0
   imageio==2.34.2
   pydantic
   zarr
   openai
   huggingface_hub==0.36.2
   h5py
   # For Description Generation
   azure==4.0.0
   azure-ai-inference
   pyglet<2
   wandb
   moviepy
   imageio
   termcolor
   av
   matplotlib
   ffmpeg
   ```

4. 修改 `script/_install.sh` 的第 8 行：

   ```bash
   pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
   ```

5. 安装依赖：

   ```bash
   bash script/_install.sh
   ```

6. 下载资源：

   ```bash
   bash script/_download_assets.sh
   ```

**部署推理服务器**

```bash
# 单 GPU
bash evaluation/robotwin/launch_server.sh

# 多 GPU
bash evaluation/robotwin/launch_server_multigpus.sh
```

**运行推理客户端**

```bash
# 单 GPU
task_name="adjust_bottle";
save_root="results/";
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}

# 多 GPU
save_root="results/"
task_group_id=0;
bash evaluation/robotwin/launch_client_multigpus.sh ${save_root} ${task_group_id}
```

相关实验结果将保存在 `/path/to/your/RoboTwin/${save_root}` 中。请注意，运行时还会生成一个 `eval_result` 文件夹；这是 RoboTwin 的原生输出，其内容与 results 文件夹相同，可以安全忽略。

推理服务器和客户端必须部署在同一台机器上。为启动多 GPU 客户端，我们通过复制任务将原有 50 个任务补齐到 56 个，并划分为 7 组，以适配推理节点的 8 GPU 配置。可以通过指定 `task_group_id`（0–6）来选择某一组进行推理。详细的分组配置请参阅 `evaluation/robotwin/launch_client_multigpus.sh`。

> **GPU 显存要求**：在启用卸载模式（将 VAE 和 `text_encoder` 卸载至 CPU）的情况下，单 GPU RoboTwin 评测约需 **24 GB 显存**。


### 在 LIBERO 上评测

按照官方说明安装 LIBERO，然后启动服务器和客户端：


```bash
# 服务器
bash evaluation/libero/launch_server.sh

# 客户端
bash evaluation/libero/launch_client.sh
```

### 运行图像到视频-动作生成

我们还提供了一个用于图像到视频-动作生成的脚本：

```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```

> **GPU 显存要求**：在启用卸载模式（将 VAE 和 `text_encoder` 卸载至 CPU）的情况下，单 GPU i2av 推理约需 **18 GB 显存**。


## LingBot-VA 后训练

我们支持使用自定义机器人操作数据集对 LingBot-VA 进行后训练（微调）。训练流水线使用 FSDP 进行分布式训练，并集成了 [LeRobot](https://github.com/huggingface/lerobot) 数据集格式。

### 额外依赖

除基础安装所需的依赖外，后训练还需要：

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```
```bash
pip install "datasets==4.1.1"
```
```bash
pip install "jsonlines>=4.0.0" "av>=14.2.0"
```



### 数据准备

从 Hugging Face 下载后训练数据集：

```bash
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/your/dataset
```

### 自定义数据集准备

如果希望使用自己的机器人操作数据微调 LingBot-VA，请按照以下步骤进行操作：

#### 示例数据集

我们提供了一个基于 [Issue #29](https://github.com/Robbyant/lingbot-va/issues/29) 中数据转换而来的示例数据集。该数据集已转换为预期格式，并完全支持训练。你可以下载该数据集，以了解所需的数据结构：

- **下载**：[示例数据集](https://drive.google.com/file/d/1D52nK4ZOJmWBXKv1nWrLb9YBwq8nKa_b/view?usp=sharing)

此示例可作为参考，帮助你将自己的机器人操作数据转换为正确格式。

#### 数据流水线概览

准备自定义数据集时，数据会经过以下处理流程：

1. **原始数据** → 转换为 LeRobot 格式（包含元数据和视频文件）
2. **添加动作分段** → 向 `episodes.jsonl` 添加 `action_config`
3. **提取潜表示** → 根据视频规格，使用 VAE 处理视频
4. **加载数据集** → 按照正确的动作维度加载处理后的数据，以用于训练

最终数据应符合以下规范：

**动作格式：**

- 输出维度：**30 维**，结构如下：
  - 左臂 EEF（末端执行器）：7 维
  - 右臂 EEF（末端执行器）：7 维
  - 左臂关节：7 维
  - 右臂关节：7 维
  - 左臂夹爪：1 维
  - 右臂夹爪：1 维
- 在数据集类的加载器中，将机器人的动作维度映射到这一标准的 30 维格式。缺失维度以 **0** 填充。

**视频格式：**

- 在提取 VAE 潜表示时，可参考将视频尺寸调整为约 **256 × 256 像素**，并降采样至 **5–15 fps**（请根据具体任务要求进行调整）。

#### 实现步骤

**步骤 1：将数据转换为 LeRobot 格式**

请按照 [LeRobot 数据集官方文档](https://github.com/huggingface/lerobot/tree/v0.3.3)，将原始数据（如 HDF5、视频文件等）转换为标准 LeRobot 数据集格式。请确保每个 episode 都包含所需的观测视频、动作和元数据。

**步骤 2：向 `episodes.jsonl` 添加 `action_config` 字段**

转换为 LeRobot 格式后，需要修改 `meta/episodes.jsonl` 文件，为每一行添加 `action_config` 字段。该字段用于描述每个 episode 中机器人动作的时间分段和自然语言描述。

`episodes.jsonl` 中的每一行应采用以下格式：

```json
{
  "episode_index": 0,
  "tasks": ["任务描述"],
  "length": 450,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 450,
      "action_text": "对该分段中机器人动作的自然语言描述。",
    }
  ]
}
```

- `start_frame` / `end_frame`：episode 内动作分段的帧范围（从 0 开始计数）。
- `action_text`：机器人在该分段中所执行动作的自然语言描述。

对于只包含一个连续动作的 episode，`start_frame` 应设为 `0`，`end_frame` 应等于该 episode 的 `length`。如果数据包含连续的多个子任务，也可以为每个 episode 定义多个分段。

**步骤 3：使用 Wan2.2 VAE 提取视频潜表示**

LingBot-VA 基于视频潜表示而非原始像素运行。你需要使用 Wan2.2 VAE 编码器提取潜特征，并将其放在转换后的 LeRobot 数据集目录下。有关如何运行 VAE 编码器的说明，请参阅 [Wan-Video 文档](https://github.com/Wan-Video)。

提取后的潜表示文件应放在数据集目录的 `latents/` 下，并与 `videos/` 的目录结构保持一致：

```
your_dataset/
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000.mp4
│           └── ...
├── latents/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_450.pth    # 命名格式：episode_{index}_{start_frame}_{end_frame}.pth
│           └── ...
└── meta/
    └── episodes.jsonl
```

每个 `.pth` 文件都是一个包含以下字段的字典：

| 键 | 类型 | 说明 |
| :--- | :--- | :--- |
| `latent` | `Tensor [N, C]`（bfloat16） | 展平后的 VAE 潜特征（例如形状为 `[latent_num_frames * latent_height * latent_width, C]`） |
| `latent_num_frames` | `int` | 潜空间中的时间帧数 |
| `latent_height` | `int` | 潜空间的空间高度 |
| `latent_width` | `int` | 潜空间的空间宽度 |
| `video_num_frames` | `int` | （采样后的）源视频帧数 |
| `video_height` | `int` | 原始视频的像素高度 |
| `video_width` | `int` | 原始视频的像素宽度 |
| `text_emb` | `Tensor [L, D]`（bfloat16） | 动作描述的文本嵌入（由 Wan2.2 文本编码器编码） |
| `text` | `str` | 原始动作描述文本 |
| `frame_ids` | `list[int]` | 从原始 episode 中采样的帧索引（按目标 fps） |
| `start_frame` | `int` | 与 `episodes.jsonl` 中 `action_config` 对应的起始帧索引 |
| `end_frame` | `int` | 与 `episodes.jsonl` 中 `action_config` 对应的结束帧索引 |
| `fps` | `int` | 提取潜表示时使用的目标采样帧率 |
| `ori_fps` | `int` | episode 数据的原始帧率 |

潜表示文件的命名约定 `episode_{index}_{start_frame}_{end_frame}.pth` 与 `episodes.jsonl` 中定义的 `action_config` 分段相对应。例如，当某个 episode 的 `"start_frame": 0, "end_frame": 450` 时，会生成名为 `episode_000000_0_450.pth` 的潜表示文件。

### 训练

#### wandb
```bash
wandb login
``` 
```bash
export WANDB_API_KEY="$(
  python -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])'
)"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="your"
export WANDB_PROJECT="self set"
```
##### 验证
```bash
printf 'WANDB_API_KEY: %s\n' "$([[ -n "$WANDB_API_KEY" ]] && echo 已设置 || echo 未设置)"
printf 'WANDB_BASE_URL: %s\n' "$WANDB_BASE_URL"
printf 'WANDB_TEAM_NAME: %s\n' "$WANDB_TEAM_NAME"
printf 'WANDB_PROJECT: %s\n' "$WANDB_PROJECT"
```

```bash
# RoboTwin
NGPU=8 CONFIG_NAME='robotwin_train' SAVE_ROOT='/self_set_path' bash script/run_va_posttrain.sh

# LIBERO
NGPU=8 CONFIG_NAME='libero_train' SAVE_ROOT='/self_set_path' bash script/run_va_posttrain.sh
```

为了获得更好的训练性能，请使用更大的全局批量大小（例如 32 或 64）。如果 GPU 资源有限，可以增加 `gradient_accumulation_steps`，以实现更大的有效批量大小。


---

# 📊 性能

我们在仿真基准和真实世界场景中对模型进行了评测，并取得了当前最佳性能。

## 仿真评测

- **RoboTwin 2.0**

我们率先将 RoboTwin 2.0 指标提升至 90+！

<table style="border-collapse: collapse; width: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; line-height: 1.2;">
<!-- 指标说明 -->
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* 所有指标均以百分比（%）报告，较高值以<b>粗体</b>标出。</p>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th align="left" style="padding: 6px 12px; white-space: nowrap;">方法（50 个任务的平均值）</th>
      <th align="center" style="padding: 6px 12px;">简单任务 SR（%）</th>
      <th align="center" style="padding: 6px 12px;">困难任务 SR（%）</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">X-VLA</td>
      <td align="center">72.9</td>
      <td align="center">72.8</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">&pi;<sub>0</sub></td>
      <td align="center">65.9</td>
      <td align="center">58.4</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">&pi;<sub>0.5</sub></td>
      <td align="center">82.7</td>
      <td align="center">76.8</td>
    </tr>
    <tr>
      <td style="padding: 4px 12px; white-space: nowrap;">Motus</td>
      <td align="center"><u>88.7</u></td>
      <td align="center"><u>87.0</u></td>
    </tr>
    <tr style="border-top: 1px solid black; border-bottom: 2px solid black;">
      <td style="padding: 6px 12px; white-space: nowrap;"><b>LingBot-VA（本方法）</b></td>
      <td align="center"><b>92.9</b> <small>(+4.2)</small></td>
      <td align="center"><b>91.6</b> <small>(+4.6)</small></td>
    </tr>
  </tbody>
</table>


- **LIBERO**

<table style="border-collapse: collapse; width: auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; line-height: 1.2;">
<!-- 指标说明 -->
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* 所有指标均以百分比（%）报告，较高值以<b>粗体</b>标出。</p>
  <thead>
    <tr style="border-top: 2px solid black; border-bottom: 1px solid black;">
      <th align="left" style="padding: 6px 10px; border-right: 1px solid black; white-space: nowrap;">方法</th>
      <th align="center" style="padding: 6px 8px;">空间</th>
      <th align="center" style="padding: 6px 8px;">物体</th>
      <th align="center" style="padding: 6px 8px;">目标</th>
      <th align="center" style="padding: 6px 8px;">长时程</th>
      <th align="center" style="padding: 6px 8px;">平均值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">&pi;<sub>0</sub></td>
      <td align="center">96.8</td><td align="center">98.8</td><td align="center">95.8</td><td align="center">85.2</td><td align="center">94.1</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">&pi;<sub>0.5</sub></td>
      <td align="center">98.8</td><td align="center">98.2</td><td align="center">98.0</td><td align="center">92.4</td><td align="center">96.9</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">OpenVLA</td>
      <td align="center">84.7</td><td align="center">88.4</td><td align="center">79.2</td><td align="center">53.7</td><td align="center">76.5</td>
    </tr>
    <tr>
      <td style="padding: 4px 10px; border-right: 1px solid black; white-space: nowrap;">X-VLA</td>
      <td align="center">98.2</td><td align="center">98.6</td><td align="center">97.8</td><td align="center">97.6</td><td align="center">98.1</td>
    </tr>
    <tr style="border-top: 1.5px solid black; border-bottom: 2px solid black;">
      <td style="padding: 5px 10px; border-right: 1px solid black; white-space: nowrap;"><b>LingBot-VA（本方法）</b></td>
      <td align="center"><b>98.5 &plusmn; 0.3</b></td>
      <td align="center"><b>99.6 &plusmn; 0.3</b></td>
      <td align="center"><b>97.2 &plusmn; 0.2</b></td>
      <td align="center"><b>98.5 &plusmn; 0.5</b></td>
      <td align="center"><b>98.5</b></td>
    </tr>
  </tbody>
</table>



&nbsp;

## 真实世界部署

我们评测了三类共六项操作任务：长时程任务（制作早餐、拾取螺钉）、精细操作任务（插入软管、拆开包裹），以及可变形物体与关节物体操作任务（折叠衣物、折叠裤子）。在每项任务**仅进行 50 次试验**的情况下，本方法在两个指标（进度分数和成功率）上都取得了当前最佳性能，并显著优于强基线 &pi;<sub>0.5</sub>。

<div style="text-align: left; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; line-height: 1.6;">

  <!-- 第一部分：PS 说明 -->
  <div style="margin-bottom: 5px;"><strong>进度分数（Progress Score，PS）：</strong>所有试验的平均得分除以最大可能得分，并以百分比表示：</div>

  PS = Average_Progress / Max_Steps &times; 100%

  <!-- 第二部分：SR 说明 -->
  <div style="margin-bottom: 5px;"><strong>成功率（Success Rate，SR）：</strong>成功试验次数除以试验总次数，并以百分比表示：</div>

  SR = Successful_Trials / N &times; 100%

</div>



<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;">
  <!-- 指标说明 -->
  <p style="font-size: 12px; color: #666; margin-bottom: 5px;">* 所有指标均以百分比（%）报告，较高值以<b>粗体</b>标出。</p>
  
  <table style="border-collapse: collapse; width: auto; font-size: 13px; line-height: 1.2;">
    <thead>
      <tr style="border-top: 2px solid black;">
        <th rowspan="2" align="left" style="padding: 4px 10px; border-bottom: 1px solid black; white-space: nowrap;"><b>任务</b></th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">制作早餐</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">拾取螺钉</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">插入软管</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">拆开包裹</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">折叠衣物</th>
        <th colspan="2" style="padding: 4px 10px; border-bottom: 1px solid black;">折叠裤子</th>
      </tr>
      <tr style="border-bottom: 1px solid black;">
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
        <th style="padding: 4px 8px;">PS</th>
        <th style="padding: 4px 8px;">SR</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 6px 10px; white-space: nowrap;">&pi;<sub>0.5</sub></td>
        <td align="center">73.0</td><td align="center">70.0</td>
        <td align="center">74.0</td><td align="center">50.0</td>
        <td align="center">79.2</td><td align="center">30.0</td>
        <td align="center">73.0</td><td align="center">25.0</td>
        <td align="center"><b>62.9</b></td><td align="center">30.0</td>
        <td align="center">30.0</td><td align="center">30.0</td>
      </tr>
      <tr style="border-bottom: 2px solid black;">
        <td style="padding: 6px 10px; white-space: nowrap;"><b>LingBot-VA（本方法）</b></td>
        <td align="center"><b>97.0</b></td><td align="center"><b>75.0</b></td>
        <td align="center"><b>82.5</b></td><td align="center"><b>70.0</b></td>
        <td align="center"><b>85.8</b></td><td align="center"><b>40.0</b></td>
        <td align="center"><b>84.5</b></td><td align="center"><b>65.0</b></td>
        <td align="center">48.8</td><td align="center"><b>35.0</b></td>
        <td align="center"><b>76.7</b></td><td align="center"><b>70.0</b></td>
      </tr>
    </tbody>
  </table>
</div>


# 🪪 许可证

本项目基于 Apache License 2.0 发布。详情请参阅 [LICENSE](LICENSE.txt) 文件。

# 📚引用

```bibtex
@article{lingbot-va2026,
  title={Causal World Modeling for Robot Control},
  author={Li, Lin and Zhang, Qihang and Luo, Yiming and Yang, Shuai and Wang, Ruilin and Han, Fei and Yu, Mingrui and Gao, Zelin and Xue, Nan and Zhu, Xing and Shen, Yujun and Xu, Yinghao},
  journal={arXiv preprint arXiv:2601.21998},
  year={2026}
}
```

# 🧩 致谢

本工作构建于以下优秀的开源项目之上：

- [Wan-Video](https://github.com/Wan-Video) — 视觉 Transformer 骨干网络
- [MoT](https://github.com/facebookresearch/Mixture-of-Transformers) — 混合 Transformer 架构
- 更广泛的开源计算机视觉和机器人社区

---

如有问题、讨论或合作意向：

- **Issues**：在 GitHub 上提交 [issue](https://github.com/robbyant/lingbot-va/issues)
- **邮箱**：联系 [Qihang Zhang 博士](https://zqh0253.github.io/)（liuhuan.zqh@antgroup.com）或 [Lin Li 博士](https://lilin-hitcrt.github.io/)（fengchang.ll@antgroup.com）
