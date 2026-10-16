# TacForceWM：触觉世界模型与 UR5 + RG2 动作策略

本目录保存模型训练、特征缓存与 TacForceWM 真机推理代码。硬件驱动和 ZMQ 通信统一位于仓库根目录的 `hardware/`，采集与服务入口位于 `ur5_rg2/`。

## 目录结构

```text
tacforce_wm/
├── README.md
├── requirements.txt
├── train.py
├── infer.py              # 策略推理、设备读取与力位控制，单一入口
├── config/
│   ├── dynamics_train.yaml
│   ├── config.yaml
│   └── scenes.yaml
├── datasets/             # 数据读取源码，不是采集数据
├── models/               # 触觉 tokenizer、世界模型和 flow-matching 策略
├── trainers/
├── tools/                # 包括 precompute_policy_latents.py
├── utils/
└── DINOv2/               # 配置与已有本地权重
```

训练源码直接从本目录加载，无需安装到 `site-packages/lerobot/`。`infer.py` 直接包含所需控制逻辑，不依赖其他推理脚本。

## 模型与数据

Dynamics 用左右触觉和六维力历史预测未来触觉特征；当前配置用 16 帧窗口中的前 8 帧预测后 8 帧，不使用机器人关节状态作为 Dynamics 条件。

Policy 结合腕部 RGB、关节状态、六维力、当前触觉特征及 Dynamics 预测的触觉特征，输出 32 步绝对动作序列，每步 7 维：6 个关节目标值和夹爪动作。真机入口默认每次执行 8 步后重规划，可通过 `--policy-execute-steps` 修改。

数据采用 LeRobot v2.1 格式，读取这些字段：

- `observation.wrist_image`
- `observation.state`
- `observation.force_torque`
- `observation.tactile_left.force`
- `observation.tactile_right.force`
- `action`

左右 Xense 的力分布各为 `[35, 20, 3]`。训练读取时重复末行到 `[36, 20, 3]`，双手合并；同步力序列按原模型接口重复 4 次。训练和验证按 episode 划分，窗口不跨 episode。

## 安装依赖与本机路径

在仓库根目录、已激活的 Python 3.10 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r tacforce_wm/requirements.txt
```

根目录依赖保留了原采集项目的 LeRobot 固定提交和 PyTorch 版本。真机使用的 Xense SDK、相机驱动以及设备配置按仓库根 README 准备。

后面的训练与推理命令都从模型目录执行：

```bash
cd tacforce_wm
```

## 按场景和实验后缀运行

在 `config/scenes.yaml` 中为每个场景登记一次数据目录，路径相对于 `tacforce_wm/`。已登记现有场景 `tactile_insert`；新增场景只需增加一项，例如：

```yaml
new_scene:
  data_root: ../data/my_new_scene_dataset
```

训练、缓存和真机推理使用相同的 `--scene 场景名 --suffix 实验后缀`。后缀可省略；同一场景用不同后缀保存不同实验。`--set KEY=VALUE` 可重复传入，直接覆盖现有 YAML 参数，无需修改配置文件。以下命令串起完整链路，设备连接参数可按实际硬件调整：

```bash
python train.py dynamics --scene tactile_insert --suffix exp_b64 \
  --set train.batch_size=64 --set train.val_batch_size=64

python tools/precompute_policy_latents.py --scene tactile_insert --suffix exp_b64 \
  --split all --batch-size 8 --no-amp

python train.py policy --scene tactile_insert --suffix exp_b64 \
  --set train.batch_size=64 --set train.val_batch_size=64 --set train.optimizer.lr=1e-4

python infer.py --scene tactile_insert --suffix exp_b64 \
  --force-serial-port /dev/ttyUSB0 \
  --no-hybrid-control --no-force-position-mixing --dry-run --max-steps 100
```

上述实验分别保存到 `outputs/tactile_insert_dynamics_exp_b64/` 和 `outputs/tactile_insert_policy_exp_b64/`，解析后的实际配置仍保存在各自的 `resolved_config.yaml` 中。Policy 和缓存生成入口自动选择同场景、同后缀的 Dynamics 配置及 `best_model.pt`；推理自动选择对应 Policy 的 `best.pt`，缺失时使用该实验的 `latest.pt`。指定的实验没有检查点时会报错。

若只调整 Policy 参数、复用已训练的 Dynamics 和缓存，可单独指定 Dynamics 后缀：

```bash
python train.py policy --scene tactile_insert --suffix policy_lr5e5 \
  --dynamics-suffix exp_b64 --set train.optimizer.lr=5e-5
```

缓存按所选 Dynamics 后缀区分，`tactile_insert` 的缓存为 `../data/tacforce_insert_policy_cache_<Dynamics后缀>/`；其他场景默认使用 `../data/<场景名>_policy_cache_<Dynamics后缀>/`，也可在场景表中设置 `cache_root`。如果修改了影响缓存的数据窗口等参数，缓存生成与 Policy 训练需传入相同的 `--set`，并重建缓存或通过 `--set data.latent_cache_root_dir=新目录` 分开保存。

现有的 `batch128` 实验可以直接用 `--scene tactile_insert --suffix batch128` 选择。`--config` 仍可选择基础 YAML，显式 `--set` 优先于场景默认值；不传 `--scene` 时保留原配置中的全部路径，`--policy-path` 仍可在推理时直接指定检查点。场景模式的输出目录默认来自基础 YAML 的 `output.root_dir`；如通过 `--set output.root_dir=其他目录` 修改，推理时传入相同的 `--output-root 其他目录`。

数据和缓存现统一位于仓库根目录的 `data/`。已有训练集的历史目录名 `ur5_rg2_real_smolvla_dataset_xense_tactile_insert` 保留，避免改变现有数据、缓存和实验的对应关系；默认配置和 `tactile_insert` 场景继续读取该目录。新采集器默认写入较短的 `data/ur5_rg2_xense_insert/`。训练新数据时，可以修改 `config/scenes.yaml`，或在训练与缓存命令中都添加 `--set data.root_dir=../data/ur5_rg2_xense_insert`。

其他默认路径如下：

| 内容 | 相对于模型目录的路径 |
| --- | --- |
| Dynamics 配置 | `config/dynamics_train.yaml` |
| Dynamics 最优检查点 | `outputs/tactile_insert_dynamics/checkpoints/best_model.pt` |
| Dynamics 训练时配置 | `outputs/tactile_insert_dynamics/resolved_config.yaml` |
| DINOv2 本地权重 | `DINOv2/model.safetensors` |
| Policy 最优检查点 | `outputs/tactile_insert_policy/checkpoints/best.pt` |

仓库保留 `DINOv2/` 下已有的本地权重。数据、特征缓存和训练输出由根目录 `.gitignore` 排除；Dynamics 与 Policy 检查点通过训练生成，或使用本机已有的对应训练结果。

## 训练 Dynamics

```bash
python train.py dynamics --config config/dynamics_train.yaml
```

输出写入 `outputs/tactile_insert_dynamics/`。`best_model.pt` 与 `best_model_by_loss_mse.pt` 按当前配置保存同次验证 MSE 最优结果，`latest.pt` 保存最近一轮。

多卡训练使用 NVIDIA GPU 和 NCCL，在 Ubuntu 环境中运行：

```bash
torchrun --standalone --nproc_per_node=2 train.py dynamics --config config/dynamics_train.yaml --ddp
```

## 生成缓存与训练 Policy

先准备 Dynamics 的 `best_model.pt`、`resolved_config.yaml` 和 DINOv2 权重，核对 `config/config.yaml` 中的路径。生成训练与验证缓存：

```bash
python tools/precompute_policy_latents.py --config config/config.yaml --split all --batch-size 8 --no-amp
```

如需重建已有缓存，再添加 `--overwrite`。缓存保存冻结的 DINOv2 图像特征及当前、未来触觉特征，并核对依赖权重的 SHA256。然后训练策略：

```bash
python train.py policy --config config/config.yaml
```

结果写入 `outputs/tactile_insert_policy/`。`best.pt` 为验证损失最优的策略检查点，`latest.pt` 为最近一轮，`epoch_XXXX.pt` 为定期保存的轮次检查点。

## 真机推理

先在另一个终端、仓库根目录启动 UR、双 RealSense 和双 Xense 服务，具体设备 IP、序列号及端口使用仓库根 README 的配置：

```bash
python ur5_rg2/launch.py --robot ur
```

Policy 检查点中已经打包冻结的 Dynamics 权重与归一化参数，真机推理仍需本地 DINOv2 权重。入口默认优先查找当前目录下的 `outputs/tactile_insert_policy/checkpoints/best.pt`，也可以显式传入 `--policy-path`。

从模型目录先使用 dry-run 核对实时输入、动作与控制响应；该模式读取设备并计算动作，不发送机器人动作命令：

```bash
python infer.py \
  --policy-path outputs/tactile_insert_policy/checkpoints/best.pt \
  --force-serial-port /dev/ttyUSB0 \
  --no-hybrid-control --no-force-position-mixing \
  --dry-run --max-steps 100
```

Windows 将串口换成实际的 `COM3` 等名称。模型输入只使用腕部图像，`infer.py` 的控制循环仍会连接基座相机服务。运行前需保证全部节点及两侧 Xense 可读取。

需要混合力位控制时，使用手动目标力参数，例如：

```bash
python infer.py \
  --policy-path outputs/tactile_insert_policy/checkpoints/best.pt \
  --force-serial-port /dev/ttyUSB0 \
  --hybrid-control --hybrid-force-axis z \
  --hybrid-manual-target-force-n 3.0 \
  --dry-run --max-steps 100
```

确认坐标、力符号、动作及阈值后再移除 `--dry-run`。腕部力保护和 Xense 夹爪过载保护沿用原实现，具体参数按实际设备标定与任务设置。
