# TacForceWM：触觉世界模型与 UR5 + RG2 动作策略

本目录集中保存训练源码、Xense 真机推理入口，以及推理使用的机器人控制依赖。模型源码直接从当前目录加载，无需放回虚拟环境的 `site-packages/lerobot/tacforce_wm`。

## 目录结构

将整个目录放在采集仓库的 `real_ur5rg2/tacforce_wm/` 下，保持单层结构：

```text
real_ur5rg2/tacforce_wm/
├── README.md
├── .gitignore
├── requirements.txt
├── train.py
├── infer_tacforce_wm_policy_real_ur5_M.py
├── infer_tactile_smolvla_real_ur5_hybrid_control.py
├── config/
│   ├── dynamics_train.yaml
│   └── config.yaml
├── datasets/     # 数据读取源码，不是采集数据
├── models/       # 触觉 tokenizer、世界模型和 flow-matching 策略
├── trainers/
├── tools/
├── utils/
└── DINOv2/
    └── config.json
```

推理入口为带 `_M` 的文件，使用双 Xense 力分布。另一个 `infer_...hybrid_control.py` 是共用的机器人、相机、力传感器通信及力位控制依赖，TacForceWM 入口复用其控制循环。本目录只有这一份 README。

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
python -m pip install -r real_ur5rg2/tacforce_wm/requirements.txt
```

根目录依赖保留了原采集项目的 LeRobot 固定提交和 PyTorch 版本。真机使用的 Xense SDK、相机驱动以及设备配置按仓库根 README 准备。

后面的训练与推理命令都从模型目录执行：

```bash
cd real_ur5rg2/tacforce_wm
```

两份 YAML 已去掉旧机器的绝对路径。默认数据目录为 `../data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_200`；如果本机数据放在其他位置，修改两份配置的 `data.root_dir`。其他默认路径如下：

| 内容 | 相对于模型目录的路径 |
| --- | --- |
| Dynamics 配置 | `config/dynamics_train.yaml` |
| Dynamics 最优检查点 | `outputs/tactile_boltnut_dynamics/checkpoints/best_model.pt` |
| Dynamics 训练时配置 | `outputs/tactile_boltnut_dynamics/resolved_config.yaml` |
| DINOv2 本地权重 | `DINOv2/model.safetensors` |
| Policy 最优检查点 | `outputs/wrist_policy_run/checkpoints/best.pt` |

网页源码包不包含数据集、模型权重、TensorBoard 日志、训练输出或编译缓存。运行机器需要另行准备数据和 DINOv2 权重；Dynamics 与 Policy 权重可以自行训练，或从本地保留的对应训练结果复制。

## 训练 Dynamics

```bash
python train.py dynamics --config config/dynamics_train.yaml
```

输出写入 `outputs/tactile_boltnut_dynamics/`。`best_model.pt` 与 `best_model_by_loss_mse.pt` 按当前配置保存同次验证 MSE 最优结果，`latest.pt` 保存最近一轮。

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

结果写入 `outputs/wrist_policy_run/`。`best.pt` 为验证损失最优的策略检查点，`latest.pt` 为最近一轮，`epoch_XXXX.pt` 为定期保存的轮次检查点。

## 真机推理

先在另一个终端、仓库根目录启动 UR、双 RealSense 和双 Xense 服务，具体设备 IP、序列号及端口使用仓库根 README 的配置：

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur --tactile-source xense
```

Policy 检查点中已经打包冻结的 Dynamics 权重与归一化参数，真机推理仍需本地 DINOv2 权重。入口默认优先查找当前目录下的 `outputs/wrist_policy_run/checkpoints/best.pt`，也可以显式传入 `--policy-path`。

从模型目录先使用 dry-run 核对实时输入、动作与控制响应；该模式读取设备并计算动作，不发送机器人动作命令：

```bash
python infer_tacforce_wm_policy_real_ur5_M.py \
  --policy-path outputs/wrist_policy_run/checkpoints/best.pt \
  --force-serial-port /dev/ttyUSB0 \
  --no-hybrid-control --no-force-position-mixing \
  --dry-run --max-steps 100
```

Windows 将串口换成实际的 `COM3` 等名称。模型输入只使用腕部图像，共用控制循环仍会连接基座相机服务。运行前需保证全部节点及两侧 Xense 可读取。

需要混合力位控制时，使用共用控制器的参数；TacForceWM 不输出目标力，所以采用手动目标力并关闭预测力混合，例如：

```bash
python infer_tacforce_wm_policy_real_ur5_M.py \
  --policy-path outputs/wrist_policy_run/checkpoints/best.pt \
  --force-serial-port /dev/ttyUSB0 \
  --hybrid-control --hybrid-force-axis z \
  --hybrid-manual-target-force-n 3.0 --hybrid-pred-force-blend 0.0 \
  --dry-run --max-steps 100
```

确认坐标、力符号、动作及阈值后再移除 `--dry-run`。腕部力保护和 Xense 夹爪过载保护沿用原实现，具体参数按实际设备标定与任务设置。

## 通过网页上传

解压提供的网页上传包，打开 GitHub 仓库的 `real_ur5rg2` 目录，选择 **Add file → Upload files**，拖入解压后的整个 `tacforce_wm` 文件夹。检查最终路径为 `real_ur5rg2/tacforce_wm/train.py`，中间没有第二层 `tacforce_wm`，再提交。

上传包中的所有文件都低于 25 MiB，文件数量低于每批 100 个的网页限制。[GitHub 官方说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

如果 `.gitignore` 没有出现在上传列表中，进入已经上传的 `tacforce_wm` 目录单独添加这个文件。网页上传不会根据本机忽略规则筛选文件，请使用准备好的源码包；本地原目录中的 `outputs/`、DINOv2 权重和 `__pycache__/` 已从包中排除。

## 本次核对范围

目录调整后已检查 Python 3.10 语法、本地源码导入路径、推理与控制依赖的接口，以及命令行帮助和 Xense 输入格式。未运行 GPU 训练、加载训练检查点或连接真实机器人。
