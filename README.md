# UR5 + RG2 触觉与力控制

使用 UR5、RG2、双 RealSense、双 Xense 和 FTS-300-S 进行键盘遥操作、LeRobot 数据采集与 TacForceWM 训练、推理。

## 目录与入口

```text
.
├── hardware/                 # 相机、机器人、力传感器、触觉驱动与 ZMQ 通信
├── ur5_rg2/
│   ├── launch.py             # 启动机器人、相机与触觉服务
│   └── collect.py            # 键盘遥操作与数据采集
├── data_tools/
│   ├── merge.py              # 合并 LeRobot 数据集
│   └── deletion.py           # 覆盖数据前的删除确认，共用模块
├── tacforce_wm/
│   ├── train.py              # Dynamics / Policy 训练
│   ├── infer.py              # TacForceWM 真机推理与力位控制
│   ├── config/               # 模型参数与场景数据路径
│   ├── datasets/             # 数据读取源码
│   ├── models/
│   ├── trainers/
│   ├── tools/                # 特征缓存等模型工具
│   ├── utils/
│   └── DINOv2/               # 图像编码器配置与本地权重
└── data/                     # 本机数据与特征缓存，不纳入 Git
```

训练、缓存和推理的完整用法见 [TacForceWM 说明](tacforce_wm/README.md)。下面的命令均从仓库根目录运行。

## 环境

建议使用已验证的 Python 3.10 环境。根依赖固定了 LeRobot 提交，采集器使用 `lerobot.common.*` API。

```bash
python -m pip install -r requirements.txt
```

Xense 的厂商 SDK `xensesdk`、设备驱动与标定文件需在同一环境中单独准备。训练依赖另见模型目录说明；本仓库保留 `tacforce_wm/DINOv2/` 下已有的图像编码器权重。

## 启动硬件服务

先在一个终端启动服务：

```bash
python ur5_rg2/launch.py --robot ur \
  --camera-width 424 --camera-height 240 --camera-fps 15 \
  --xense-left-serial OG001783 --xense-right-serial OG001782
```

| 设备或服务 | 默认值 | 修改参数 |
| --- | --- | --- |
| UR 地址 | `192.168.1.102` | `--robot-ip` |
| 腕部 RealSense 序列号 | `412622271117` | `--wrist-camera-device-id` |
| 基座 RealSense 序列号 | `335522073597` | `--base-camera-device-id` |
| 机器人 ZMQ | `127.0.0.1:6001` | `--hostname`、`--robot-port` |
| 腕部/基座相机 ZMQ 端口 | `5000` / `5001` | `--wrist-camera-port`、`--base-camera-port` |
| 左/右 Xense ZMQ 端口 | `5100` / `5101` | `--left-tactile-port`、`--right-tactile-port` |

触觉源为 Xense。相机后端可选 `--camera-backend lerobot`（默认）或 `--camera-backend realsense`，`--enable-depth` 仅对 `realsense` 后端生效。Xense 也可用 `--xense-left-cam-id 0 --xense-right-cam-id 1` 按索引打开。

## 采集

在第二个终端运行，并按实际设备修改串口、IP 和序列号：

```bash
python ur5_rg2/collect.py \
  --repo-id ur5_rg2_xense_insert --root ./data/ur5_rg2_xense_insert \
  --task "Insert the black cylinder vertically into the hole." --num-episodes 20 \
  --force-serial-port /dev/ttyUSB0 --force-serial-timeout-s 0.1 --force-serial-retries 5
```

Windows 可将串口替换为实际的 `COM3` 等名称。保持键盘窗口处于焦点：

| 按键 | 操作 |
| --- | --- |
| `W/S/A/D/R/F` | 平移 TCP |
| 方向键、`Q/E` | 旋转 TCP |
| `Shift` | 按 `--shift-speed-scale` 加速 |
| `Space` | 切换 RG2 开合 |
| `P` | 开始记录一条 episode |
| `Enter` | 保存当前 episode |
| `Z` | 丢弃当前 episode 缓冲并停止记录 |
| `Esc` | 退出；正在记录且有帧时会保存 |
| `L` | 通过逆运动学移动至随机复位位姿 |
| `O` | 记录当前 TCP 姿态用于后续 `L` 复位 |
| `M` | 设置预设 TCP x/y 和竖直姿态，保持当前 z |
| `T/G` | 切换持续旋入/旋出，再次按同一个键停止 |
| `Y/H` | 按住旋入/旋出，松开停止 |

使用 `L/M` 前，根据实际工装校核 `ur5_rg2/collect.py` 中的 `RESET_TCP_POSITION`、`RESET_ORIENTATION_TCP_XY` 及力/触觉阈值。

常用参数：`--resume` 续采、`--no-preview` 关闭预览、`--no-collect-tactile` 禁用触觉采集、`--no-collect-force` 禁用力采集。默认启用触觉、力采集和力安全控制；力源可选 `--force-source modbus-serial`（默认）、`modbus-tcp` 或 `ur-rtde`。

数据保存到 `--root`，默认 `./data/ur5_rg2_xense_insert`。采集器覆盖已有数据、合并器使用 `--overwrite` 时，均通过 `data_tools/deletion.py` 显示目录与数据摘要，并要求输入 `confirm` 才删除。

## 数据字段

| 字段 | 内容 |
| --- | --- |
| `observation.image` / `observation.wrist_image` | 两路 RGB 图像，默认 `256 × 256 × 3` |
| `observation.state` | 执行动作后读取的 6 个关节角 |
| `action` | 执行动作后读取的 6 个关节角和夹爪标量，7 维 |
| `obj_init` | 9 维辅助字段 |
| `observation.tactile_{left,right}.force` | 触觉力场，默认 `35 × 20 × 3` |
| `observation.tactile_{left,right}.force6d` | Xense 六维合力/力矩 |
| `observation.tactile_{left,right}.depth` / `.rectify` | 深度与校正图像 |
| `observation.tactile.timestamp` / `.host_timestamp` | 双触觉设备及主机时间戳 |
| `observation.force_torque` | 力传感器六维力/力矩 |

## 合并数据集

合并两个结构兼容的 LeRobot v2.1 数据集，重新编号 episode、frame、task 和元数据。输入目录需显式指定；默认合并全部 episode，输出到 `data/merged/`：

```bash
python data_tools/merge.py \
  --first-root ./data/session_a --second-root ./data/session_b \
  --output-root ./data/merged
```

如需筛除数据，使用 `--drop-second-episodes 31` 排除第二个输入的 episode，或 `--drop-merged-episodes 0-19` 按合并后的索引排除。两者均接受逗号分隔的索引和范围，默认不筛除。输出目录必须与输入目录不同。采集器将图像存入 Parquet；合并器不复制独立视频文件。

完整参数：

```bash
python ur5_rg2/launch.py --help
python ur5_rg2/collect.py --help
python data_tools/merge.py --help
```

`.gitignore` 统一管理本机数据、训练输出、缓存与日志；已有 DINOv2 权重保留在版本控制中。
