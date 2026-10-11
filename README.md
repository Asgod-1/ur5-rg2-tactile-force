# UR5 + RG2 + Xense 真机数据采集

键盘遥操作 UR5 和 RG2，采集双 RealSense RGB 图像、双 Xense 触觉与 FTS-300-S 六维力/力矩，保存为 LeRobot 数据集。

采集入口为 `real_ur5rg2/collect_lerobot_smolvla_real_xenstactile_w_force.py`。文件名中的 SmolVLA 表示数据用途；采集时不需要下载或加载策略模型。

## 文件结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── dataset_delete_utils.py             # 数据集覆盖前的确认工具
├── real_ur5rg2/
│   ├── collect_lerobot_smolvla_real_xenstactile_w_force.py
│   ├── merge_lerobot_datasets.py        # 两个 LeRobot v2.1 数据集的合并工具
│   └── experiments/
│       └── launch_nodes.py             # UR、相机及 Xense 的 ZMQ 服务
└── gello/
    ├── cameras/                        # RealSense 驱动及 LeRobot 相机适配
    ├── force/                          # FTS-300-S Modbus RTU/TCP 驱动
    ├── robots/                         # UR RTDE、RG2 及机器人接口
    ├── tactile/                        # Xense SDK 适配及异步读取
    └── zmq_core/                       # 机器人、相机、触觉通信
```

## 环境

建议复用原项目已验证的 Python 3.10 采集环境。`requirements.txt` 沿用原项目的 LeRobot 固定提交和主要机器学习依赖版本，补齐直接使用的依赖，去掉仿真、绘图和其他触觉设备的专用依赖。

```bash
python -m pip install -r requirements.txt
```

Xense 的厂商 SDK `xensesdk` 需要单独安装在同一 Python 环境中。仓库不包含厂商 SDK、设备标定包、模型权重或采集数据。不要直接用新版 LeRobot 替换固定提交；采集器使用 `lerobot.common.*` API。

## 启动硬件服务

所有命令都从仓库根目录运行。先在一个终端启动服务：

```bash
python real_ur5rg2/experiments/launch_nodes.py --robot ur --camera-width 424 --camera-height 240 --camera-fps 15 --tactile-source xense --xense-left-serial OG001783 --xense-right-serial OG001782
```

| 设备或服务 | 当前默认值 | 修改参数 |
| --- | --- | --- |
| UR 地址 | `192.168.1.102` | `--robot-ip` |
| 腕部 RealSense 序列号 | `412622271117` | `--wrist-camera-device-id` |
| 基座 RealSense 序列号 | `335522073597` | `--base-camera-device-id` |
| 机器人 ZMQ | `127.0.0.1:6001` | `--hostname`、`--robot-port` |
| 腕部/基座相机 ZMQ 端口 | `5000` / `5001` | `--wrist-camera-port`、`--base-camera-port` |
| 左/右 Xense ZMQ 端口 | `5100` / `5101` | `--left-tactile-port`、`--right-tactile-port` |

启动器仅保留 Xense 触觉源，默认即为 `xense`；旧的 `--tactile-source xense` 命令仍可使用。相机保留 `--camera-backend lerobot`（默认）和 `--camera-backend gello` 两种后端。`--enable-depth` 仅对 `gello` 相机后端生效。

若设备需要按相机索引打开，可用 `--xense-left-cam-id 0 --xense-right-cam-id 1`。查看完整参数：

```bash
python real_ur5rg2/experiments/launch_nodes.py --help
```

## 采集

在第二个终端运行：

```bash
python real_ur5rg2/collect_lerobot_smolvla_real_xenstactile_w_force.py --repo-id ur5_rg2_xense_insert --root ./real_ur5rg2/data/ur5_rg2_xense_insert --task "Insert the black cylinder vertically into the hole." --num-episodes 20 --force-serial-port /dev/ttyUSB0 --force-serial-timeout-s 0.1 --force-serial-retries 5
```

`--force-serial-port` 使用实际 FTS-300-S 串口；上面是 Linux 示例，Windows 可以使用 `COM3` 等实际端口名。Xense、RealSense 和 UR 的 IP/序列号也需对应当前硬件。

保持键盘窗口处于焦点：

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

采集脚本中的 `RESET_TCP_POSITION`、`RESET_ORIENTATION_TCP_XY` 和力/触觉阈值沿用原项目。使用 `L/M` 前需根据实际工装校核预设位姿。主采集脚本的控制、复位、数据格式和安全逻辑均保持原样。

常用参数：`--resume` 续采、`--no-preview` 关闭图像预览、`--no-collect-tactile` 禁用触觉采集、`--no-collect-force` 禁用力采集。默认启用触觉、力采集和力安全控制；FTS-300-S 可通过 `--force-source modbus-serial`（默认）或 `modbus-tcp` 读取，也可选择 `ur-rtde`。

```bash
python real_ur5rg2/collect_lerobot_smolvla_real_xenstactile_w_force.py --help
```

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

数据保存到 `--root` 指定的目录；默认位于 `real_ur5rg2/data/`，由 `.gitignore` 排除。

## 合并数据集

保留原合并脚本。合并器读取两个结构兼容的 LeRobot v2.1 数据集，重新编号 episode、frame、task 和元数据，写入独立输出目录。目标采集器把图像存入 Parquet；合并器没有视频文件复制逻辑。

下面明确关闭原脚本针对旧实验的默认 episode 筛除规则，合并全部 episode：

```bash
python real_ur5rg2/merge_lerobot_datasets.py --first-root ./real_ur5rg2/data/session_a --second-root ./real_ur5rg2/data/session_b --output-root ./real_ur5rg2/data/merged --drop-second-episodes "" --drop-merged-episodes ""
```

原脚本默认 `--drop-second-episodes 31`，并默认 `--drop-merged-episodes 0-19`；按需显式传入要排除的 episode 索引或范围。输出目录必须与输入目录不同。

## Git

仅提交源码、依赖清单和说明。`.gitignore` 排除采集数据、训练输出、模型权重、缓存、日志、SDK 和压缩包。新仓库不携带旧提交历史或旧远程地址；配置新远程后再上传。
