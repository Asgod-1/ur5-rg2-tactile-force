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
  --camera-width 640 --camera-height 480 --camera-fps 30 \
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
  --repo-id ur5_rg2_xense_insert_30hz --root ./data/ur5_rg2_xense_insert_30hz \
  --task "Insert the black cylinder vertically into the hole." --num-episodes 20 \
  --force-serial-port /dev/ttyUSB0 \
  --fps 30 --image-width 640 --image-height 480 --no-preview
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

常用参数：`--resume` 续采、`--no-preview` 关闭预览、`--no-collect-tactile` 禁用触觉采集、`--no-collect-force` 禁用力采集。默认启用触觉、力采集和力安全控制；力源可选 `--force-source fts300-stream`（默认）、`modbus-serial`、`modbus-tcp` 或 `ur-rtde`。

FT300-S 默认使用厂商规定的 `100 Hz` 连续流，串口仍为 `19200` 波特率；每包验证 CRC，传感器没有提供采样时间戳或帧号。Linux FTDI 适配器在进程持有串口期间尝试启用低延迟模式（本机 USB 等待由 `16 ms` 降到 `1 ms`），退出时恢复原设置；不支持时继续取流并输出提示，可用 `--no-force-serial-low-latency` 禁用。`--force-source modbus-serial` 可回到逐次问答；`--force-serial-retries`、寄存器和 function-code 参数仅用于这个旧模式。

串口默认超时为 `0.1` 秒，无需显式传入；它是等待上限，收齐后会提前返回。连续流启动后立即由独立线程接收，覆盖设备检查、数据集加载、等待续采选择和去零校准，避免这段时间串口积压；旧 Modbus 模式仍在去零后开始后台轮询。校准、控制与保护读取最新缓存。

CRC 错包会被丢弃并重新寻找校验通过的新包，保留未收完整的包；只发布重新同步后的连续有效包，不使用坏包或将错误前的旧包冒充新数据，并限频报告 CRC 累计数。等待超时仍无有效包则报错；样本从读取开始计龄，超过默认 `0.1` 秒（`--force-max-age-s`）或后台读取失败时，该轮跳过正常运动指令及记录。

启动及退出流模式各需约 `0.6 s`，在录制循环外完成；退出时停止连续流、恢复适配器原设置并关闭串口。初始化、校准或其他清理失败时仍会尝试关闭其余资源。数据字段和续采格式兼容。

RG2 开合在硬件节点的后台线程执行，录制持续进行。开合脚本执行期间手臂保持，手臂目标不排队；完成后根据当前位姿响应新的输入。再次切换开合时只保留最新待执行目标。若夹爪脚本失败或超时，终端会报告错误并禁用手臂指令，仍可保存和退出。启用本改动前需要重启硬件节点；退出硬件节点会取消未执行目标、等待当前夹爪操作完成并释放 RTDE 连接。

新采与续采均使用后台无损 PNG 写图，默认压缩级别为 `1`，减少编码对触觉推理的资源竞争；解码后的像素不变，文件通常会增大。可用 `--image-png-compress-level 6` 恢复原压缩级别。

触觉在一次 SDK 查询中同时取得处理时间与原始 `CaptureTimeStamp`，按原始捕获去重。两路 SDK 调用仍互斥，按请求顺序排队；ONNX 使用 intra-op `6`、inter-op `1`，关闭自旋。后台读失败、原始捕获超过默认 `0.15` 秒没有更新，或结果本身已超过该年龄时，读取会报告错误，由已有触觉保护逻辑处理；硬件节点可用 `--xense-max-age-s` 调整上限。

录制默认最多回看上一帧触觉（约 `33 ms`，实际随原始帧间隔波动），控制和保护始终使用最新触觉。没有新的可用捕获时跳过该条记录，避免用重复帧填满标称频率；可用 `--no-tactile-recording-buffer` 关闭录制缓冲。启用新触觉逻辑前须重启 `launch.py`。

保存 episode 时，原始捕获时间、最新捕获时间及实际记录时刻另存到 `meta/tactile_timing/episode_XXXXXX.json`，用于核查原始帧重复、缓冲延迟与实际记录频率。标准 Parquet 字段保持兼容，原数据集可继续续采；旧 episode 没有这些时间记录。该文件是额外的审计元数据，现有合并器只合并标准数据字段。

录制默认启用软件时间配对：以左触觉原始捕获时间为参考，从短历史中选择最近的 RGB、右触觉、机器人状态和力样本；禁用触觉时以较早的当前 RGB 帧时间为参考。每路到参考时间默认最多 `20 ms`（`--recording-max-skew-ms`），两路之间的差值可能更大。没有可配对的新 RGB/触觉时跳过该行，不等待未来帧，也不重复旧图补帧。开始一条 episode 时需约 `0.1 s` 历史预热，实际录制频率可能略低于目标值。控制与保护仍使用最新样本，没有新增固定等待。夹爪标签取参考时刻已存在的请求值，机械开合完成时间仍未测量。

双触觉先读取快照，再进行图像缩放，减少两次读取之间的计算间隔。旧左帧无法配对、但较新左帧可能符合窗口时，最多重新选择一次左帧；保持原来的时间阈值、捕获去重和一帧缓冲。UR 状态历史按控制器原始包时间去重，避免同一个包重复读取时，主机时钟转换的微小波动被误判为时钟倒退。

原本因缺少新触觉或时间配对失败而要丢行时，录制端先重新读取当前双触觉快照，尝试用已到达的新捕获配对已读取的 RGB、状态和力历史。若仍配对失败，且双触觉均有候选，则最多补试一次原有一帧缓冲中较旧、未使用的左捕获，并为它重新选择右触觉和整行历史；现有成功路径不增加读取。补试最多额外查询左、右快照各一次，未找到较旧左候选则不查询右侧。成功后再提交游标；同一捕获的缩放结果可复用。时钟异常、无效帧及传感器读取错误不按缺帧重试。重选只影响记录，控制和保护仍使用本轮最初的最新值；保持原来的 `20 ms` 阈值、去重和一帧缓冲，不等待未来新帧。新快照有额外 RPC/缩放开销，不能消除所有真实缺帧或高负载跳行。

按 `P` 后先准备配对历史，终端提示预热中；预热等待的轮次不写入 Parquet 或逐帧时间文件，也不计入正式录制的循环、跳行及阶段耗时统计。第一组完整且有效的配对数据就绪后，先提示 `Recording ready`，再把这组数据保存为第 `0` 帧、训练 `timestamp=0`；后续按正常规则录制。就绪时保留已建立的历史，不再清空重做预热，也不额外丢弃这组有效样本。看到就绪后再开始任务动作，以免开头约 `0.1 s` 的动作未被完整记录。

预热时长、等待轮次及原因另存在录制性能文件的 `preparation` 中，`ready_host_timestamp` / `ready_monotonic` 记录就绪提示前的时刻；正式统计保留首条有效记录这一轮的读取、配对和重选耗时。仅首条数据前、开始后 `0.5 s` 内的已知录制边界预热不打印跳帧告警，持续不就绪及真实读取、时钟错误仍照常提示。就绪后的真实跳行仍计入正式统计；没有新增固定等待。旧数据及其统计不回写，更新后只需重启采集进程。

较旧候选补试在采集端执行；更新该逻辑后只需重新启动采集进程，沿用原参数，硬件节点可以继续运行。

每条保存的 episode 另有 `meta/recording_performance/episode_XXXXXX.json`，汇总录制循环频率、跳行原因、各阶段耗时、保护/夹爪忙计数和力串口 CRC 增量，并在 `recording_retry_counts` 记录重选尝试、恢复、失败次数。其中 `older_candidate_attempts` / `older_candidate_recovered` / `older_candidate_failed` 是较旧候选补试的子集计数，不与总重选次数相加；对应恢复行在传感器时间文件中标记 `recording_older_candidate_used`。`alignment` 已包含 `recording_retry`，后者包含 `tactile_retry_format`，分析耗时时不要重复相加。统计在内存累计，保存 episode 时写出，供检查真实操作期间掉行的来源；这些统计不参与训练。

共同参考时间、各路所选样本时间、偏差和保存入口时间另存 `meta/sensor_timing/episode_XXXXXX.json`。RGB/触觉使用 SDK 帧时间。UR 后台跟踪控制器原生 `timestamp`，用最快到达包的时间下沿估计主机时钟偏移及漂移，给每个状态包固定的映射时间；控制器重启或映射失效时清空历史并重新预热。无法仅凭单向接收消除恒定网络延迟。本机 CB3 控制器状态更新为 `125 Hz`，录制与发送动作仍为 `30 Hz`。

连续流的力样本时间由完整包接收时刻减去 `16` 字节的串口传输时间（约 `8.33 ms`）估计；USB 批量到达的包按标称 `100 Hz` 回推，另存接收时刻和主机接收计数。此计数不是设备帧号，估计也未消除传感器内部采样/滤波延迟；旧 Modbus 模式保留事务中点估计。因此软件时间偏差不等同物理采样误差或硬件同步。训练继续使用标准 Parquet `timestamp`（帧号/配置帧率），真实间隔和跳帧仅通过附加时间文件检查。附加时间文件不会由当前合并器自动合并。旧 episode 不回写，可继续续采。

启用时间配对前须重启 `launch.py` 和采集进程，不需要新增启动参数。可用 `--no-recording-time-alignment` 回到此前按读取顺序拼行的方式；与 `--no-tactile-recording-buffer` 一起使用可对照无触觉缓冲的旧行为。

采集端通过 RGB 专用接口读取两路相机，返回 RGB 和原生帧时间，不再分配或传输未使用的全零 depth 占位图；默认双路 `640×480 @ 30 fps` 减少约 `36.9 MB/s` 的本机数组传输。原有 RGB/深度接口及真实深度返回保留，触觉 depth 照常记录。启用此优化须重启 `launch.py` 和采集进程，无需新增参数；已有数据可以续采。

数据保存到 `--root`，默认 `./data/ur5_rg2_xense_insert_30hz`。相机默认以 `640 × 480`、`30 fps` 取流，保存的 RGB 图像也是 `640 × 480`，采集目标频率为 `30 Hz`；RGB 保存尺寸可用 `--image-width`、`--image-height` 修改。触觉 Rectify 保存尺寸仍默认 `256 × 256`。已有的 `./data/ur5_rg2_xense_insert` 使用 `20 fps` 元数据和 `256 × 256` RGB；改为新配置时使用新目录，继续旧数据集时显式传入原 `--repo-id`、`--root`，并使用 `--fps 20 --image-size 256`。采集器覆盖已有数据、合并器使用 `--overwrite` 时，均通过 `data_tools/deletion.py` 显示目录与数据摘要，并要求输入 `confirm` 才删除。

## 数据字段

| 字段 | 内容 |
| --- | --- |
| `observation.image` / `observation.wrist_image` | 两路 RGB 图像，默认高 × 宽 × 通道为 `480 × 640 × 3` |
| `observation.state` | 执行动作后读取的 6 个关节角 |
| `action` | 执行动作后读取的 6 个关节角和夹爪标量，7 维 |
| `obj_init` | 9 维辅助字段 |
| `observation.tactile_{left,right}.force` | 触觉力场，默认 `35 × 20 × 3` |
| `observation.tactile_{left,right}.force6d` | Xense 六维合力/力矩 |
| `observation.tactile_{left,right}.depth` / `.rectify` | 深度与校正图像 |
| `observation.tactile.timestamp` / `.host_timestamp` | 双触觉 SDK 处理结果时间及主机取得结果的时间；原始捕获另见 `meta/tactile_timing/` |
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
