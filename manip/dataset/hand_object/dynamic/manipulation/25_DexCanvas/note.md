1. 数据采集链路上；
14 mocap keypoint输入mano optimazer得到21个joint pos；这个joint pos和mediapipe不一致，拓扑一致但是需要修正坐标系（mediapipe是相机坐标，mano是世界坐标）
2. dexcanvas数据模态：每一帧包含：
- **手部姿态**: MANO 参数（shape 10D + pose 45D + translation 3D）

shape是手的外形属性，捏脸参数
pose是3d轴角（15*3）
TRANSLATION 是world frame to 手腕


- **物体状态**: 6-DoF 位姿
- **关节坐标**: 21 个关节的 3D 坐标
- **注意力掩码**: 有效帧标记

其中手部姿态是urdf解出来的；关节坐标是1求解出来的；注意力是提示contact状态的


  "The metadata specifies... along with frame ranges marking active
  manipulation periods"

  生成方式：基于物体运动自动检测 + 可能的人工校验

  从 data_format.md 的元数据结构：
  trajectory_meta_data = {
      'object_move_start_frame': int,  # 有效操作起始帧
      'object_move_end_frame': int,    # 有效操作结束帧
      ...
  }

  推断的检测逻辑：
  1. 物体位姿变化速度 > 阈值 → 标记为操作开始
  2. 物体位姿变化速度 < 阈值持续 N 帧 → 标记为操作结束
  3. attention_mask[start:end] = 1.0

  为什么原始数据会有这个字段：
  - 采集流程是：放置物体 → 执行操作 → 放下物体 → 结束
  - 开头/结尾的准备阶段不是有效操作数据
  - 论文提到 "excluding trials with drops or significant
  occlusions"，说明有质量筛选过程

  论文没有明确说明是纯自动还是有人工校验，但基于 12,000
  条序列的规模，很可能是：
  - 自动检测 + 质量评分（rating 字段）+ 异常剔除

3. mediapipe的世界系定义在相机下；mano的定义是没有没有相机的；

4. retargeting的wuijhand末端在完美求解下也不一定等于mediapipe末端，由于scaling和多优化目标的问题；retargetig大概过程：mediapipe输入21个keypoint 的世界系坐标，将其中的手上点减去wrist坐标得到相对的position ，再使用其中的部分点估计手掌方向（上或下），之后做旋转来对齐urdf的坐标系定义，之后将这些坐标加入优化流程；