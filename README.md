# dwaplanner

基于 `voxsense` 可通行栅格的最小 DWA 本地规划器实现。主流程会复用 `voxsense` 的点云建图，再把结果直接叠加到二维 BEV 图片上。项目包含：

- `dwaplanner/dwa_planner.py`: DWA 核心采样、轨迹仿真、方向可通行检查和评分。
- `dwaplanner/dwa_visualizer.py`: 把候选轨迹、最佳轨迹、起点、目标和速度箭头叠加到二维图片上。
- `dwaplanner/voxsense_adapter.py`: 导入相邻 `voxsense` 仓库里的建图和 BEV 渲染函数。
- `test_dwa_planner.py`: 读取点云、运行规划并输出带叠加轨迹的 BEV PNG。
- `tests/test_dwa_planner.py`: 基于二维栅格图的最小单元测试。

## 依赖

可以直接复用已有 conda 环境：

```bash
source ~/anaconda3/bin/activate
conda activate /home/pc/code/voxsense/.conda-env
```

如果需要单独环境：

```bash
source ~/anaconda3/bin/activate
conda env create -f environment.yml -p /home/pc/code/dwaplanner/.conda-env
conda activate /home/pc/code/dwaplanner/.conda-env
```

## 运行演示

默认会读取相邻 `voxsense` 仓库里的点云文件并生成 BEV 底图。

```bash
python3 test_dwa_planner.py /home/pc/code/voxsense/pcd/平地无障碍.pcd --goal-x 0.0 --goal-y 2.0
```

绕障示例：

```bash
python3 test_dwa_planner.py /home/pc/code/voxsense/pcd/柱子.pcd --goal-y 2.5
```

脚本会先调用 `voxsense` 的 traversability 构图，再生成二维 BEV 图，并把 DWA 结果直接叠加到同一张 PNG 上，输出到 `outputs/`。

主要参数：

```bash
python3 test_dwa_planner.py /home/pc/code/voxsense/pcd/平地无障碍.pcd \
  --voxel-size 0.15 \
  --pixels-per-meter 120 \
  --goal-y 2.0 \
  --output-path outputs/flat_dwa_bev.png
```

## 算法说明

机器人模型采用差速底盘 `(x, y, theta)` 与控制 `(v, w)`：

- 动态窗口默认限制：`v ∈ [0.5, 1.0] m/s`，`|w| <= 1.0 rad/s`
- 预测时域：`2.0 s`
- 仿真步长：`0.1 s`
- 采样密度：`7 x 11`

轨迹评分：

```text
score =
  heading_weight * heading_alignment +
  goal_progress_weight * goal_progress +
  clearance_weight * obstacle_clearance +
  velocity_weight * speed_preference
```

其中碰撞检查使用三类信息：

- `PASSABLE` 落脚区域：规划器会从 `state` 和 `passable_mask` 显式派生 cell 级 `PASSABLE` 区域；`EMPTY` 和仅“被观测到但没有可通行方向”的 cell 都视为不可落脚。
- 方向可通行掩码：跨 cell 移动时必须满足 `passable_mask[row, col, direction] = True`。
- 机器人半径：使用 `scipy.ndimage.distance_transform_edt` 预计算 clearance 距离场；若任一轨迹采样点的 clearance 小于 `robot_radius`，该轨迹直接判为非法。

`goal_progress` 表示一条候选轨迹在预测时域内让机器人离目标减少了多少距离。这个项的作用是避免机器人只是在原地保持朝向正确，却没有真正向目标推进。

## 测试

```bash
python3 -m unittest tests.test_dwa_planner
```
