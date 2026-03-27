# dwaplanner

基于 `voxsense` 2.5D 可通行栅格的最小 DWA 本地规划器实现。项目包含：

- `dwaplanner/dwa_planner.py`: DWA 核心采样、轨迹仿真、方向可通行检查和评分。
- `dwaplanner/dwa_visualizer.py`: 基于 Open3D 的轨迹、速度指令、目标点可视化。
- `test_dwa_planner.py`: 读取点云构建 `voxsense` 栅格并运行一次规划的演示脚本。
- `tests/test_dwa_planner.py`: 不依赖点云文件的最小单元测试。

## 依赖

推荐直接复用上游 `voxsense` 环境：

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

默认会尝试从相邻仓库 `/home/pc/code/voxsense` 导入构图和可视化模块。

```bash
python test_dwa_planner.py /home/pc/code/voxsense/pcd/平地无障碍.pcd --goal-x 0.0 --goal-y 2.0
```

不打开 Open3D 窗口，只打印规划结果：

```bash
python test_dwa_planner.py /home/pc/code/voxsense/pcd/柱子.pcd --goal-y 2.5 --show-blocked-directions --no-vis
```

如果 `voxsense` 仓库不在同级目录，可显式指定：

```bash
export VOXSENSE_REPO=/path/to/voxsense
```

## 算法说明

机器人模型采用差速底盘 `(x, y, theta)` 与控制 `(v, w)`：

- 动态窗口默认限制：`v ∈ [0, 1.0] m/s`，`|w| <= 1.0 rad/s`
- 预测时域：`2.0 s`
- 仿真步长：`0.1 s`
- 采样密度：`7 x 11`

轨迹评分：

```text
score =
  heading_weight * heading_alignment +
  clearance_weight * obstacle_clearance +
  velocity_weight * speed_preference
```

其中碰撞检查同时使用两类信息：

- 栅格占据状态：轨迹必须始终位于非 `EMPTY` 的 cell 内。
- 方向可通行掩码：跨 cell 移动时必须满足 `passable_mask[row, col, direction] = True`。

## 测试

```bash
python -m unittest tests.test_dwa_planner
```
