# DWA Local Planner Implementation Plan

## Context

Building a DWA (Dynamic Window Approach) local planner for the dwaplanner project that integrates with the voxsense traversability grid system. The user wants to:
1. Design a local planner based on the traversability map in `/home/pc/code/voxsense`
2. Visualize generated local paths and velocity commands
3. Set up git version control and conda environment

The voxsense system provides a 2.5D traversability grid with 8-directional passability information from point cloud data.

---

## Implementation Approach

### 1. Project Setup
- Create `/home/pc/code/dwaplanner` directory
- Initialize git repository
- Create conda environment with Python 3.10, numpy, open3d
- Import and reuse voxsense modules (TraversabilityGrid, visualization utilities)

### 2. DWA Algorithm Design

**Robot Model:** Differential drive with state (x, y, θ) and control (v, w)

**Dynamic Window:**
- Max velocities: v_max=1.0 m/s, w_max=1.0 rad/s
- Acceleration limits: v_acc=0.5 m/s², w_acc=1.0 rad/s²
- Sample 7 linear × 11 angular velocities = ~77 trajectories

**Trajectory Simulation:**
- 2-second prediction horizon with 0.1s timesteps
- Simulate arcs (circular motion) or straight lines
- Check collision against traversability grid

**Collision Checking:**
- Query grid for each trajectory point
- Validate cell occupancy (state != EMPTY)
- Check directional passability using passable_mask[row, col, direction]

**Cost Function:**
```
score = α_heading × heading_alignment +
        α_clearance × obstacle_distance +
        α_velocity × speed_preference
```
Default weights: heading=1.0, clearance=0.5, velocity=0.2

### 3. Code Structure

**Three minimal files:**

1. **dwa_planner.py** - Core algorithm
   - `DWAConfig` dataclass: parameters
   - `DWAResult` dataclass: output (best velocity, trajectory, all candidates)
   - `DWAPlanner` class:
     - `plan(grid, goal_pos)` - main entry point
     - `_compute_dynamic_window()` - velocity bounds
     - `_sample_velocities()` - generate candidates
     - `_simulate_trajectory(v, w)` - forward simulation
     - `_check_collision(trajectory, grid)` - validation
     - `_compute_score(trajectory, v, w, goal, grid)` - scoring

2. **dwa_visualizer.py** - Visualization
   - `build_trajectory_lines()` - color-coded trajectories (red→yellow→green by score, cyan for best)
   - `build_velocity_arrow()` - orange arrow showing selected command
   - `build_goal_marker()` - magenta sphere at goal
   - `visualize_dwa_result()` - integrate with grid visualization

3. **test_dwa_planner.py** - Demo script
   - Load point cloud and build traversability grid
   - Create DWA planner with config
   - Set goal position (command-line args)
   - Run planner and visualize result

### 4. Grid Integration

**Coordinate conversion:**
```python
def world_to_grid(x, y, grid):
    ix = int(floor(x / grid.voxel_size))
    iy = int(floor(y / grid.voxel_size))
    row = iy - grid.min_iy
    col = ix - grid.min_ix
    return row, col
```

**Direction matching:**
Map motion vector (dx, dy) to nearest of 8 cardinal directions, then check `grid.passable_mask[row, col, direction_idx]`

**Clearance computation:**
Search nearby cells for EMPTY state, compute Euclidean distance

### 5. Visualization Design

**Overlay on traversability grid:**
- Reuse `build_grid_mesh()` from voxsense for base grid
- Add trajectory lines (Open3D LineSet)
- Add goal marker (sphere)
- Add velocity arrow (mesh arrow)
- Add coordinate frame

**Color scheme:**
- Trajectories: red (low score) → yellow → green (high score)
- Best trajectory: cyan
- Velocity arrow: orange
- Goal: magenta

### 6. Verification

**Test scenarios using voxsense point clouds:**
1. Flat ground - validate straight-line planning
2. Pillars - test obstacle avoidance
3. Wall - test turning behavior
4. Stairs - test blocked direction handling

**Expected output:**
- Velocity command: (v, w) in m/s and rad/s
- Best trajectory: list of (x, y, θ) poses
- Visualization: 3D window showing grid + trajectories + goal

---

## Critical Files

**From voxsense (reuse):**
- `/home/pc/code/voxsense/test_directional_traversability_vis.py`
  - TraversabilityGrid, build_traversability_run(), build_grid_mesh(), CellState, DIRECTION_UNIT_VECTORS

- `/home/pc/code/voxsense/test_single_frame_voxel_vis.py`
  - load_point_cloud_xyz(), rotate_ccw_90_about_z(), remove_origin_points()

**To create:**
- `/home/pc/code/dwaplanner/dwa_planner.py` (~200 lines)
- `/home/pc/code/dwaplanner/dwa_visualizer.py` (~150 lines)
- `/home/pc/code/dwaplanner/test_dwa_planner.py` (~100 lines)
- `/home/pc/code/dwaplanner/README.md` (usage instructions)
- `/home/pc/code/dwaplanner/requirements.txt` (numpy, open3d)

---

## Implementation Sequence

1. **Setup** - Create directory, git init, conda environment
2. **Core DWA** - Implement planner class with trajectory simulation and basic collision checking
3. **Grid Integration** - Add direction passability and clearance computation
4. **Visualization** - Implement trajectory rendering and integrate with grid visualization
5. **Testing** - Run on test scenarios and tune parameters

---

## Minimal Scope

**Included:**
- Single-goal planning
- Static environment
- Differential drive kinematics
- 8-direction passability checking
- Basic cost function

**Excluded (keep it simple):**
- Path smoothing (DWA output is smooth)
- Dynamic obstacles
- Recovery behaviors
- Multi-goal planning
- Trajectory caching
