# 合金 Figure 4 复现工作流

本文档记录当前代码中已经实现的工作流和规则。当前实现覆盖 Figure 4 的 `w/o CGFM` 与 `w/ CGFM` 两条曲线；Figure 5 / DDTS 尚未实现，但本文档已给出参考 Figure 4 代码落地 Figure 5 的复现工作流和验收边界。

## 项目状态

- Figure 4 已实现两条 setting：`wo_cgfm` 和 `w_cgfm`。
- Figure 5 尚未有可运行代码；复现时应优先复用 Figure 4 的编码、FM、QUBO、SA、checkpoint 和输出布局能力，并新增多目标 scalarization / Pareto 分析层。
- 当前代码使用 schema v2 checkpoint 和 schema v2 summary；不再兼容旧 checkpoint/summary schema。
- 旧成功输出目录受保护，可查看结果，但不保证新版代码可以直接 `--resume`。
- 运行入口是 `figure4_runner.py`，画图入口是 `plot_figure4.py`。

## 代码结构

- `alloy_dataset_generator.py`：生成合金组成行，并计算 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T`。
- `figure4_config.py`：定义 preset、配置 dataclass、`ObjectiveSpec` 和 canonical `OBJECTIVES` 顺序。
- `figure4_settings.py`：定义 `wo_cgfm` 和 `w_cgfm` 的 setting strategy。
- `figure4_qubo_math.py`：定义 one-hot 编码、CGFM 角度映射、QUBO 构造、惩罚项和 simulated annealing。
- `figure4_fm_torch.py`：定义 PyTorch FM、Optuna 调参、LBFGS 训练和 FM-to-QUBO 转换。
- `figure4_pipeline.py`：运行 trajectory、恢复 checkpoint、验收或替换候选、聚合结果、写 summary。
- `figure4_outputs.py`：统一输出路径、checkpoint schema 和 I/O、summary 读取、文件日志和输出目录分类。
- `figure4_runner.py`：解析 CLI、配置日志、检查训练依赖和 CUDA 可用性、调用 pipeline、写 manifest。
- `plot_figure4.py`：读取一个或多个 schema v2 summary，绘制 Figure 4 曲线。

Figure 5 建议新增独立模块，不直接把多目标逻辑塞进 `figure4_pipeline.py`：

- `figure5_config.py`：定义 Figure 5 preset、目标集合 `kappa/E/rho`、DDTS 和 weighted-sum 配置。
- `figure5_scalarization.py`：定义 `w_ddts` 与 `wo_ddts` 的人工训练目标构造。
- `figure5_pipeline.py`：复用 Figure 4 的底层训练/求解组件，运行多目标 active learning。
- `figure5_outputs.py`：定义 Figure 5 summary、checkpoint、manifest、日志和输出图路径。
- `figure5_runner.py`：解析 Figure 5 CLI，调用 Figure 5 pipeline。
- `plot_figure5.py`：读取 Figure 5 summary，绘制 3D objective space、Pareto front 和分段采样图。

## Figure 4 算法流程

1. `figure4_runner.py` 根据 preset 和 CLI override 解析 `ExperimentConfig`。
2. `run_figure4_experiment` 为每个 seed 生成初始数据集。
3. 初始行用 `discretize_rows_for_figure4` 离散化；底层 `prepare_discrete_composition` 将四个相分数放到 `num_levels` 网格上，并保持总和为 1。
4. trajectory 顺序是 `seed -> objective -> setting`。
5. 每轮 active learning 创建 setting-specific encoding：
   - `wo_cgfm`：直接编码 4 个相分数 block，特征维度是 `4 * num_levels`。
   - `w_cgfm`：编码 3 个 CGFM 角度 block，特征维度是 `3 * num_levels`。
6. 当前数据集被 strategy 编码后，objective 被转换成训练用的最小化目标，并对 target 做 z-score。
7. `fit_torch_fm` 训练 FM，`fm_to_qubo` 将 FM 转换为 QUBO 项。
8. `build_single_objective_qubo` 构造最终 QUBO：
   - `wo_cgfm`：`FM + system penalty + one-hot penalty`。
   - `w_cgfm`：`FM + one-hot penalty`。
9. `simulated_annealing_qubo` 求解候选 bit vector。
10. setting strategy 解码候选。每个 block 允许全零表示 0；一个 block 中多于一个 active bit 为非法。
11. 候选必须解码为非负四相分数、总和为 1，且不能重复已有 composition。
12. 非法或重复候选使用 random replacement；replacement 也会检查 `seen_compositions`，避免加入重复 composition。
13. 追加候选行或 replacement 行后更新 `best_so_far`，并写入 trajectory checkpoint。
14. 结果按 `{objective}:{setting}` 聚合为 best-so-far 的 mean、std、min、max 曲线。

## 配置规则

- preset 包括 `paper`、`quick_l50`、`test`。
- `paper`：`num_samples=100`，`iterations=600`，`num_levels=50`，`optuna_trials=20`，`sa_runs=1000`，`sa_sweeps=3000`；默认 seed 数是 20。
- `quick_l50`：`num_samples=100`，`iterations=100`，`num_levels=50`，`optuna_trials=3`，`sa_runs=100`，`sa_sweeps=500`；默认 seed 数是 3。
- `test`：`num_samples=10`，`iterations=2`，`num_levels=8`，`optuna_trials=0`，`sa_runs=2`，`sa_sweeps=6`；默认 seed 数是 1。
- CLI 数值参数会覆盖 preset，例如 `--iterations`、`--num-levels`、`--optuna-trials`、`--sa-runs`、`--sa-sweeps`。
- `--settings` 必填，可选值是 `wo_cgfm` 和 `w_cgfm`。
- `--objectives` 可选；不传时按 `figure4_config.OBJECTIVES` 的 canonical 顺序运行全部五个 objective。
- `EncodingConfig` 要求 `num_levels > 1`。
- `ExperimentConfig` 要求 `num_samples` 和 `iterations` 为正数。
- `FMConfig` 要求 `optuna_trials >= 0`，且 `device` 为 `cpu` 或 `cuda`。
- `SAConfig` 要求 `runs` 和 `sweeps` 为正数。
- runner 不强制解释器名称；如果传入 `--device cuda` 但 PyTorch CUDA 不可用，会在训练前失败。

## Figure 4 运行方法

推荐用 `quick_l50` 同时运行两条曲线：

```powershell
python figure4_runner.py `
  --output-dir figure4_compare_quick_l50 `
  --device cuda `
  --resume `
  --preset quick_l50 `
  --settings wo_cgfm w_cgfm
```

只运行 `w/o CGFM`：

```powershell
python figure4_runner.py `
  --output-dir figure4_wo_cgfm_quick_l50 `
  --device cuda `
  --resume `
  --preset quick_l50 `
  --settings wo_cgfm
```

只运行 `w/ CGFM`：

```powershell
python figure4_runner.py `
  --output-dir figure4_w_cgfm_quick_l50 `
  --device cuda `
  --resume `
  --preset quick_l50 `
  --settings w_cgfm
```

画单个 summary：

```powershell
python plot_figure4.py `
  --summary figure4_compare_quick_l50\figure4_summary.json `
  --output figure4_compare_quick_l50\figure4.png
```

合并分别运行得到的两个 summary 画图：

```powershell
python plot_figure4.py `
  --summary figure4_wo_cgfm_quick_l50\figure4_summary.json figure4_w_cgfm_quick_l50\figure4_summary.json `
  --output figure4_compare_quick_l50\figure4.png
```

## 输出规则

`figure4_outputs.py` 定义统一输出布局：

- summary：`figure4_summary.json`
- manifest：`manifest.json`
- 默认图像：`figure4.png`
- runner 日志：`logs/figure4_runner.log`
- plot 日志：`logs/plot_figure4.log`
- checkpoint：`trajectories/{setting}_{objective}_seed_{seed}.json`

`classify_output_path(path)` 返回 `managed_current`、`protected_legacy`、`temporary_test` 或 `unknown`。

受保护旧输出包括 `figure4_quick_l50`、`figure4_cgfm_quick_l50`、`figure4_compare_l50_from_separate`，以及名字包含 `quick_150` 或 `quick150` 的目录。这些目录可用于查看历史结果，但新版代码不保证读取旧 checkpoint 或旧 summary schema。

## 命名与 schema 规则

- setting 名称是 `wo_cgfm` 和 `w_cgfm`。
- objective 名称是 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T`。
- summary 聚合 key 是 `{objective}:{setting}`，例如 `kappa:wo_cgfm`。
- summary schema 和 checkpoint schema 都是 version 2。
- 聚合曲线字段是 `best_so_far_mean`、`best_so_far_std`、`best_so_far_min`、`best_so_far_max`。
- replacement 计数字段是 `duplicate_replacements`、`invalid_replacements`、`random_replacements`、`random_replacement_draws`、`accepted_sa_candidates`、`completed_iterations`。

## 返回类型与调用规则

- `QuboStats` 和 `QuboBuildResult` 描述构造出的 QUBO。
- `IterationEncoding` 描述一轮 one-hot block 布局和可选 CGFM phase permutation。
- `TrajectoryState` 是可 checkpoint 的 resumable state。
- `TrajectoryResult` 是写入 summary 的单条 trajectory 结果。
- `CandidateDecision` 记录候选是 accepted、invalid replacement 还是 duplicate replacement。
- `TrainingIterationResult` 保存一轮训练的 encoding、SA candidate bits、FM metadata 和 QUBO stats。
- `AggregatedTrajectory` 保存 best-so-far 曲线的 mean、std、min、max。
- `Figure4Summary` 是序列化为 JSON 的顶层 summary dataclass。
- `load_checkpoint` 校验 schema、metadata、setting、config 和 state 后返回原始 checkpoint payload。
- `load_summary` 返回原始 summary JSON dict；plot 要求 schema version 2 且 `aggregated` 非空。
- `plot_figure4.py` 可读取一个或多个 schema v2 summary，合并不重复的 aggregated key 后画图。
- `delta_alpha` 和 `delta_T` 绘图时使用 log y 轴，并仅在绘图层把显示值 clamp 到 `1e-12`；JSON 保留真实值。

## 测试与验收

运行完整单元测试：

```powershell
python -m unittest discover -v
```

运行 Figure 4 小规模验收：

```powershell
python figure4_runner.py `
  --output-dir figure4_refactor_test `
  --device cuda `
  --resume `
  --settings wo_cgfm w_cgfm `
  --objectives kappa `
  --preset test
```

```powershell
python plot_figure4.py `
  --summary figure4_refactor_test\figure4_summary.json `
  --output figure4_refactor_test\figure4.png
```

## Figure 5 复现工作流

当前代码尚未实现 Figure 5 / DDTS；本节定义后续实现时应遵守的目标、复用边界、数据流、输出格式和验收标准。Figure 5 不应复用 Figure 4 的 `best_so_far` 单目标曲线作为核心结果，因为 Figure 5 比较的是多目标优化中采样到的解在三维目标空间中的覆盖情况。

### Figure 5 目标

- 复现论文 Figure 5 的多目标 FM+QO 对比：`w/ DDTS` 与 `w/o DDTS`。
- 多目标集合固定为 `kappa`、`E`、`rho`：
  - `kappa`：越大越好。
  - `E`：越大越好。
  - `rho`：越小越好。
- 默认 paper 规模应对齐论文 Figure 5：
  - 初始数据集：`num_samples=500`。
  - active learning：`iterations=1000`。
  - 编码：优先复用 Figure 4 的 `w_cgfm` 编码，即 3 个 CGFM 角度 block，每个 block `num_levels=50`。
  - QUBO 求解：复用 D-Wave Ocean `neal` simulated annealing。
- 输出重点不是单目标 best-so-far，而是：
  - 每次迭代采样到的合金设计。
  - 每个设计的 `kappa/E/rho` 真实值。
  - `w_ddts` 和 `wo_ddts` 各自找到的非支配 Pareto front。
  - 按迭代区间分段的采样进展。

### 与 Figure 4 的复用边界

Figure 5 应直接复用：

- `alloy_dataset_generator.py` 中的真实物性计算和 `build_dataset_row`。
- `figure4_qubo_math.py` 中的 CGFM 编码、one-hot 约束、FM QUBO 构造和 SA 求解。
- `figure4_fm_torch.py` 中的 FM 训练和 `fm_to_qubo`。
- `figure4_outputs.py` 的输出布局思想、checkpoint 原子写入、日志配置和 schema 校验风格。
- `figure4_pipeline.py` 中的候选验收思想：decode 后必须得到非负、总和为 1 的四相 composition；非法或重复候选用 random replacement 处理。

Figure 5 应新增而不是硬塞到 Figure 4 中：

- 多目标初始数据生成器。
- 多目标 scalarization 策略。
- Pareto front 计算。
- 3D objective-space summary 和 plot schema。
- Figure 5 专用 runner、summary 和 checkpoint dataclass。

### 初始数据生成

Figure 4 的单目标初始数据使用 sorted uniform cuts，论文 Figure 5 的多目标实验应使用更大的初始数据集和更偏向大相分数覆盖的随机生成器。建议在 `alloy_dataset_generator.py` 增加：

```python
generate_initial_dataset_multi_objective(
    num_samples: int = 500,
    seed: int | None = None,
)
```

推荐生成流程：

1. 固定 Al matrix 为 80%，四个次相归一化分数总和为 1。
2. 先从 `U(0, 1)` 生成第一个临时分数。
3. 后续临时分数从剩余区间中逐步生成。
4. 最后一个临时分数取剩余量。
5. 打乱四个分数分配到 Si、Mg2Si、Al3Ni、Al2Cu。
6. 用 `build_dataset_row` 计算所有物性。
7. 数据集必须只包含可行 composition。

这样可以比 Figure 4 的单目标初始生成器更容易覆盖靠近边界和大相分数的区域，避免多目标 Pareto front 初始覆盖过窄。

### Figure 5 setting

Figure 5 至少包含两个 setting：

- `w_ddts`：使用 DDTS，即 data-driven Tchebycheff scalarization。
- `wo_ddts`：使用标准 weighted-sum scalarization。

二者应使用相同的：

- seed 列表。
- 初始数据集。
- FM 配置。
- CGFM 编码配置。
- SA 配置。
- random replacement 规则。

这样 Figure 5 的差异只来自多目标 scalarization，而不是来自训练、编码或采样参数。

### `wo_ddts` weighted-sum 流程

每轮 active learning：

1. 当前数据集包含 composition 和真实 `kappa/E/rho`。
2. 生成一组 preference weights `w = (w_kappa, w_E, w_rho)`，要求非负且总和为 1。
3. 对三个目标统一成最小化方向：
   - `kappa_train = -kappa`
   - `E_train = -E`
   - `rho_train = rho`
4. 对每个目标在当前数据集上做 z-score。
5. 构造 weighted-sum 人工训练目标：

```text
y_hat = w_kappa * z(-kappa) + w_E * z(-E) + w_rho * z(rho)
```

6. 用 `y_hat` 训练一个 FM。
7. FM 转 QUBO。
8. 使用 CGFM setting 构造最终 QUBO：`FM + one-hot penalty`。
9. SA 求解候选 bit vector。
10. CGFM decode 为 composition。
11. 用真实物性模型验证候选，得到 `kappa/E/rho`。
12. 非法或重复候选使用 random replacement。
13. 把候选或 replacement 加入数据集和 checkpoint。

weighted-sum 是 Figure 5 的 baseline；它通常更容易集中在凸区域或目标空间边界附近。

### `w_ddts` DDTS 流程

DDTS 的核心是用 Tchebycheff 风格的“离 utopian point 最远的加权目标”构造人工训练目标，而不是直接对目标加权求和。

每轮 active learning：

1. 当前数据集包含 composition 和真实 `kappa/E/rho`。
2. 生成 preference weights `w = (w_kappa, w_E, w_rho)`，要求非负且总和为 1。
3. 计算或更新 utopian point `u`：
   - `u_kappa`：当前数据集中最大的 `kappa`。
   - `u_E`：当前数据集中最大的 `E`。
   - `u_rho`：当前数据集中最小的 `rho`。
4. 对每个数据点 `d` 计算到 utopian point 的方向一致距离：
   - `dist_kappa(d) = u_kappa - kappa(d)`
   - `dist_E(d) = u_E - E(d)`
   - `dist_rho(d) = rho(d) - u_rho`
5. 对距离做尺度归一化，避免某个目标因单位量纲支配训练目标。
6. 对每个数据点计算加权距离：

```text
t_kappa(d) = w_kappa * normalized_dist_kappa(d)
t_E(d) = w_E * normalized_dist_E(d)
t_rho(d) = w_rho * normalized_dist_rho(d)
```

7. 取最大加权距离作为该点的人工训练目标：

```text
y_hat(d) = max(t_kappa(d), t_E(d), t_rho(d))
```

8. 用 `y_hat` 训练一个 FM。
9. FM 转 QUBO。
10. 使用 CGFM setting 构造最终 QUBO：`FM + one-hot penalty`。
11. SA 求解、decode、真实物性验证、replacement、checkpoint，规则与 `wo_ddts` 一致。

DDTS 的预期效果是比 weighted-sum 更稳定地覆盖非凸 Pareto front 区域。

### Pareto front 计算

Figure 5 summary 必须保存所有被采样或替换后加入数据集的多目标解，并为每个 setting 计算非支配解。

对 `kappa/E/rho`，解 A 支配解 B 当且仅当：

```text
A.kappa >= B.kappa
A.E >= B.E
A.rho <= B.rho
```

并且至少一个目标严格更好。未被任何其他点支配的点属于 Pareto front。

实现建议新增：

```python
is_dominated(candidate, other) -> bool
pareto_front(points) -> list[points]
```

测试必须覆盖：

- 全部目标方向正确：`kappa/E` 最大化，`rho` 最小化。
- 完全相同点互不严格支配。
- 边界点和非凸区域点不会被错误删除。

### Figure 5 输出布局

建议 Figure 5 采用独立输出目录和 schema，不复用 `figure4_summary.json`：

```text
figure5_output_dir/
  figure5_summary.json
  manifest.json
  logs/
    figure5_runner.log
    plot_figure5.log
  trajectories/
    w_ddts_seed_0.json
    wo_ddts_seed_0.json
  figure5.png
```

`figure5_summary.json` 建议字段：

```json
{
  "schema_version": 1,
  "training_backend": "pytorch_fm_lbfgs",
  "config": {},
  "seed_list": [0],
  "objectives": ["kappa", "E", "rho"],
  "settings": ["w_ddts", "wo_ddts"],
  "solutions": [
    {
      "setting": "w_ddts",
      "seed": 0,
      "iteration": 1,
      "candidate_status": "accepted",
      "composition": [0.0, 0.2, 0.3, 0.5],
      "kappa": 130.0,
      "E": 80.0,
      "rho": 2.7
    }
  ],
  "pareto_front": {
    "w_ddts": [],
    "wo_ddts": []
  }
}
```

checkpoint 中必须保存：

- 当前 rows。
- 已加入的 solution records。
- 当前 iteration。
- random replacement 计数。
- 当前 setting、seed、config。
- 最近一轮 weights、utopian point、FM metadata 和 QUBO stats。

### Figure 5 绘图

`plot_figure5.py` 应生成一个接近论文 Figure 5 的 2x2 图：

1. panel a：`w_ddts` 全部采样点和 Pareto front。
2. panel b：`wo_ddts` 全部采样点和 Pareto front。
3. panel c：`w_ddts` 按迭代区间分段的采样进展。
4. panel d：`wo_ddts` 按迭代区间分段的采样进展。

默认分段：

```text
0-250
250-500
500-750
750-1000
```

坐标轴：

```text
x = kappa
y = E
z = rho
```

绘图规则：

- 灰色点：该 setting 找到的所有候选设计。
- 蓝色点：该 setting 的 Pareto front。
- 分段图用不同颜色表示迭代区间。
- random replacement 可以保存在 summary 中，但论文图中默认只画 QO/SA 找到的最佳 alloy designs；是否显示 replacement 应由 `plot_figure5.py` CLI 参数控制。

### Figure 5 运行命令（代码落地后）

代码尚未实现前，不应执行以下命令；这些命令定义的是目标 CLI 形态。

小规模验收：

```powershell
python figure5_runner.py `
  --output-dir figure5_test `
  --device cuda `
  --resume `
  --preset test `
  --settings w_ddts wo_ddts
```

论文规模：

```powershell
python figure5_runner.py `
  --output-dir figure5_paper `
  --device cuda `
  --resume `
  --preset paper `
  --settings w_ddts wo_ddts
```

画图：

```powershell
python plot_figure5.py `
  --summary figure5_paper\figure5_summary.json `
  --output figure5_paper\figure5.png
```

### Figure 5 验收标准

Figure 5 代码落地后，至少需要通过：

1. 单元测试：
   - 多目标初始数据生成器生成的每行 composition 非负且总和为 1。
   - weighted-sum 的目标方向正确。
   - DDTS 的 utopian point 和 `max(weighted distance)` 计算正确。
   - Pareto 支配关系正确。
   - checkpoint resume 不重复迭代、不丢失已有 solution records。
2. 小规模端到端验收：
   - `figure5_runner.py --preset test --settings w_ddts wo_ddts` 能生成 summary、manifest、日志和 checkpoint。
   - `plot_figure5.py` 能读取 summary 并生成非空 PNG。
3. 论文规模输出验收：
   - `figure5_summary.json` 中包含 `w_ddts` 与 `wo_ddts` 两个 setting。
   - 每个 setting 至少有 `1000` 条 iteration solution record。
   - `pareto_front.w_ddts` 和 `pareto_front.wo_ddts` 非空。
   - `figure5.png` 包含全局采样图和分段采样图。

### 实现顺序建议

1. 在 `alloy_dataset_generator.py` 增加 Figure 5 多目标初始数据生成器。
2. 新增 `figure5_scalarization.py`，先实现 weighted-sum，再实现 DDTS。
3. 新增 Pareto front 工具和测试。
4. 新增 `figure5_pipeline.py`，复用 Figure 4 的 FM/QUBO/SA/decode/replacement。
5. 新增 `figure5_runner.py` 和 `figure5_outputs.py`，保持 checkpoint、manifest、日志风格与 Figure 4 一致。
6. 新增 `plot_figure5.py`。
7. 先跑 `preset test`，再跑较小 quick preset，最后跑 paper preset。
