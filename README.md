# 合金 Figure 4 / Figure 5 复现工作流

本文档按复现目标拆成两部分：第一部分记录 Figure 4 单目标复现的当前实现、代码结构和运行方法；第二部分记录 Figure 5 多目标复现的当前实现、代码结构和运行方法。

## 项目复现定位与运行环境

本项目不追求论文完整计算规模或逐点数值复现，目标是复现 Figure 4/5 的核心算法流程，并在可接受的计算预算下验证论文报告的定性趋势：

- `test`：单元测试和烟测规模，只验证代码路径、schema、checkpoint 和绘图，不用于趋势结论。
- `quick_l50` / `quick150`：项目实际运行规模，用于流程复现和定性趋势验证，不应解释为论文规模结果。
- `paper`：保留论文参数作为对照，不在本项目中实际运行，也不属于验收或交付范围。

论文补充材料使用老旧的 `fastFM + ALS`。当前 Conda 环境 `env_torch` 不提供可运行的 `fastFM`，因此项目使用 PyTorch 二阶 FM 和 LBFGS 作为工程替代，保留 rank 6、最多 2000 training steps、target z-score、Optuna 调参和 FM-to-QUBO 流程。训练后端和优化器与论文不同，因此不保证 active-learning trajectory 或最终数值与论文一致。

所有命令通过 Conda 环境 `env_torch` 运行。当前工作区验证环境为 Python 3.9.21、PyTorch 1.9.1+cu111，CUDA 可用；可用下列命令重新检查当前机器状态：

```powershell
conda run -n env_torch python -c "import importlib.util, sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('fastFM:', importlib.util.find_spec('fastFM'))"
```

输出目录约定：

- 单元测试和 synthetic 绘图产生的临时文件统一写入工作区 `.tmp_test/`；该目录被 Git 忽略，可以安全删除并由测试重新生成。
- 用户运行 `figure4_runner.py`、`figure5_runner.py` 和绘图脚本时，输出写入命令指定的工作区目录，例如 `figure4_compare_quick_l50/` 或 `figure5_quick150/`，不会重定向到 `.tmp_test/`。

## 第一部分：Figure 4 复现

### Figure 4 开发情况

- 已实现 `w/o CGFM` 和 `w/ CGFM` 两条流程，对应 setting 名称 `wo_cgfm` 和 `w_cgfm`。
- 已实现 schema v2 checkpoint 和 schema v2 summary；不再兼容旧 checkpoint/summary schema。
- 已实现 per-trajectory checkpoint、`--resume`、多 seed 聚合、mean/std/min/max 曲线和 Figure 4 绘图。
- 运行入口是 `figure4_runner.py`，绘图入口是 `plot_figure4.py`。
- 旧成功输出目录受保护，可查看结果，但不保证新版代码可以直接 `--resume`。

### Figure 4 代码结构

- `alloy_dataset_generator.py`：生成合金组成行，并计算 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T`。
- `figure4_experiment_config.py`：定义 Figure 4 preset、配置 dataclass、`ObjectiveSpec` 和 canonical `OBJECTIVES` 顺序。
- `figure4_setting_strategies.py`：定义 `wo_cgfm` 和 `w_cgfm` 的编码/解码 strategy。
- `figure4_qubo_math.py`：定义 one-hot 编码、CGFM 角度映射、QUBO 构造、惩罚项和 simulated annealing。
- `figure4_fm_torch.py`：定义 PyTorch FM、Optuna 调参、LBFGS 训练和 FM-to-QUBO 转换。
- `figure4_pipeline.py`：运行 trajectory、恢复 checkpoint、验收或替换候选、聚合结果、写 summary。
- `figure4_outputs.py`：统一输出路径、checkpoint schema 和 I/O、summary 读取、文件日志和输出目录分类。
- `figure4_runner.py`：解析 CLI、配置日志、检查训练依赖和 CUDA 可用性、调用 pipeline、写 manifest。
- `plot_figure4.py`：读取一个或多个 schema v2 summary，绘制 Figure 4 曲线。

### Figure 4 算法流程

1. `figure4_runner.py` 根据 preset 和 CLI override 解析 `ExperimentConfig`。
2. `run_figure4_experiment` 为每个 seed 生成初始数据集。
3. 初始行用 `discretize_rows_for_figure4` 离散化；底层 `prepare_discrete_composition` 将四个相分数放到 `num_levels` 网格上，并保持总和为 1。
4. trajectory 粒度是固定的 `setting + objective + seed`。
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

### Figure 4 配置规则

- preset 包括 `paper`、`quick_l50`、`test`。
- `paper`（论文参数参考，不运行）：`num_samples=100`，`iterations=600`，`num_levels=50`，`optuna_trials=20`，`sa_runs=1000`，`sa_sweeps=3000`；默认 seed 数是 20。
- `quick_l50`（流程复现和趋势验证）：`num_samples=100`，`iterations=100`，`num_levels=50`，`optuna_trials=3`，`sa_runs=100`，`sa_sweeps=500`；默认 seed 数是 3。
- `test`（烟测）：`num_samples=10`，`iterations=2`，`num_levels=8`，`optuna_trials=0`，`sa_runs=2`，`sa_sweeps=6`；默认 seed 数是 1。
- CLI 数值参数会覆盖 preset，例如 `--iterations`、`--num-levels`、`--optuna-trials`、`--sa-runs`、`--sa-sweeps`。
- `--settings` 必填，可选值是 `wo_cgfm` 和 `w_cgfm`。
- `--objectives` 可选；不传时按 `figure4_experiment_config.OBJECTIVES` 的 canonical 顺序运行全部五个 objective。
- `EncodingConfig` 要求 `num_levels > 1`；`ExperimentConfig` 要求 `num_samples` 和 `iterations` 为正数。
- `FMConfig` 要求 `optuna_trials >= 0`，且 `device` 为 `cpu` 或 `cuda`。
- `SAConfig` 要求 `runs` 和 `sweeps` 为正数。
- runner 不强制解释器名称；如果传入 `--device cuda` 但 PyTorch CUDA 不可用，会在训练前失败。

### Figure 4 运行方法

推荐用 `quick_l50` 同时运行两条曲线，作为 Figure 4 流程复现和定性趋势验证：

```powershell
conda run -n env_torch python figure4_runner.py `
  --output-dir figure4_compare_quick_l50 `
  --device cuda `
  --resume `
  --preset quick_l50 `
  --settings wo_cgfm w_cgfm
```

只运行 `w/o CGFM`：

```powershell
conda run -n env_torch python figure4_runner.py `
  --output-dir figure4_wo_cgfm_quick_l50 `
  --device cuda `
  --resume `
  --preset quick_l50 `
  --settings wo_cgfm
```

只运行 `w/ CGFM`：

```powershell
conda run -n env_torch python figure4_runner.py `
  --output-dir figure4_w_cgfm_quick_l50 `
  --device cuda `
  --resume `
  --preset quick_l50 `
  --settings w_cgfm
```

画单个 summary：

```powershell
conda run -n env_torch python plot_figure4.py `
  --summary figure4_compare_quick_l50\figure4_summary.json `
  --output figure4_compare_quick_l50\figure4.png
```

合并分别运行得到的两个 summary 画图：

```powershell
conda run -n env_torch python plot_figure4.py `
  --summary figure4_wo_cgfm_quick_l50\figure4_summary.json figure4_w_cgfm_quick_l50\figure4_summary.json `
  --output figure4_compare_quick_l50\figure4.png
```

### Figure 4 输出规则

`figure4_outputs.py` 定义统一输出布局：

- summary：`figure4_summary.json`
- manifest：`manifest.json`
- 默认图像：`figure4.png`
- runner 日志：`logs/figure4_runner.log`
- plot 日志：`logs/plot_figure4.log`
- checkpoint：`trajectories/{setting}_{objective}_seed_{seed}.json`

`classify_output_path(path)` 返回 `managed_current`、`protected_legacy`、`temporary_test` 或 `unknown`。

受保护旧输出包括 `figure4_quick_l50`、`figure4_cgfm_quick_l50`、`figure4_compare_l50_from_separate`，以及名字包含 `quick_150` 或 `quick150` 的目录。这些目录可用于查看历史结果，但新版代码不保证读取旧 checkpoint 或旧 summary schema。

### Figure 4 命名与 schema 规则

- setting 名称是 `wo_cgfm` 和 `w_cgfm`。
- objective 名称是 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T`。
- summary 聚合 key 是 `{objective}:{setting}`，例如 `kappa:wo_cgfm`。
- summary schema 和 checkpoint schema 都是 version 2。
- 聚合曲线字段是 `best_so_far_mean`、`best_so_far_std`、`best_so_far_min`、`best_so_far_max`。
- replacement 计数字段是 `duplicate_replacements`、`invalid_replacements`、`random_replacements`、`random_replacement_draws`、`accepted_sa_candidates`、`completed_iterations`。
- `delta_alpha` 和 `delta_T` 绘图时使用 log y 轴，并仅在绘图层把显示值 clamp 到 `1e-12`；JSON 保留真实值。

### Figure 4 测试与验收

运行完整单元测试：

```powershell
conda run -n env_torch python -m unittest discover -v
```

运行 Figure 4 小规模验收：

```powershell
conda run -n env_torch python figure4_runner.py `
  --output-dir figure4_refactor_test `
  --device cuda `
  --resume `
  --settings wo_cgfm w_cgfm `
  --objectives kappa `
  --preset test
```

```powershell
conda run -n env_torch python plot_figure4.py `
  --summary figure4_refactor_test\figure4_summary.json `
  --output figure4_refactor_test\figure4.png
```

## 第二部分：Figure 5 复现

### Figure 5 开发情况

- 已实现多目标初始数据生成器：`generate_initial_dataset_multi_objective` 和 `generate_initial_dataset_multi_objective_batch`。
- 已实现 `figure5_scalarization.py`：支持论文式 DDTS 人工目标和 weighted-sum baseline 的三个独立 FM targets。
- 已实现 `figure5_pareto.py`：支持 `is_dominated` 和 `pareto_front`。
- 已实现 Figure 5 pipeline、`test/quick150/paper` 配置、checkpoint/resume、summary schema v1、manifest 和 runner。
- 已实现 `plot_figure5.py`：支持单 seed 3D Pareto 总览、分段采样进展、可选 replacement 叠加和多 seed 显式选择。
- Figure 5 不应复用 Figure 4 的 `best_so_far` 单目标曲线作为核心结果；它比较的是多目标优化中采样到的解在三维目标空间中的覆盖情况。

### Figure 5 当前代码结构

- `alloy_dataset_generator.py`：已有 Figure 5 多目标初始数据生成器，复用 Figure 4 的 row schema、物性计算和 CSV 写出。
- `figure5_scalarization.py`：定义 `w_ddts` 与 `wo_ddts` 的人工训练目标构造，以及 preference weights 采样/校验。
- `figure5_pareto.py`：定义 Pareto 支配关系和非支配 front 筛选。
- `figure5_experiment_config.py`：定义 Figure 5 的 `paper/quick150/test` preset 和 CLI override 规则。
- `figure5_pipeline.py`：运行两条多目标 trajectory，处理 FM/QUBO/SA、decode、replacement、resume 和 Pareto summary。
- `figure5_outputs.py`：定义 Figure 5 输出布局、checkpoint schema v1、原子 JSON 写入和日志。
- `figure5_runner.py`：解析 CLI、检查 CUDA、运行 pipeline 并写 manifest。
- `plot_figure5.py`：校验 summary schema v1，按 seed 重算 Pareto front，并绘制总览和时间窗图。
- `test_alloy_dataset_generator.py`：覆盖多目标初始数据生成。
- `test_figure5_scalarization.py`：覆盖 weighted-sum、DDTS、权重采样和输入校验。
- `test_figure5_pareto.py`：覆盖支配方向、相同点、折中点、非凸点和非法输入。
- `test_figure5_pipeline.py`：覆盖 preset、一/三 FM 分支、QUBO 合并、replacement、resume、summary 和 manifest。
- `test_plot_figure5.py`：覆盖 seed 选择、Pareto front 重算、时间窗、输入校验和 PNG 输出。

### Figure 5 目标

- 复现论文 Figure 5 的多目标 FM+QO 对比：`w/ DDTS` 与 `w/o DDTS`。
- 多目标集合固定为 `kappa`、`E`、`rho`：
  - `kappa`：越大越好。
  - `E`：越大越好。
  - `rho`：越小越好。
- 论文参考规模对齐 Figure 5：
  - 初始数据集：`num_samples=500`。
  - active learning：`iterations=1000`。
  - 编码：按论文 Figure 5 使用四相直接 one-hot 编码，4 个 block，每个 block `num_levels=25`，并加入 system penalty。
  - QUBO 求解：复用 D-Wave Ocean `neal` simulated annealing。
- 由于论文规模运行时间过长，本项目仅保留 `paper` 参数用于对照，不尝试实际运行；项目流程复现和趋势验证输出以 `quick150` 为准。
- 输出重点不是单目标 best-so-far，而是：
  - 每次迭代采样到的合金设计。
  - 每个设计的 `kappa/E/rho` 真实值。
  - `w_ddts` 和 `wo_ddts` 各自找到的非支配 Pareto front。
  - 按迭代区间分段的采样进展。

### Figure 5 配置规模

Figure 5 runner 提供 `paper`、`quick150`、`test` 三个 preset。`quick150` 是本项目实际运行的流程复现和趋势验证规模；`test` 只用于小规模端到端验收，`paper` 只保留论文原始规模参数供对照。

| preset | initial samples | iterations | direct one-hot levels | Optuna trials | SA reads | SA sweeps | default seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test` | 10 | 2 | 8 | 0 | 2 | 6 | 1 |
| `quick150` | 500 | 150 | 25 | 3 | 100 | 500 | 1 |
| `paper` | 500 | 1000 | 25 | 20 | 1000 | 3000 | 1 |

- `quick150` 保留论文规模的 500 条初始数据和 25-level 直接编码，将 active-learning 迭代缩短到 150，并降低 Optuna 和 SA 预算。
- Figure 5 不做 Figure 4 式的默认多 seed 统计；三个 preset 都默认只使用 seed 0。
- 一次默认实验共运行两条 trajectory：`w_ddts + seed 0` 和 `wo_ddts + seed 0`，两者使用同一份初始数据。
- `--num-seeds` 和 `--seed-start` 仅用于用户主动执行额外稳定性实验；`N` 个 seeds 与两个 settings 会产生 `2 * N` 条 trajectory。
- `quick150` 的 summary、Pareto front 和 Figure 5 图是本项目的流程复现和趋势验证产物，不代表论文完整规模结果。
- `paper` 不进入实际运行、结果验收或输出交付范围。
- CLI 数值参数应能覆盖 preset，与 Figure 4 runner 的规则保持一致。

### Figure 5 与 Figure 4 的复用边界

Figure 5 应直接复用：

- `alloy_dataset_generator.py` 中的真实物性计算和 `build_dataset_row`。
- `figure4_qubo_math.py` 中的直接四相编码、system/one-hot penalty、FM QUBO 构造和 SA 求解。
- `figure4_fm_torch.py` 中的 FM 训练和 `fm_to_qubo`。
- `figure4_outputs.py` 的输出布局思想、checkpoint 原子写入、日志配置和 schema 校验风格。
- `figure4_pipeline.py` 中的候选验收思想：decode 后必须得到非负、总和为 1 的四相 composition；非法或重复候选用 random replacement 处理。

Figure 5 应新增而不是硬塞到 Figure 4 中：

- 多目标 scalarization 策略。
- Pareto front 计算。
- 3D objective-space summary 和 plot schema。
- Figure 5 专用 runner、summary 和 checkpoint dataclass。

### Figure 5 初始数据生成

Figure 5 的多目标初始数据生成器已实现：

```python
generate_initial_dataset_multi_objective(
    num_samples: int = 500,
    seed: int | None = None,
    output_path: str | Path | None = None,
)
```

生成流程：

1. 固定 Al matrix 为 80%，四个次相归一化分数总和为 1。
2. 从剩余 simplex 区间中逐步抽取前三个临时分数。
3. 最后一个临时分数取剩余量。
4. 打乱四个分数分配到 Si、Mg2Si、Al3Ni、Al2Cu。
5. 用 `build_dataset_row` 计算所有物性。
6. 可选复用现有 CSV schema 写出数据。

### Figure 5 scalarization

Figure 5 至少包含两个 setting：

- `w_ddts`：使用 DDTS，即 data-driven Tchebycheff scalarization。
- `wo_ddts`：使用标准 weighted-sum scalarization。

`wo_ddts` weighted-sum 每轮 active learning：

1. 当前数据集包含 composition 和真实 `kappa/E/rho`。
2. 生成 preference weights `w = (w_kappa, w_E, w_rho)`，要求非负且总和为 1。
3. 对三个目标统一成最小化方向：`[-kappa, -E, rho]`。
4. 对每个目标在当前数据集上做 z-score。
5. 分别训练三个 FM，并将它们转换为 `Q_kappa/Q_E/Q_rho`。
6. 在 QUBO 层按 preference weights 合并：

```text
Q_weighted = w_kappa * Q_kappa + w_E * Q_E + w_rho * Q_rho
```

`w_ddts` DDTS 每轮 active learning：

1. 当前数据集包含 composition 和真实 `kappa/E/rho`。
2. 生成 preference weights `w = (w_kappa, w_E, w_rho)`。
3. 对 `kappa/E/rho` 分别做 z-score。
4. 在 z-score 空间计算 utopian point：`1.1*max(z_kappa)`、`1.1*max(z_E)`、`1.1*min(z_rho)`。
5. 计算方向一致距离，构造人工训练目标：

```text
y_hat(d) = max(
  w_kappa * (u_kappa - z_kappa(d)),
  w_E * (u_E - z_E(d)),
  w_rho * (z_rho(d) - u_rho)
)
```

`w_ddts` 输出一个“越小越好”的 FM 训练目标；`wo_ddts` 的三个方向一致 target 分别用于三个 FM，weighted sum 发生在 QUBO 层。

### Figure 5 Pareto front

Figure 5 trajectory 分开保存每轮 proposed QUBO solution 和实际 added solution。summary 的扁平 `solutions` 只包含有效 proposed solution；重复 proposed solution 保留，random replacement 不进入论文式 Pareto front。

对 `kappa/E/rho`，解 A 支配解 B 当且仅当：

```text
A.kappa >= B.kappa
A.E >= B.E
A.rho <= B.rho
```

并且至少一个目标严格更好。未被任何其他点支配的点属于 Pareto front。

当前已实现：

```python
is_dominated(candidate, other) -> bool
pareto_front(points) -> list[points]
```

`pareto_front` 保持输入顺序，并返回原始 point 对象，方便后续 summary 保留 `setting/seed/iteration/composition` 等字段。

### Figure 5 目标输出布局

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
  "trajectories": [],
  "solutions": [],
  "pareto_front": {
    "w_ddts": [],
    "wo_ddts": []
  }
}
```

checkpoint 中必须保存：

- 当前 rows。
- 每轮 proposed solution 和 added solution。
- 当前 iteration。
- random replacement 计数。
- 当前 setting、seed、config。
- 最近一轮 weights、utopian point、FM metadata 和 QUBO stats。

### Figure 5 绘图

`plot_figure5.py` 生成接近论文 Figure 5 的三行布局：

1. 顶部 panel a/b：`w_ddts` 与 `wo_ddts` 的全部 proposed QUBO solutions 和 Pareto front。
2. 中部 panel c：`w_ddts` 按迭代区间横向排列的采样进展。
3. 底部 panel d：`wo_ddts` 按迭代区间横向排列的采样进展。

本项目 `quick150` 趋势验证结果的默认分段：

```text
0-50
50-100
100-150
```

`paper` 参数对照的分段为 `0-250`、`250-500`、`500-750`、`750-1000`，但本项目不生成该规模图。其他迭代规模自动划分为最多四个近似等宽时间窗，也可用 `--iteration-boundaries` 覆盖。

坐标轴：

```text
x = kappa
y = rho
z = E
```

绘图规则：

- 灰色点：该 setting 找到的所有候选设计。
- 蓝色点：该 setting 的 Pareto front。
- 分段图用不同颜色表示迭代区间。
- 单 seed summary 自动选择唯一 seed；多 seed summary 必须通过 `--seed` 选择一条 trajectory，不做跨 seed 聚合。
- 绘图时针对选定 seed 和每个 setting 重新计算 Pareto front，不使用多 seed summary 中的联合 front。
- random replacement 默认不显示，也不参与 Pareto front；传入 `--include-replacements` 后以橙色 `x` 叠加。
- 所有 3D panel 使用相同的坐标范围、视角和单位，重复 proposed solution 保留且 Pareto 点不连线。

### Figure 5 运行命令

小规模验收：

```powershell
conda run -n env_torch python figure5_runner.py `
  --output-dir figure5_test `
  --device cuda `
  --resume `
  --preset test `
  --settings w_ddts wo_ddts
```

本项目流程复现和趋势验证运行：

```powershell
conda run -n env_torch python figure5_runner.py `
  --output-dir figure5_quick150 `
  --device cuda `
  --resume `
  --preset quick150 `
  --settings w_ddts wo_ddts
```

可选的多 seed 稳定性实验（非论文 Figure 5 默认流程）：

```powershell
conda run -n env_torch python figure5_runner.py `
  --output-dir figure5_quick150_multiseed `
  --device cuda `
  --resume `
  --preset quick150 `
  --num-seeds 3 `
  --seed-start 0 `
  --settings w_ddts wo_ddts
```

生成项目 Figure 5 趋势验证图：

```powershell
conda run -n env_torch python plot_figure5.py `
  --summary figure5_quick150\figure5_summary.json `
  --output figure5_quick150\figure5.png
```

显示 random replacement 并自定义时间窗：

```powershell
conda run -n env_torch python plot_figure5.py `
  --summary figure5_quick150\figure5_summary.json `
  --output figure5_quick150\figure5_with_replacements.png `
  --include-replacements `
  --iteration-boundaries 0 30 75 110 150
```

绘制多 seed 实验中的指定 seed：

```powershell
conda run -n env_torch python plot_figure5.py `
  --summary figure5_quick150_multiseed\figure5_summary.json `
  --output figure5_quick150_multiseed\figure5_seed_1.png `
  --seed 1
```

`paper` preset 仅作为论文参数参考保留，本项目不安排运行或绘图。

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
   - 当前通过 mock FM/SA 测试 summary、manifest、日志和 checkpoint schema。
   - synthetic summary 能生成非空、非纯色 PNG，并写入 `logs/plot_figure5.log`。
3. `quick150` 流程复现和趋势验证输出验收：
   - `figure5_summary.json` 中包含 `w_ddts` 与 `wo_ddts` 两个 setting。
   - 每个 setting 完成 1 条 trajectory，每条 trajectory 包含 `150` 条 iteration solution record。
   - `pareto_front.w_ddts` 和 `pareto_front.wo_ddts` 非空。
   - `figure5.png` 包含全局采样图和分段采样图。

### Figure 5 实现顺序

1. 已完成：在 `alloy_dataset_generator.py` 增加 Figure 5 多目标初始数据生成器。
2. 已完成：新增 `figure5_scalarization.py`，实现 weighted-sum 和 DDTS。
3. 已完成：新增 `figure5_pareto.py` 和 `test_figure5_pareto.py`。
4. 已完成：新增 `figure5_pipeline.py`，复用 Figure 4 的 FM/QUBO/SA/decode/replacement。
5. 已完成：新增 `figure5_experiment_config.py`、`figure5_runner.py` 和 `figure5_outputs.py`，实现 preset、checkpoint、manifest、summary 和日志。
6. 已完成：新增 `plot_figure5.py` 和 `test_plot_figure5.py`。
7. 待执行：先用 `preset test` 验收，再用 `preset quick150` 产出本项目流程复现和趋势验证结果；不运行 `preset paper`。
