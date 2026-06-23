# 合金 Figure 4 复现工作流

本文档记录当前代码中已经实现的工作流和规则。当前实现覆盖 Figure 4 的 `w/o CGFM` 与 `w/ CGFM` 两条曲线；Figure 5 / DDTS 尚未实现，只在文末预留位置。

## 项目状态

- Figure 4 已实现两条 setting：`wo_cgfm` 和 `w_cgfm`。
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

## Figure 5 工作流（预留）

当前代码尚未实现 Figure 5 / DDTS 工作流。后续实现时，本节应补充：

- Figure 5 对应的算法入口和 runner。
- Figure 5 配置 preset、CLI 参数和默认规模。
- Figure 5 checkpoint、日志、summary、manifest 和 plot 输出布局。
- Figure 5 绘图命令和验收命令。
- Figure 5 与 Figure 4 共用或独立的代码边界。

在 Figure 5 代码落地前，不在本文档中提供 Figure 5 的运行命令。
