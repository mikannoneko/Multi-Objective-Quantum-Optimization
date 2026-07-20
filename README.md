# Figure 4 / Figure 5 小规模流程复现

本项目复现论文 *Progress on Data-Driven, Multi-Objective Quantum Optimization*（arXiv:2512.11479）中 Figure 4 和 Figure 5 的算法流程。论文元数据、本地 PDF 文件名及 SHA-256 见 `references/README.md`。

## 项目边界

本项目只验收小规模流程复现，不验收论文计算规模，也不以复现论文数值或定性优势作为通过条件。

- `quick`：Figure 4 和 Figure 5 统一使用的小规模正式验收名称。
- `test`：单元测试和端到端烟测，只确认最短代码路径可运行。
- `paper`：论文参数的参考配置；可以显式运行，但不属于本项目验收边界，验收器不会接受 paper-scale summary。
- 新输出目录统一命名为 `figure4_quick`、`figure5_quick`；`quick_l50`、`quick150`、`quick_150` 等旧名称仅用于识别历史输出，不再用于新调用。
- `validate_reproduction.py` 只验收 canonical `quick` 规模、轨迹完整性、输出结构和候选计数一致性。
- Figure 4 的论文趋势，以及 Figure 5 的 DDTS 覆盖率与均匀性，仍会写入验收报告的 `diagnostics`，但不影响 `passed`。

论文补充材料使用 `fastFM + ALS`。本项目使用 PyTorch 二阶 FM + LBFGS，并保留 rank 6、最多 2000 次训练步、target z-score、Optuna 和 FM-to-QUBO 流程。这是工程替代，因此结果只能称为“小规模流程复现”。

当前仓库尚未提交新版本 canonical `quick` 验收通过报告；实时状态见 `results/reproduction_status.json`。旧输出审查报告只说明修复前的问题，不作为当前验收证据。

## 环境与测试

```powershell
conda env create -f environment.yml
conda activate env_torch
```

已有环境可更新：

```powershell
conda env update -n env_torch -f environment.yml --prune
```

运行全部测试：

```powershell
conda run -n env_torch python -m unittest discover -v
```

`manifest.json` 会记录命令、解析后的配置、Python 与依赖版本、CUDA 信息、Git commit/dirty 状态，以及本地论文 PDF 的路径、存在状态和 SHA-256。原始实验目录、checkpoint 和 PNG 默认不进入 Git；验收器生成的精简 JSON 报告应保存到 `results/`。

## 共同的候选选择规则

Figure 4 和 Figure 5 共用 `figure4_qubo_math.py` 中的退火求解逻辑：

1. 保留 `neal` 返回的全部 SA reads，并按能量升序排列。
2. 从低能量到高能量逐个解码，选择第一个同时满足编码约束和四相总和约束的候选。
3. 如果整批 reads 都没有可行候选，本轮明确失败，不用随机样本伪装成 QUBO 结果。
4. 只有可行候选与已有 composition 重复时，才生成唯一的 random replacement。

summary 会记录 `infeasible_sa_samples_skipped`、`max_feasible_candidate_rank`、`duplicate_replacements`、`random_replacements` 和 `random_replacement_draws`。旧版 `invalid_replacement` 路径已移除。

命名统一使用 D-Wave 的 `reads` 术语：配置字段为 `SAConfig.reads`，CLI 首选 `--sa-reads`；`--sa-runs` 只保留为兼容别名。`experiment_runtime.py` 统一负责 device 检查、seed 列表校验和 manifest 运行环境元数据。

按照论文补充材料 S1.1，FM-to-QUBO 会丢弃不影响最优 bit 状态的整体偏置 `w0`；各 QUBO 项只按非恒定矩阵系数计算归一化尺度，常数项不得改变 FM 目标与约束惩罚的相对强度。

按照论文 Eq. 18，one-hot 的数值层级固定为 `alpha_i = i / N_bits`，所有 block 使用相同的升序 bit 映射，零值仍由全零 bit-string 表示。补充材料 S7/S8 后所述的逐轮随机化只适用于 CGFM 正反映射中的相变量分配 `phase_permutation`，不打乱 `alpha_i`；Figure 4 的直接编码和 Figure 5 的四相编码因此不随 iteration seed 改变数值层级。

## 第一部分：Figure 4 复现

### Figure 4 当前状态

- 已完成 `wo_cgfm`（直接四相编码）和 `w_cgfm`（CGFM 角度编码）两条流程。
- 已完成 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T` 五个 objective；优化方向由 `ObjectiveSpec` 统一定义。
- 已完成 PyTorch FM、FM-to-QUBO、SA 全 reads 可行解筛选、重复候选替换、best-so-far 更新及多 seed 聚合。
- 已完成 per-trajectory checkpoint、`--resume`、manifest、runner/plot 日志、schema v3 summary 和多 summary 合并绘图；旧 schema 不自动迁移。
- `figure4_runner.py` 默认使用 `quick`，`test` 用于烟测，`paper` 只作参数参考。
- 单元测试和烟测流程已通过；当前仍待执行并保存新版 Figure 4 canonical `quick` 运行及验收报告，状态以 `results/reproduction_status.json` 为准。

### Figure 4 代码结构

Figure 4 按“数据与运行环境 → 配置与策略 → 数学与模型 → pipeline → I/O 与入口”分层：

```text
alloy_dataset_generator.py       组成采样、四相归一化、五个真实性质计算
experiment_runtime.py            device/seed 校验、运行环境与论文哈希元数据
figure4_experiment_config.py     objective、嵌套配置、paper/quick/test preset
figure4_setting_strategies.py    wo_cgfm/w_cgfm 编码与解码策略分发
figure4_qubo_math.py             离散编码、CGFM、QUBO 惩罚、SA 与可行解筛选
figure4_fm_torch.py              PyTorch FM、Optuna、LBFGS、FM-to-QUBO
figure4_pipeline.py              trajectory、active learning、checkpoint 恢复与聚合
figure4_outputs.py               schema v3 checkpoint/summary、标准路径、原子 JSON、日志
figure4_runner.py                CLI、配置解析、环境检查、manifest、pipeline 调用
plot_figure4.py                  读取一个或多个 schema v3 summary 并绘图
validate_reproduction.py         canonical quick 结构验收与非阻断 diagnostics
test_figure4_pipeline.py         配置、数学、流程、输出、绘图和 README 契约测试
```

主依赖方向为：

```text
figure4_runner
  -> figure4_experiment_config / experiment_runtime / figure4_outputs
  -> figure4_pipeline
       -> alloy_dataset_generator
       -> figure4_setting_strategies
       -> figure4_fm_torch
       -> figure4_qubo_math
       -> figure4_outputs

plot_figure4 / validate_reproduction -> figure4_summary.json
```

runner 只做参数解析和编排；active-learning 逻辑只放在 pipeline；编码差异只放在 strategy；路径和 JSON 写入只由 outputs 模块管理。

### Figure 4 算法流程

1. runner 解析 preset 和显式覆盖参数，检查训练依赖与 device，生成连续且非负的 seed 列表，并写 `manifest.json`。
2. `generate_initial_dataset_batch` 为每个 seed 生成一份初始合金数据；同一 seed 下所有 setting/objective 从相同初始数据出发。
3. pipeline 对 `seed × objective × setting` 的笛卡尔积运行独立 trajectory；一条 trajectory 的唯一身份为 `(setting, objective, seed)`。
4. 每轮读取当前数据集的真实 objective。最大化目标先取负，最小化目标保持原值，再做 z-score，使 FM/QUBO 始终按“越小越好”求解。
5. strategy 创建当轮离散编码并编码训练特征：
   - `wo_cgfm` 使用四个 composition block；各 block 按 Eq. 18 固定使用升序 `alpha_i`，QUBO 需要 `system penalty + one-hot penalty`。
   - `w_cgfm` 每轮只随机打乱四个相进入 S7/S8 映射的顺序，再使用三个固定 `alpha_i` 的 CGFM angle block；解码天然得到非负且总和为 1 的四相 composition，因此只加 `one-hot penalty`。
6. `fit_torch_fm` 训练二阶 FM；`fm_to_qubo` 按补充材料 S1.1 丢弃整体偏置 `w0`，将线性项和交互项展开为 QUBO；`build_single_objective_qubo` 只按非恒定 QUBO 系数归一化，再叠加所需约束。
7. `solve_qubo_with_sa` 返回全部能量排序后的 reads，`select_lowest_energy_feasible_sample` 选择最低能量可行状态并记录其 rank 和跳过数量。
8. strategy 解码候选。新 composition 直接加入数据集；重复 composition 保留“重复”判定并加入唯一 random replacement；不可行候选不会进入替换分支。
9. 用真实性质更新该 objective 的 best-so-far，更新审计计数，并在每轮后原子写入 trajectory checkpoint。
10. 全部 trajectory 完成后，按 `{objective}:{setting}` 聚合各 seed 的 `mean/std/min/max` 曲线并写 schema v3 summary；绘图和验收只读取 summary，不重新训练。

### Figure 4 对象命名与调用规则

| 层 | 对象或值 | 命名与职责 |
| --- | --- | --- |
| 规模 | `RunScale` | 只允许 `paper`、`quick`、`test`；正式验收只使用 `quick`。 |
| 目标 | `ObjectiveSpec`、`OBJECTIVES` | objective 名固定为 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T`，顺序以 `OBJECTIVES` 为准。 |
| 配置 | `ExperimentConfig` | 顶层持有 `EncodingConfig`、`FMConfig`、`SAConfig`，不在 pipeline 中散落独立参数。 |
| setting | `Figure4Setting` | 只允许 `wo_cgfm`、`w_cgfm`；不得使用大小写变体或显示名称代替机器值。 |
| 策略 | `SettingStrategy` | 统一接口为 `create_encoding`、`encode_rows`、`decode_candidate`；实现类为 `WOCGFMStrategy`、`WCGFMStrategy`。 |
| 编码 | `IterationEncoding` | `positive_count_by_bit` 固定实现 Eq. 18；仅 `w_cgfm` 设置逐轮变化的 `phase_permutation`。 |
| 运行状态 | `TrajectoryState` | 只表示可 checkpoint 的单轨迹可变状态。 |
| 结果 | `TrajectoryResult`、`AggregatedTrajectory`、`Figure4Summary` | 分别表示单轨迹、跨 seed 曲线和完整 summary；summary 聚合键固定为 `{objective}:{setting}`。 |
| 输出 | `Figure4OutputLayout` | 统一派生 summary、manifest、日志、图片和 checkpoint 路径。 |

调用规则：

1. 完整实验优先调用 `figure4_runner.py`，因为它会执行依赖/device 检查并写 manifest。
2. 程序化调用依次使用 `resolve_experiment_config(...)` 和 `run_figure4_experiment(...)`；直接调用 pipeline 不会自动写 manifest。
3. `run_single_trajectory(...)` 只用于聚焦测试或高级编排；普通调用者不应自行拼接 checkpoint payload。
4. 以 `_` 开头的函数和未列入模块 `__all__` 的迭代中间对象属于模块内部实现，不作为跨模块调用接口。
5. setting 和 seed 列表必须非空、无重复；seed 必须是非负整数。objective 子集会按 canonical `OBJECTIVES` 顺序执行，而不是按 CLI 输入顺序执行。
6. 路径必须通过 `figure4_output_layout(...)` / `Figure4OutputLayout` 获取；同一输出目录不得混用不同 config、setting、objective 或 seed 身份后继续 `--resume`。

程序化调用示例：

```python
from figure4_experiment_config import OBJECTIVES, resolve_experiment_config
from figure4_pipeline import run_figure4_experiment

config = resolve_experiment_config(preset="quick", device="cpu")
summary = run_figure4_experiment(
    seed_list=[0, 1, 2],
    config=config,
    output_dir="figure4_quick",
    objectives=OBJECTIVES,
    settings=("wo_cgfm", "w_cgfm"),
    resume=True,
)
```

### Figure 4 配置规则

`ExperimentConfig` 的结构为：

```text
ExperimentConfig
  num_samples
  iterations
  encoding: EncodingConfig(num_levels)
  fm: FMConfig(optuna_trials, device)
  sa: SAConfig(reads, sweeps)
```

| preset | initial samples | iterations | levels | Optuna trials | SA reads | SA sweeps | default seeds | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test` | 10 | 2 | 8 | 0 | 32 | 12 | 1 | 烟测 |
| `quick` | 100 | 100 | 50 | 3 | 100 | 500 | 3 | 项目验收 |
| `paper` | 100 | 600 | 50 | 20 | 1000 | 3000 | 20 | 参数参考，不验收 |

- 配置先由 `preset_config` 解析，再由 `resolve_experiment_config` 应用 CLI 显式覆盖；未覆盖字段保留 preset 值。
- `num_samples`、`iterations`、SA reads/sweeps 必须为正数；`num_levels > 1`；`optuna_trials >= 0`；device 只允许 `cpu` 或 `cuda`。
- CLI 可覆盖 `--num-samples`、`--iterations`、`--num-levels`、`--optuna-trials`、`--sa-reads`、`--sa-sweeps`。
- `--settings` 必填；`--objectives` 省略或传空时运行全部五个目标。
- `--seed-start` 必须非负，`--num-seeds` 必须为正；默认 seed 为从 0 开始的连续列表。
- 请求 `--device cuda` 但 CUDA 不可用时，会在实验开始前失败，不静默回退到 CPU。
- canonical `quick` 验收要求表中精确配置、3 个默认 seed、全部 setting 和 objective；任何数值或 seed 覆盖都可运行，但不属于 canonical 验收结果。

### Figure 4 运行方法

最短烟测：

```powershell
conda run -n env_torch python figure4_runner.py `
  --output-dir figure4_test `
  --preset test `
  --settings wo_cgfm w_cgfm `
  --objectives kappa `
  --resume

conda run -n env_torch python plot_figure4.py `
  --summary figure4_test\figure4_summary.json `
  --output figure4_test\figure4.png
```

正式小规模流程：

```powershell
conda run -n env_torch python figure4_runner.py `
  --output-dir figure4_quick `
  --preset quick `
  --settings wo_cgfm w_cgfm `
  --device cuda `
  --resume

conda run -n env_torch python plot_figure4.py `
  --summary figure4_quick\figure4_summary.json `
  --output figure4_quick\figure4.png
```

验收 canonical `quick` summary：

```powershell
conda run -n env_torch python validate_reproduction.py figure4 `
  --summary figure4_quick\figure4_summary.json `
  --output results\figure4_quick_validation.json
```

### Figure 4 输出规则

```text
<output-dir>/
  figure4_summary.json
  manifest.json
  figure4.png
  logs/
    figure4_runner.log
    plot_figure4.log
  trajectories/
    {setting}_{objective}_seed_{seed}.json
```

| 输出 | 规则 |
| --- | --- |
| `manifest.json` | 记录命令、preset、resolved config、runtime、seed/objective/setting 选择和标准输出路径。 |
| trajectory checkpoint | schema v3；绑定 `setting + objective + seed + config`，只在全部身份字段精确匹配时恢复。 |
| `figure4_summary.json` | schema v3；包含 config、seed/objective/setting、所有 `trajectories` 和 `{objective}:{setting}` 聚合曲线。 |
| `figure4.png` | 由 `plot_figure4.py` 从一个或多个 schema v3 summary 生成；不是训练输入。 |
| 日志 | runner 和 plot 分文件记录；不得用日志替代机器可读 summary/validation report。 |

每条轨迹必须满足：`completed_iterations == iterations`、`len(best_so_far) == iterations`、`final_dataset_size == num_samples + iterations`，并且 `accepted_sa_candidates + duplicate_replacements == iterations`。不同配置必须使用不同输出目录；旧 schema checkpoint 不会自动覆盖或迁移。

### Figure 4 验收规则

Figure 4 canonical `quick` 只检查：

- 配置为 100 条初始数据、100 次迭代、50 levels、3 seeds。
- 五个 objectives × 两个 settings × 三个 seeds，共 30 条轨迹全部完成。
- 聚合曲线字段完整、长度正确且数值有限。
- accepted/replacement 计数、数据集增长和迭代数一致。

`w_cgfm` 是否表现出论文报告的优势只写入 `diagnostics.cgfm_paper_direction`，不影响验收结果。

## 第二部分：Figure 5 复现

### Figure 5 当前状态

- 已完成 `w_ddts` 和 `wo_ddts`（weighted-sum baseline）两条多目标流程。
- 已完成最大化 `kappa`、最大化 `E`、最小化 `rho` 的统一方向处理、可复现 preference weights、DDTS 人工目标、三 FM QUBO 合并和 Pareto front。
- 已完成每轮 proposed solution 与实际 added solution 的分别记录；重复 proposed solution 会保留，random replacement 不进入 Pareto front。
- 已完成 SA 全 reads 可行解筛选、候选 rank 诊断、schema v2 checkpoint/summary、`--resume`、manifest 和 runner/plot 日志；旧 schema 不自动迁移。
- 已完成单/双 setting 绘图、3D 总览、迭代窗口及多 seed 显式选择；canonical `quick` 要求两个 setting 完整对比。
- 单元测试和烟测流程已通过；当前仍待执行并保存新版 Figure 5 canonical `quick` 运行及验收报告，状态以 `results/reproduction_status.json` 为准。

### Figure 5 代码结构

Figure 5 复用数据、FM、QUBO 和运行环境基础层，并在其上增加多目标 scalarization、Pareto 和独立 pipeline：

```text
alloy_dataset_generator.py       多目标初始数据、组成归一化、真实性质计算
experiment_runtime.py            device/seed 校验、运行环境与论文哈希元数据
figure4_experiment_config.py     复用 EncodingConfig/FMConfig/SAConfig
figure4_fm_torch.py              复用 PyTorch FM、Optuna、LBFGS、FM-to-QUBO
figure4_qubo_math.py             复用直接编码、QUBO 惩罚、SA 与可行解筛选
figure5_experiment_config.py     Figure 5 配置与 paper/quick/test preset
figure5_scalarization.py         preference weights、DDTS、三目标标准化与数学对照
figure5_pareto.py                mixed-sense 支配关系与 Pareto front
figure5_pipeline.py              多目标 active learning、checkpoint、solutions 与 summary
figure5_outputs.py               schema v2 checkpoint/summary、标准路径、原子 JSON、日志
figure5_runner.py                CLI、配置解析、环境检查、manifest、pipeline 调用
plot_figure5.py                  单/双 setting、seed 选择、3D 总览和迭代窗口
validate_reproduction.py         canonical quick 结构验收与精确 front diagnostics
test_figure5_*.py                scalarization、Pareto、pipeline 和输出契约测试
test_plot_figure5.py             summary 校验、seed 选择和绘图测试
```

主依赖方向为：

```text
figure5_runner
  -> figure5_experiment_config / experiment_runtime / figure5_outputs
  -> figure5_pipeline
       -> alloy_dataset_generator
       -> figure5_scalarization / figure5_pareto
       -> figure4_fm_torch / figure4_qubo_math
       -> figure5_outputs

plot_figure5 / validate_reproduction -> figure5_summary.json
```

Figure 5 不调用 Figure 4 pipeline，也不使用 CGFM strategy；它只复用稳定的配置子对象、FM 和 QUBO/SA 数学组件。

### Figure 5 算法流程

1. runner 解析 preset 和覆盖参数，检查依赖/device，生成 seed 列表并写 `manifest.json`。
2. `generate_initial_dataset_multi_objective_batch` 为每个 seed 生成共享初始数据；同一 seed 下 `w_ddts` 和 `wo_ddts` 从相同数据出发。
3. pipeline 对 `seed × setting` 运行独立 trajectory；一条 trajectory 的唯一身份为 `(setting, seed)`。
4. `preference_weights_for_iteration(seed, iteration)` 为每轮生成三个非负且和为 1 的确定性权重；同一 seed/iteration 的两个 setting 使用相同权重。
5. 两条 scalarization 路径分别构造 FM/QUBO：
   - `w_ddts`：对三个真实目标做 z-score，构造 utopian point，并以最大加权方向距离作为一个 FM 的人工 target：

     ```text
     max(
       w_kappa * (u_kappa - z_kappa),
       w_E     * (u_E     - z_E),
       w_rho   * (z_rho   - u_rho)
     )
     ```

   - `wo_ddts`：将目标转换为 `-kappa`、`-E`、`rho` 后分别 z-score，训练三个 FM，再在 QUBO 层合并：

     ```text
     Q = w_kappa * Q_kappa + w_E * Q_E + w_rho * Q_rho
     ```

6. 两条路径都使用按 Eq. 18 固定升序 `alpha_i` 的四个 composition one-hot block，并对合并后的 QUBO 加 `system penalty + one-hot penalty`；Figure 5 不使用 CGFM，因此没有逐轮 phase permutation。
7. SA 返回全部 reads；流程选择最低能量可行候选，记录 `sa_energy`、`feasible_candidate_rank` 和跳过的不可行样本数。
8. 每轮同时保存 `proposed_solution` 与 `added_solution`。新候选两者相同；重复候选的 proposed 保持不变，added 改为唯一 random replacement，状态为 `duplicate_replacement`。
9. 每轮更新 trajectory state 并原子写 checkpoint；完成后将所有 proposed solution 展平到 summary 的 `solutions`。
10. `pareto_front` 按 setting 从 proposed solutions 计算非支配集：A 支配 B 当且仅当 `A.kappa >= B.kappa`、`A.E >= B.E`、`A.rho <= B.rho`，且至少一个目标严格更优。
11. 绘图器对选定 seed 重新校验并计算展示 front；验收器枚举 25-level 离散设计空间，计算精确 front coverage、precision 和 spacing diagnostics。

### Figure 5 对象命名与调用规则

| 层 | 对象或值 | 命名与职责 |
| --- | --- | --- |
| 规模 | `Figure5RunScale` | 只允许 `paper`、`quick`、`test`；正式验收只使用 `quick`。 |
| 配置 | `Figure5ExperimentConfig` | 顶层持有复用的 `EncodingConfig`、`FMConfig`、`SAConfig`。 |
| setting | `Figure5Setting` | 只允许 `w_ddts`、`wo_ddts`；顺序以 `SUPPORTED_SETTINGS` 为准。 |
| 目标 | `FIGURE5_OBJECTIVES` | 固定顺序为 `kappa`、`E`、`rho`；方向由 `FIGURE5_OBJECTIVE_SENSES` 定义。 |
| scalarization | `ScalarizationResult`、`ObjectiveTargetsResult` | 分别表示单人工 target 和三目标独立 target；权重顺序必须与 `FIGURE5_OBJECTIVES` 一致。 |
| 迭代记录 | `SolutionPoint`、`IterationRecord` | 分别表示一个设计点和一轮 proposed/added 决策；iteration 使用从 0 开始的整数。 |
| 运行状态 | `Figure5TrajectoryState`、`Figure5TrajectoryResult` | 分别表示可恢复状态和完整单轨迹结果。 |
| 总结果 | `Figure5Summary` | 保存 trajectories、展平后的 proposed `solutions` 和按 setting 分组的 `pareto_front`。 |
| 输出 | `Figure5OutputLayout` | 统一派生 summary、manifest、日志、图片和 checkpoint 路径。 |

调用规则：

1. 完整实验优先调用 `figure5_runner.py`；程序化调用依次使用 `resolve_experiment_config(...)` 和 `run_figure5_experiment(...)`。
2. pipeline 的 `w_ddts` 路径调用 `compute_ddts_targets(...)`；`wo_ddts` 路径必须调用 `compute_individual_objective_targets(...)` 后训练三个 FM，并在 QUBO 层合并。
3. `compute_weighted_sum_targets(...)` 和 `scalarize_training_targets(...)` 保留为数学对照/测试工具，不代表生产 pipeline 的三 FM baseline 调用路径。
4. `run_single_trajectory(...)` 只用于聚焦测试或高级编排；以 `_` 开头的函数和未公开的训练中间对象不作为跨模块 API。
5. `decision_status` 只允许 `accepted` 或 `duplicate_replacement`；Pareto front 只读取 `proposed_solution`，不得把 random replacement 当成优化器提出的点。
6. setting 和 seed 列表必须非空、无重复；seed 必须为非负整数。同一 seed/iteration 的权重由函数确定，不允许调用方为两个 setting 分别随机采样。
7. 路径必须通过 `figure5_output_layout(...)` / `Figure5OutputLayout` 获取；多 seed summary 绘图必须用 `--seed` 明确选择一条 seed，避免隐式混合轨迹。

程序化调用示例：

```python
from figure5_experiment_config import resolve_experiment_config
from figure5_pipeline import run_figure5_experiment

config = resolve_experiment_config(preset="quick", device="cpu")
summary = run_figure5_experiment(
    seed_list=[0],
    config=config,
    output_dir="figure5_quick",
    settings=("w_ddts", "wo_ddts"),
    resume=True,
)
```

### Figure 5 配置规则

`Figure5ExperimentConfig` 的结构为：

```text
Figure5ExperimentConfig
  num_samples
  iterations
  encoding: EncodingConfig(num_levels)
  fm: FMConfig(optuna_trials, device)
  sa: SAConfig(reads, sweeps)
```

| preset | initial samples | iterations | levels | Optuna trials | SA reads | SA sweeps | default seeds | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test` | 10 | 2 | 8 | 0 | 32 | 12 | 1 | 烟测 |
| `quick` | 500 | 150 | 25 | 3 | 100 | 500 | 1 | 项目验收 |
| `paper` | 500 | 1000 | 25 | 20 | 1000 | 3000 | 1 | 参数参考，不验收 |

- 配置先由 Figure 5 的 `preset_config` 解析，再由 `resolve_experiment_config` 应用显式覆盖；Figure 4 和 Figure 5 的 preset 名相同，但数值彼此独立。
- 数值、device、seed 和 SA 参数遵循与 Figure 4 相同的合法性规则；CLI 首选 `--sa-reads`。
- `--settings` 必填；objectives 固定为 `kappa E rho`，Figure 5 CLI 不接受 objective 子集。
- Figure 5 使用四相直接 one-hot 编码和 system penalty，不接受 CGFM setting。
- 默认运行 seed 0；显式传入 `--num-seeds N` 时每个 setting 都运行 N 条 trajectory。
- 多 seed 和数值覆盖是受支持的开发配置，但 canonical `quick` 验收要求表中精确配置、seed 0 和两个 setting。
- 请求 `--device cuda` 但 CUDA 不可用时，会在实验开始前失败，不静默回退到 CPU。

### Figure 5 运行方法

最短烟测：

```powershell
conda run -n env_torch python figure5_runner.py `
  --output-dir figure5_test `
  --preset test `
  --settings w_ddts wo_ddts `
  --resume

conda run -n env_torch python plot_figure5.py `
  --summary figure5_test\figure5_summary.json `
  --output figure5_test\figure5.png
```

正式小规模流程：

```powershell
conda run -n env_torch python figure5_runner.py `
  --output-dir figure5_quick `
  --preset quick `
  --settings w_ddts wo_ddts `
  --device cuda `
  --resume

conda run -n env_torch python plot_figure5.py `
  --summary figure5_quick\figure5_summary.json `
  --output figure5_quick\figure5.png
```

单 setting 开发路径仍受支持：

```powershell
conda run -n env_torch python figure5_runner.py `
  --output-dir figure5_w_ddts_test `
  --preset test `
  --settings w_ddts

conda run -n env_torch python plot_figure5.py `
  --summary figure5_w_ddts_test\figure5_summary.json `
  --output figure5_w_ddts_test\figure5.png
```

验收 canonical `quick` summary：

```powershell
conda run -n env_torch python validate_reproduction.py figure5 `
  --summary figure5_quick\figure5_summary.json `
  --output results\figure5_quick_validation.json
```

### Figure 5 输出规则

```text
<output-dir>/
  figure5_summary.json
  manifest.json
  figure5.png
  logs/
    figure5_runner.log
    plot_figure5.log
  trajectories/
    {setting}_seed_{seed}.json
```

| 输出 | 规则 |
| --- | --- |
| `manifest.json` | 记录命令、preset、resolved config、runtime、seed/setting 选择、固定 objectives 和标准输出路径。 |
| trajectory checkpoint | schema v2；绑定 `setting + seed + config`，只在全部身份字段精确匹配时恢复。 |
| `figure5_summary.json` | schema v2；包含 config、trajectories、proposed `solutions` 和每个 setting 的 `pareto_front`。 |
| `figure5.png` | 由 `plot_figure5.py` 从 schema v2 summary 生成；多 seed 时必须显式传 `--seed`。 |
| 日志 | runner 和 plot 分文件记录；验收结论只来自 validation report。 |

每条轨迹必须满足：`completed_iterations == iterations`、`len(iteration_records) == iterations`、`final_dataset_size == num_samples + iterations`，并且 `accepted_sa_candidates + duplicate_replacements == iterations`。每条 `IterationRecord` 必须同时保留 setting、seed、iteration、weights、scalarization method、decision status、proposed/added solution 和 SA 诊断字段。

### Figure 5 验收规则

Figure 5 canonical `quick` 只检查：

- 配置为 500 条初始数据、150 次迭代、25 levels、seed 0。
- `w_ddts` 和 `wo_ddts` 两条 trajectory 均完整，共有 300 条 proposed solution records。
- composition、真实目标值、iteration identity 和候选计数一致。
- 两个 summary Pareto fronts 非空，并且精确离散 Pareto front 指标可成功计算。

验收器仍会枚举 25-level 离散设计空间，报告精确 front coverage、precision、spacing CV 及两种 setting 的比值。这些数据只位于 `metrics` 和 `diagnostics`，不设置覆盖率或均匀性通过阈值。
