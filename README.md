# Figure 4 / Figure 5 小规模流程复现

本项目复现论文 *Progress on Data-Driven, Multi-Objective Quantum Optimization*（arXiv:2512.11479）中 Figure 4 和 Figure 5 的算法流程。论文元数据、本地 PDF 文件名及 SHA-256 见 `references/README.md`。

## 项目边界

本项目只验收小规模流程复现，不验收论文计算规模，也不以复现论文数值或定性优势作为通过条件。

- `quick`：Figure 4 和 Figure 5 统一使用的小规模正式验收名称。
- `test`：单元测试和端到端烟测，只确认最短代码路径可运行。
- `paper`：论文参数的参考配置；可以显式运行，但不属于本项目验收边界，验收器不会接受 paper-scale summary。
- 新运行建议把 canonical 小规模输出目录命名为 `figure4_quick`、`figure5_quick`；已完成的 canonical 结果分别保留在 `figure4_quick_gamma2`、`figure5_quick_gamma2`，canonical 身份由 summary 中的完整配置而不是目录名判定。
- `validate_reproduction.py` 按“严格契约解析 → Figure 自洽重建 → 非阻断 diagnostics → JSON-safe report”四层验收 canonical `quick`；它不重新训练 FM、不重建 QUBO，也不运行 SA。
- Figure 4 的论文趋势，以及 Figure 5 的 DDTS 覆盖率与均匀性，仍会写入验收报告的 `diagnostics`，但不影响 `passed`。

Figure 4 的 `paper` preset 保留论文的 50-level 编码：`wo_cgfm` 为 200-bit 密集 QUBO，`w_cgfm` 为 150 bits。论文使用 fastFM+ALS，并以 1000 runs、3000 sweeps 求解；本项目的 PyTorch FM+LBFGS 会形成不同的能量地形，而 soft penalty 只改变能量、并不保证任意有限 SA batch 一定含有可行终态。实际诊断中，某个 50-level `rho/wo_cgfm` 迭代的 1000 个终态有 335 个满足系统总和、仅 2 个满足 one-hot、没有一个同时满足；同轮可行域本身并非空集。仅凭“整批无可行终态”不能判断是采样不足还是 penalty 配比失衡，增加 reads/sweeps 也只会提高搜索量，并不提供可行性保证。

因此 Figure 4 canonical `quick` 有意使用 10 levels：`wo_cgfm` 为 40 bits、`w_cgfm` 为 30 bits；直接编码仍有 `C(13,3)=286` 个可行 composition，足以容纳 100 个初始样本和 100 轮新增样本。这是“小规模流程复现”的边界，不是论文 50-level SA 表现的替代证据，而且减少到 10 levels 本身也不能保证有限 SA 必然返回可行终态：项目曾在 30-bit `w_cgfm` 上观测到 100 个终态全部违反 one-hot，精确能量比较将该批次分类为 `QUBO penalty-balance failure`。为适配本项目的 PyTorch FM 能量地形，Figure 4/5 的 `quick`、`test` preset 将 one-hot penalty `gamma` 固定为 `2.0`；`paper` 保留论文值 `1.0`。这是 non-paper 小规模流程的 penalty-balance 修正，不是有限 SA 可行性的形式化保证。

`paper` 或显式 `--num-levels 50` 仍受支持，但整批 reads 无可行候选时会明确失败；程序不会把不可行状态 cure、精确枚举结果或随机替换冒充 QUBO 解。此时才会运行错误诊断：若最低采样不可行能量严格低于精确可行域最优能量，报告 `QUBO penalty-balance failure`；反之若精确可行最优严格更低，报告 `heuristic-sampling failure`；两者在 `1e-12` 容差内相等或可行状态数超过诊断上限时，只报告 `no feasible SA candidate; cause not classified`。该诊断不修改实验结果。由于 seed schedule 是确定的，保持 config 不变执行 `--resume` 会重现同一失败批次；应根据分类检查 penalty 配比或调整 reads/sweeps/求解器，并为任何配置变化使用新的 `--output-dir`。不同配置不能混用同一 checkpoint 目录。

论文补充材料使用 `fastFM + ALS`。本项目使用 PyTorch 二阶 FM + LBFGS，并保留 rank 6、最多 2000 次训练步、target z-score、Optuna 和 FM-to-QUBO 流程。这是工程替代，因此结果只能称为“小规模流程复现”。

Figure 4 和 Figure 5 的 canonical `quick` 输出均已完成并通过严格验收，报告分别位于 `figure4_quick_gamma2/figure4_validation.json`、`figure5_quick_gamma2/figure5_validation.json`。旧输出审查报告只说明修复前的问题，不作为当前验收证据。

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

`figure4_manifest.json`、`figure5_manifest.json` 会分别记录命令、解析后的配置、Python 与依赖版本、CUDA 信息、Git commit/dirty 状态，以及本地论文 PDF 的路径、存在状态和 SHA-256；共用 output-dir 时不会互相覆盖。原始实验目录、checkpoint 和 PNG 默认不进入 Git。验收器默认把 `figure4_validation.json` / `figure5_validation.json` 写在 summary 与 PNG 所在的同一输出目录；`--output` 只用于显式覆盖该路径。

## 共同的候选选择规则

Figure 4 和 Figure 5 共用 `qubo_math.py` 中的退火求解逻辑：

1. 保留 `neal` 返回的全部 SA reads，并按能量升序排列。
2. 从低能量到高能量逐个解码，选择第一个同时满足编码约束和四相总和约束的候选。
3. 如果整批 reads 都没有可行候选，才用同一个 `x^T Q x + bias` 精确比较“最低采样不可行能量”和“可行编码域最优能量”，据此分类错误；正常成功路径不执行枚举。
4. 诊断最多枚举 `MAX_EXACT_DIAGNOSTIC_STATES = 1_000_000` 个可行状态。直接编码只枚举 count 总和为 `num_levels` 的状态，CGFM 枚举全部角度 count；超限时不开始枚举并保持原因未分类。所有内置 `paper/quick/test` 编码均低于该上限。
5. 无论诊断分类为何，本轮都明确失败；精确枚举只提供错误证据，不返回候选、不执行 cure、replacement 或精确求解 fallback。
6. 只有已找到的可行 SA 候选与已有 composition 重复时，才生成唯一的 random replacement。

summary 会记录 `infeasible_sa_samples_skipped`、`max_feasible_candidate_rank`、`duplicate_replacements`、`random_replacements` 和 `random_replacement_draws`。旧版 `invalid_replacement` 路径已移除。

命名统一使用 D-Wave 的 `reads` 术语：配置字段为 `SAConfig.reads`，CLI 首选 `--sa-reads`；`--sa-runs` 只保留为兼容别名。`experiment_runtime.py` 统一负责 device、依赖、seed schedule、FM seed plan、训练后端名、replacement 尝试上限、严格整数、原子 JSON、文件日志和 manifest 运行环境元数据。所有整数配置接受 Python/NumPy 整数并规范化为 Python `int`，但运行后 JSON 验收只接受非 bool 的原生 JSON integer；两者都不把浮点数或字符串静默转换成整数。

项目 base seed 必须位于 `0 <= seed <= 2**31 - 1`。所有依赖有界的逐轮随机流使用 `mixed_radix_v1`：

```text
inner = (((base_seed * trajectory_count + trajectory_index) * iterations + iteration)
         * stream_count + stream_index)
bounded_seed = inner * 2 + figure_index
```

`figure_index` 对 Figure 4/5 分别为 `0/1`。Figure 4 固定 10 个 trajectory 槽位（canonical objective index × 2 + setting index）和 `cgfm=0, sa=1` 两个 bounded stream；Figure 5 固定 `w_ddts=0, wo_ddts=1` 两个槽位和一个 SA stream。runner 与 pipeline 都在 manifest、数据生成和训练前按完整 canonical 布局验证上限；任何越界直接失败，不取模或重映射。

每次 FM 训练在 NumPy seed 空间的高半区 `[2**31, 2**32 - 1]` 预留 `optuna_trials + 4` 个连续 seed：两个数据拆分 seed、一个 Optuna sampler seed、每个 trial 的 model seed，以及一个 final-fit seed。CGFM/SA 的 bounded seed 位于低半区，因此不同 stage 不会数值碰撞。Figure 4 每轮预留一个 FM block；Figure 5 每轮预留三个，`w_ddts` 使用 block 0，`wo_ddts` 的三个 objective 使用 block 0/1/2。实际 seed plan 写入 FM metadata，checkpoint 恢复会按 trajectory 和最后一轮重建核对。

Python `random.Random` 不受 `neal`/NumPy 上限约束，使用固定字段位打包 `namespace << 192 | base_seed << 128 | trajectory_index << 64 | iteration`。只有同一 base seed 的初始数据，以及 Figure 5 两种 setting 的 preference weights，是有意共享的随机流；replacement、CGFM、SA 和 FM 均按完整 trajectory/stage 隔离。

按照论文补充材料 S1.1，FM-to-QUBO 会丢弃不影响最优 bit 状态的整体偏置 `w0`。各 QUBO 项按实际二元多项式系数计算归一化尺度：线性项使用 `q[i,i]`，二次项使用 `q[i,j] + q[j,i]`；因此对称半系数矩阵和上三角全系数矩阵得到相同尺度，常数项及矩阵存储形式都不得改变 FM 目标与约束惩罚的相对强度。

完整 QUBO 的参数由共享 `QuboConfig` 作为一个契约管理：

```text
QuboConfig
  fm_objective_weight: 1.0
  system_penalty_weight: 650.0
  one_hot_penalty_weight: 1.0
  normalization_scheme: max_abs_polynomial_coefficient_v1
```

共享 dataclass 默认值和 `paper` preset 的 one-hot weight 都是论文值 `1.0`；两个 Figure 的 `quick/test` preset 显式覆盖为 `2.0`，不会改变直接构造 `QuboConfig()` 的语义。唯一受支持的 normalization scheme 是 `max_abs_polynomial_coefficient_v1`，语义就是上一段所述的“最大绝对二元多项式系数”，常数偏置不参与尺度计算。完整构建公式为：

```text
Q = w_fm * N(Q_fm)
  + w_sys_applied * N(Q_system)
  + w_one_hot * N(Q_one_hot)
```

bias 使用相同权重。Figure 4 的 `w_cgfm` 不适用 system term，因此其实际 `system_penalty_weight` 与 `system_scale` 都保存为 `0.0`；Figure 4 `wo_cgfm` 和 Figure 5 使用配置中的 system weight。三个权重都必须是有限非负实数并允许为 `0.0`；零权重项不进入最终 QUBO，其实际 scale 记为 `0.0`，但 pipeline 不因此跳过 FM 训练或其他算法步骤。`QuboStats` 保存 normalization scheme、三个实际应用权重、三个实际 normalization scale、变量数和最终加权 QUBO 的 `max_abs`，恢复和验收都与 `config.qubo` 核对。

Figure 4 和 Figure 5 共用 FM 数据拆分与训练规则：不少于 5 条数据时显式计算整数 train/validation/test 大小并保证三个子集非空，少于 5 条时三种用途共享当前小数据集。Optuna trial 只评估 train/validation，并且仅以 `validation_loss` 选择超参数；保留的 test 集只在所选超参数的最终模型拟合后评估。`tune_fm_hparams` 只返回超参数，`fit_torch_fm` 只执行一次最终模型训练。

按照论文 Eq. 18，one-hot 的数值层级固定为 `alpha_i = i / N_bits`，所有 block 使用相同的升序 bit 映射，零值仍由全零 bit-string 表示。补充材料 S7/S8 后所述的逐轮随机化只适用于 CGFM 正反映射中的相变量分配 `phase_permutation`，不打乱 `alpha_i`；Figure 4 的直接编码和 Figure 5 的四相编码因此不随 iteration seed 改变数值层级。

`compute_delta_t` 当前按论文 Eq. 15 的舍入系数实现分段线性拟合，并只把精确共晶点 `fSi = 0.128` 设为 `delta_T = 0`。由于两侧舍入后的直线没有严格通过该零点，极窄邻域内可能得到非物理负值；canonical `quick` 的离散网格不会落入该区间，但连续输入或自定义 levels 仍可能触发。后续修正应保留论文系数并将计算结果截断到非负范围，同时增加共晶点两侧的定向测试。

## 第一部分：Figure 4 复现

### Figure 4 当前状态

- 已完成 `wo_cgfm`（直接四相编码）和 `w_cgfm`（CGFM 角度编码）两条流程。
- 已完成 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T` 五个 objective；优化方向由 `ObjectiveSpec` 统一定义。
- 已完成 PyTorch FM、FM-to-QUBO、SA 全 reads 可行解筛选、重复候选替换、best-so-far 更新及多 seed 聚合。
- 已完成 per-trajectory checkpoint、两层恢复一致性校验、`--resume`、manifest、runner/plot 日志、checkpoint schema v6、summary schema v5 和多 summary 合并绘图；旧 schema 或错误 seed scheme 不自动迁移。
- `figure4_runner.py` 默认使用 `quick`，`test` 用于烟测，`paper` 只作参数参考。
- 单元测试和烟测流程已通过；现有 `figure4_quick_gamma2/figure4_summary.json` 已按 one-hot penalty `gamma=2.0` 的 canonical `quick` 契约验收通过，报告与图片同目录保存。

### Figure 4 代码结构

Figure 4 按“数据与运行环境 → 配置与策略 → 数学与模型 → pipeline → I/O 与入口”分层：

```text
alloy_dataset_generator.py       组成采样、四相归一化、五个真实性质计算
experiment_runtime.py            seed/I-O/日志/依赖/device 与运行元数据
experiment_config.py             Figure 4/5 共用 EncodingConfig/FMConfig/SAConfig/QuboConfig
figure4_experiment_config.py     Figure 4 objective、运行配置和 preset
figure4_setting_strategies.py    wo_cgfm/w_cgfm 编码与解码策略分发
qubo_math.py                     共用离散编码、CGFM、QUBO 惩罚、SA、筛选与失败诊断
fm_torch.py                      共用 PyTorch FM、Optuna、LBFGS、FM-to-QUBO
figure4_pipeline.py              trajectory、active learning、checkpoint 恢复与聚合
figure4_outputs.py               checkpoint v6 外层校验与 Figure 4 标准路径
figure4_runner.py                CLI、配置解析、环境检查、manifest、pipeline 调用
plot_figure4.py                  严格读取一个或多个 summary v5 并绘图
validate_reproduction.py         严格契约、自洽重建、report v3 与非阻断 diagnostics
test_figure4_pipeline.py         配置、数学、流程、输出、绘图和 README 契约测试
```

主依赖方向为：

```text
figure4_runner
  -> experiment_config / figure4_experiment_config / experiment_runtime / figure4_outputs
  -> figure4_pipeline
       -> alloy_dataset_generator
       -> figure4_setting_strategies
       -> fm_torch
       -> qubo_math
       -> figure4_outputs

plot_figure4 / validate_reproduction -> figure4_summary.json
```

runner 是唯一受支持的运行入口；active-learning 逻辑只放在 pipeline；编码差异只放在 strategy；Figure 专属路径/外层 schema 放在 outputs，共用 JSON 原子写入和日志放在 `experiment_runtime.py`。

### Figure 4 算法流程

1. runner 解析 preset 和显式覆盖参数，检查训练依赖与 device，生成 base seed 列表并验证完整 `mixed_radix_v1` schedule，然后写 `figure4_manifest.json`。
2. `generate_initial_dataset_batch` 为每个 seed 生成一份初始合金数据；同一 seed 下所有 setting/objective 从相同初始数据出发。
3. pipeline 对 `seed × objective × setting` 的笛卡尔积运行独立 trajectory；一条 trajectory 的唯一身份为 `(setting, objective, seed)`。
4. 每轮读取当前数据集的真实 objective。最大化目标先取负，最小化目标保持原值，再做 z-score，使 FM/QUBO 始终按“越小越好”求解。
5. strategy 创建当轮离散编码并编码训练特征：
   - `wo_cgfm` 使用四个 composition block；各 block 按 Eq. 18 固定使用升序 `alpha_i`，QUBO 需要 `system penalty + one-hot penalty`。
   - `w_cgfm` 每轮只随机打乱四个相进入 S7/S8 映射的顺序，再使用三个固定 `alpha_i` 的 CGFM angle block；解码天然得到非负且总和为 1 的四相 composition，因此只加 `one-hot penalty`。
6. 当轮 CGFM、FM block 和 SA seed 由完整 trajectory/iteration/stage 身份派生。`tune_fm_hparams` 只选择 FM 超参数，`fit_torch_fm` 随后执行一次 final fit；`fm_to_qubo` 按补充材料 S1.1 丢弃整体偏置 `w0`，将线性项和交互项展开为 QUBO；`build_single_objective_qubo` 按 `config.qubo` 的完整契约归一化并加权 FM、system 与 one-hot 项。
7. `solve_qubo_with_sa` 返回全部能量排序后的 reads，`select_lowest_energy_feasible_sample` 选择最低能量可行状态并记录其 rank 和跳过数量；仅当整批无可行候选时，才精确枚举编码可行域以分类 penalty-balance、heuristic-sampling 或未分类错误，随后仍终止该轮。
8. strategy 解码候选。新 composition 直接加入数据集；重复 composition 保留“重复”判定并加入唯一 random replacement；不可行候选不会进入替换分支。
9. 用真实性质更新该 objective 的 best-so-far，更新审计计数，并在每轮后原子写入 trajectory checkpoint。
10. 全部 trajectory 完成后，按 `{objective}:{setting}` 聚合各 seed 的 `mean/std/min/max` 曲线并写 summary schema v5；绘图和验收只读取 summary，不重新训练。

### Figure 4 对象命名与调用规则

| 层 | 对象或值 | 命名与职责 |
| --- | --- | --- |
| 规模 | `Figure4RunScale`、`FIGURE4_PRESET_NUM_SEEDS` | 只允许 `paper`、`quick`、`test`；默认 seed 数与 preset 一起由 Figure 4 配置模块定义，正式验收只使用 `quick`。 |
| 目标 | `ObjectiveSpec`、`FIGURE4_OBJECTIVES`、`FIGURE4_OBJECTIVE_NAMES` | objective 名固定为 `kappa`、`E`、`rho`、`delta_alpha`、`delta_T`；名称顺序由 objective specs 直接派生。 |
| 配置 | `Figure4ExperimentConfig` | 顶层持有 `experiment_config.py` 的 `EncodingConfig`、`FMConfig`、`SAConfig`、`QuboConfig`。 |
| setting | `Figure4Setting`、`FIGURE4_SETTINGS` | canonical 顺序为 `wo_cgfm`、`w_cgfm`，统一定义在 Figure 4 配置模块；不得使用大小写变体或显示名称代替机器值。 |
| 策略 | `SettingStrategy` | 统一接口为 `create_encoding`、`encode_rows`、`decode_candidate`；实现类为 `WOCGFMStrategy`、`WCGFMStrategy`。 |
| 编码 | `IterationEncoding` | `positive_count_by_bit` 固定实现 Eq. 18；仅 `w_cgfm` 设置逐轮变化的 `phase_permutation`。 |
| 运行状态 | `TrajectoryState` | 只表示可 checkpoint 的单轨迹可变状态。 |
| 结果 | `TrajectoryResult`、`AggregatedTrajectory`、`Figure4Summary` | 分别表示单轨迹、跨 seed 曲线和完整 summary；summary 聚合键固定为 `{objective}:{setting}`。 |
| 输出 | `Figure4OutputLayout` | 统一派生 summary、manifest、日志、图片和 checkpoint 路径。 |

调用规则：

1. Figure 4 实验只支持通过 `figure4_runner.py` 启动；runner 的公开导出仅为 `main` 和 `parse_args`，pipeline 中的实验和单 trajectory 函数只供 runner 与白盒测试使用，不提供程序化 API 或跨版本兼容承诺；内部实验函数未显式传 settings 时使用完整 `FIGURE4_SETTINGS`。
2. runner 统一执行依赖/device 检查、配置解析、manifest 写入以及 objective/setting/seed 公共契约；pipeline 不重复校验 objective 列表。
3. setting 和 seed 列表必须非空、无重复；seed 必须满足共同的整数、范围和逐轮派生规则。objective 子集由 runner 去重并按 canonical `FIGURE4_OBJECTIVES` 顺序执行，而不是按 CLI 输入顺序执行。
4. 用户通过必填的 `--output-dir` 明确选择输出位置；内部路径统一由 `figure4_output_layout(...)` / `Figure4OutputLayout` 派生。同一输出目录不得混用不同 config、setting、objective 或 seed 身份后继续 `--resume`。
5. 恢复时先由 outputs 校验 checkpoint 外层 schema、`seed_derivation`、字段类型和 trajectory 身份，再由 pipeline 校验 `TrajectoryState` 的 rows、best 曲线、计数、SA 审计、QUBO stats 和最新 FM seed plan；任何不一致都在完成跳过或下一轮随机流重建前失败。

### Figure 4 配置规则

`Figure4ExperimentConfig` 的结构为：

```text
Figure4ExperimentConfig
  num_samples
  iterations
  encoding: EncodingConfig(num_levels)
  fm: FMConfig(optuna_trials, device)
  sa: SAConfig(reads, sweeps)
  qubo: QuboConfig(fm_objective_weight, system_penalty_weight,
                   one_hot_penalty_weight, normalization_scheme)
```

| preset | initial samples | iterations | levels | Optuna trials | SA reads | SA sweeps | default seeds | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test` | 10 | 2 | 8 | 0 | 32 | 12 | 1 | 烟测 |
| `quick` | 100 | 100 | 10 | 3 | 100 | 500 | 3 | 小规模项目验收；不承担论文 50-level SA 难度 |
| `paper` | 100 | 600 | 50 | 20 | 1000 | 3000 | 20 | 参数参考，不验收 |

- 配置先由 `preset_config` 解析，再由 `resolve_experiment_config` 应用 CLI 显式覆盖；未覆盖字段保留 preset 值。
- `num_samples`、`iterations`、SA reads/sweeps 必须为严格正整数；`num_levels` 必须是大于 1 的整数；`optuna_trials` 必须是非负整数；device 只允许 `cpu` 或 `cuda`。
- CLI 可覆盖 `--num-samples`、`--iterations`、`--num-levels`、`--optuna-trials`、`--sa-reads`、`--sa-sweeps`，以及 `--fm-objective-weight`、`--system-penalty-weight`、`--one-hot-penalty-weight`。normalization scheme 当前只有一种受支持值，因此不开放 CLI。
- QUBO preset 契约为：`paper` 的 one-hot penalty `gamma=1.0`，`quick/test` 为 `gamma=2.0`；CLI 显式 `--one-hot-penalty-weight` 最后应用并优先于 preset。
- `--settings` 必填；`--objectives` 省略或传空时运行全部五个目标。
- `--seed-start` 必须非负，`--num-seeds` 必须为正；默认 seed 为从 0 开始的连续列表，列表末端和最后一轮派生 SA seed 都不得超过 `2**31 - 1`。
- 请求 `--device cuda` 但 CUDA 不可用时，会在实验开始前失败，不静默回退到 CPU。
- canonical `quick` 验收要求表中精确配置、3 个默认 seed、全部 setting 和 objective；任何数值或 seed 覆盖都可运行，但不属于 canonical 验收结果。
- validator 不再维护第二套 objective、setting 或 quick 数值；验收规模由 `preset_config("quick")` 和 `FIGURE4_PRESET_NUM_SEEDS` 派生。
- 旧 schema 的 50-level `quick` checkpoint 不再读取或迁移；50-level 开发运行仍可通过新输出目录显式传 `--num-levels 50` 重跑。若它在某一确定性 SA 批次失败，保持同一 config 和 seed 恢复不会改变该批次。

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
  --summary figure4_quick\figure4_summary.json
```

未传 `--output` 时，报告写入 `figure4_quick\figure4_validation.json`。

### Figure 4 输出规则

```text
<output-dir>/
  figure4_summary.json
  figure4_manifest.json
  figure4.png
  figure4_validation.json
  logs/
    figure4_runner.log
    plot_figure4.log
  trajectories/
    {setting}_{objective}_seed_{seed}.json
```

| 输出 | 规则 |
| --- | --- |
| `figure4_manifest.json` | schema v2；记录 `figure=4`、`mixed_radix_v1`、命令、preset、含完整 QUBO 契约的 resolved config、runtime、seed/objective/setting 和标准路径。 |
| trajectory checkpoint | schema v6；绑定 `seed_derivation + setting + objective + seed + config`，并持久化新的 `QuboStats`。每轮仍写完整状态，但使用紧凑流式 JSON；旧 schema、错误 scheme 或损坏状态都不迁移。 |
| `figure4_summary.json` | schema v5；包含 `seed_derivation`、完整 config、seed/objective/setting、所有 `trajectories` 和 `{objective}:{setting}` 聚合曲线。 |
| `figure4.png` | 由 `plot_figure4.py` 从一个或多个 schema v5 summary 生成；曲线字段必须是一维、等长且有限。 |
| `figure4_validation.json` | report schema v3；validator 默认写在 summary/图片同目录。通过、验收失败和输入 JSON 损坏都会落盘；可用 `--output` 显式覆盖。 |
| 日志 | runner 和 plot 分文件记录；不得用日志替代机器可读 summary/validation report。 |

summary 和 manifest 保留缩进以便审查；所有 JSON 都通过同目录唯一临时文件流式写入，`flush + fsync` 后原子替换，并拒绝 NaN/Infinity。`trajectories/*.json` 是“当前单条轨迹的完整 checkpoint 快照”，每轮原子覆盖同一个文件；紧凑单行是为了降低每轮完整重写的 I/O 常数，不是 JSONL，也不表示只保存了一条记录或发生数据缺失。可用 `python -m json.tool trajectories\wo_cgfm_kappa_seed_0.json` 格式化查看。每条轨迹必须满足：`completed_iterations == iterations`、`len(best_so_far) == iterations`、`final_dataset_size == num_samples + iterations`，并且 `accepted_sa_candidates + duplicate_replacements == iterations`。不同配置必须使用不同输出目录；旧 schema checkpoint 不会自动覆盖或迁移。

### Figure 4 验收规则

Figure 4 canonical `quick` 严格检查：

- summary、trajectory、FM metadata 和 QUBO stats 使用精确字段集合；整数不得是 bool、浮点数或字符串，所有数值必须有限。
- 完整配置必须精确等于 `quick` preset（device 允许 `cpu` 或 `cuda`），seed 必须为 `[0, 1, 2]`，objective/setting 的值和顺序必须是 canonical 顺序。
- 轨迹必须按 `seed → objective → setting` 顺序完整出现；best-so-far 按各 objective 方向单调，计数、SA rank/read 范围、数据集增长、最新 FM seed plan 和 QUBO 编码维度必须自洽。
- 验收器从 30 条轨迹重新计算每个 `{objective}:{setting}` 的 mean/std/min/max，以 `rtol=atol=1e-12` 精确核对 stored aggregate；Figure 4 summary v5 没有逐轮 proposed/added records，因此验收器不声称重建这些信息。
- validation report 使用 `report_schema_version: 3`。失败 detail 为 JSON-safe 的 `{path, rule, expected, actual}` error 列表；非法 JSON、非对象根和读取失败也会生成 report v3 并返回退出码 1。

只有 canonical quick 规模不匹配、而其余结构和数值证据有效时，仍会生成开发配置 diagnostics；数据完整性失败时 diagnostics 写入 `skipped_reason`。`w_cgfm` 是否表现出论文报告的优势只写入 `diagnostics.cgfm_paper_direction`，不影响验收结果。

## 第二部分：Figure 5 复现

### Figure 5 当前状态

- 已完成 `w_ddts` 和 `wo_ddts`（weighted-sum baseline）两条多目标流程。
- 已完成最大化 `kappa`、最大化 `E`、最小化 `rho` 的统一方向处理、可复现 preference weights、DDTS 人工目标、三 FM QUBO 合并和 Pareto front。
- 已完成每轮 proposed solution 与实际 added solution 的分别记录；重复 proposed solution 会保留，random replacement 不进入 Pareto front。
- 已完成 SA 全 reads 可行解筛选、候选 rank 诊断、checkpoint schema v5、summary schema v5、两层恢复一致性校验、`--resume`、manifest 和 runner/plot 日志；旧 schema 或错误 seed scheme 不自动迁移。
- 已完成单/双 setting 绘图、3D 总览、迭代窗口及多 seed 显式选择；canonical `quick` 要求两个 setting 完整对比。
- 单元测试和烟测流程已通过；现有 `figure5_quick_gamma2/figure5_summary.json` 已按 one-hot penalty `gamma=2.0` 的 canonical `quick` 契约验收通过，报告与图片同目录保存。

### Figure 5 代码结构

Figure 5 复用数据、FM、QUBO 和运行环境基础层，并在其上增加多目标 scalarization、Pareto 和独立 pipeline：

```text
alloy_dataset_generator.py       多目标初始数据、组成归一化、真实性质计算
experiment_runtime.py            seed/I-O/日志/依赖/device 与运行元数据
experiment_config.py             共用 EncodingConfig/FMConfig/SAConfig/QuboConfig
fm_torch.py                      共用 PyTorch FM、Optuna、LBFGS、FM-to-QUBO
qubo_math.py                     共用直接编码、QUBO 惩罚、SA、可行解筛选与失败诊断
figure5_experiment_config.py     Figure 5 配置与 paper/quick/test preset
figure5_scalarization.py         setting 契约、preference、DDTS 与数学对照
figure5_pareto.py                mixed-sense 支配关系与 Pareto front
figure5_pipeline.py              多目标 active learning、checkpoint、solutions 与 summary
figure5_outputs.py               checkpoint v5 外层校验与 Figure 5 标准路径
figure5_runner.py                CLI、配置解析、环境检查、manifest、pipeline 调用
plot_figure5.py                  单/双 setting、seed 选择、3D 总览和迭代窗口
validate_reproduction.py         rows/records/front 重建、report v3 与精确 front diagnostics
test_figure5_*.py                scalarization、Pareto、pipeline 和输出契约测试
test_plot_figure5.py             summary 校验、seed 选择和绘图测试
```

主依赖方向为：

```text
figure5_runner
  -> experiment_config / figure5_experiment_config / experiment_runtime / figure5_outputs
  -> figure5_pipeline
       -> alloy_dataset_generator
       -> figure5_scalarization / figure5_pareto
       -> fm_torch / qubo_math
       -> figure5_outputs

plot_figure5 / validate_reproduction -> figure5_summary.json
```

Figure 5 不调用 Figure 4 pipeline，也不使用 CGFM strategy；它只复用稳定的配置子对象、FM 和 QUBO/SA 数学组件。

### Figure 5 算法流程

1. runner 解析 preset 和覆盖参数，检查依赖/device，生成 base seed 列表并验证完整 `mixed_radix_v1` schedule，然后写 `figure5_manifest.json`。
2. `generate_initial_dataset_multi_objective_batch` 为每个 seed 生成共享初始数据；同一 seed 下 `w_ddts` 和 `wo_ddts` 从相同数据出发。
3. pipeline 对 `seed × setting` 运行独立 trajectory；一条 trajectory 的唯一身份为 `(setting, seed)`。
4. `figure5_scalarization.preference_weights_for_iteration(seed, iteration)` 用独立 Python namespace 为每轮生成三个非负且和为 1 的确定性权重；同一 seed/iteration 的两个 setting 使用相同权重。pipeline 和验收器共同调用这一纯函数，不各自维护权重重建规则。
5. 两条 scalarization 路径分别构造 FM/QUBO：
   - `w_ddts`：对三个真实目标做 z-score，构造 utopian point，并以最大加权方向距离作为一个 FM 的人工 target：

     ```text
     max(
       w_kappa * (u_kappa - z_kappa),
       w_E     * (u_E     - z_E),
       w_rho   * (z_rho   - u_rho)
     )
     ```

     生产实现对应论文实际 DDTS Eq. 6 的最大加权距离，不把 Eq. 4 中用于一般 Tchebycheff 参考形式的 Manhattan augmentation/正则项加入人工 target。

   - `wo_ddts`：将目标转换为 `-kappa`、`-E`、`rho` 后分别 z-score，训练三个 FM，再在 QUBO 层合并：

     ```text
     Q = w_kappa * Q_kappa + w_E * Q_E + w_rho * Q_rho
     ```

6. `w_ddts` 使用 FM block 0；`wo_ddts` 的三个 objective 分别使用 block 0/1/2。三 FM 的 objective QUBO 先按 preference weights 合并，再由 `config.qubo.fm_objective_weight` 整体加权；两条路径都使用按 Eq. 18 固定升序 `alpha_i` 的四个 composition one-hot block，并按 `config.qubo` 加权 system 与 one-hot penalty。Figure 5 不使用 CGFM，因此没有逐轮 phase permutation。
7. SA 返回全部 reads；流程选择最低能量可行候选，记录 `sa_energy`、`feasible_candidate_rank` 和跳过的不可行样本数；仅当整批无可行候选时，才执行与 Figure 4 共用的精确错误分类并终止该轮。
8. 每轮同时保存 `proposed_solution` 与 `added_solution`。新候选两者相同；重复候选的 proposed 保持不变，added 改为唯一 random replacement，状态为 `duplicate_replacement`。
9. 每轮更新 trajectory state 并原子写紧凑 checkpoint；恢复时先由 outputs 校验外层 schema、`seed_derivation`、字段类型和 trajectory 身份，再由 pipeline 校验 rows、iteration records、候选决策、计数、SA 审计、最新 scalarization 元数据、FM seed plans 和 QUBO stats。下一轮 replacement stream 由 iteration 身份直接重建，不依赖跨轮 RNG 状态；任何不一致都在完成跳过或随机流重建前失败。
10. 完成后将所有 proposed solution（包括重复 proposal）展平到 summary 的 `solutions`；random replacement 不进入该列表。
11. `pareto_front` 先按 composition 去除重复 proposal并保留首次记录，再按 `setting × seed` 独立计算非支配集：A 支配 B 当且仅当 `A.kappa >= B.kappa`、`A.E >= B.E`、`A.rho <= B.rho`，且至少一个目标严格更优。
12. summary schema v5 的 `pareto_fronts` 为 `{setting, seed, solutions}` 记录列表；绘图器读取指定 seed 的 stored front 并用 proposed solutions 重算核对，验收器同时检查 front 身份、去重和精确 front diagnostics。

### Figure 5 对象命名与调用规则

| 层 | 对象或值 | 命名与职责 |
| --- | --- | --- |
| 规模 | `Figure5RunScale`、`FIGURE5_PRESET_NUM_SEEDS` | 只允许 `paper`、`quick`、`test`；默认 seed 数由 Figure 5 配置模块定义，正式验收只使用 `quick`。 |
| 配置 | `Figure5ExperimentConfig` | 顶层持有 `experiment_config.py` 的 `EncodingConfig`、`FMConfig`、`SAConfig`、`QuboConfig`。 |
| setting | `Figure5Setting`、`FIGURE5_SETTINGS` | 只允许 `w_ddts`、`wo_ddts`；canonical 顺序和类型契约均定义在 `figure5_scalarization.py`。 |
| 目标 | `FIGURE5_OBJECTIVES` | 固定顺序为 `kappa`、`E`、`rho`；方向由 `FIGURE5_OBJECTIVE_SENSES` 定义。 |
| scalarization | `ScalarizationResult`、`ObjectiveTargetsResult` | 分别表示单人工 target 和三目标独立 target；权重顺序必须与 `FIGURE5_OBJECTIVES` 一致。 |
| 迭代记录 | `SolutionPoint`、`IterationRecord` | 分别表示一个设计点和一轮 proposed/added 决策；iteration 使用从 0 开始的整数。 |
| 运行状态 | `Figure5TrajectoryState`、`Figure5TrajectoryResult` | 分别表示可恢复状态和完整单轨迹结果。 |
| Pareto 结果 | `Figure5ParetoFront` | 保存一个固定 `setting + seed` 的去重非支配 solutions。 |
| 总结果 | `Figure5Summary` | 保存 trajectories、展平后的 proposed `solutions` 和按 `setting × seed` 分组的 `pareto_fronts`。 |
| 输出 | `Figure5OutputLayout` | 统一派生 summary、manifest、日志、图片和 checkpoint 路径。 |

调用规则：

1. Figure 5 完整实验只支持通过 `figure5_runner.py` 启动；runner 的公开导出仅为 `main` 和 `parse_args`，pipeline 中的实验和单 trajectory 函数只供 runner 与白盒测试使用，不提供程序化 API 或跨版本兼容承诺。
2. runner 统一执行 settings/seed 公共契约、固定 objectives、依赖/device 检查、配置解析和 manifest 写入；连续 seed 直接通过共享的 `resolve_contiguous_seeds(...)` 解析，不定义 Figure 专属包装函数。
3. pipeline 的 `w_ddts` 路径调用 `compute_ddts_targets(...)`；`wo_ddts` 路径必须调用 `compute_individual_objective_targets(...)` 后训练三个 FM，并在 QUBO 层合并。
4. `compute_weighted_sum_reference_targets(...)` 只用于 weighted-sum 数学对照，不表示生产 pipeline 的训练调用路径；不存在按 setting 分发单 FM target 的公共函数。
5. `decision_status` 只允许 `accepted` 或 `duplicate_replacement`；Pareto front 只读取 `proposed_solution`，不得把 random replacement 当成优化器提出的点。
6. setting 和 seed 列表必须非空、无重复，并满足共同的整数、范围和逐轮派生规则。同一 seed/iteration 的权重由函数确定，不允许调用方为两个 setting 分别随机采样。
7. 路径必须通过 `figure5_output_layout(...)` / `Figure5OutputLayout` 获取；多 seed summary 绘图必须用 `--seed` 明确选择一条 seed，stored front 不跨 seed 混合。
8. 恢复时 latest FM metadata 必须包含实际 split/tuner/trial/final seed plan；`w_ddts` 核对一个 model，`wo_ddts` 核对三个 model。

### Figure 5 配置规则

`Figure5ExperimentConfig` 的结构为：

```text
Figure5ExperimentConfig
  num_samples
  iterations
  encoding: EncodingConfig(num_levels)
  fm: FMConfig(optuna_trials, device)
  sa: SAConfig(reads, sweeps)
  qubo: QuboConfig(fm_objective_weight, system_penalty_weight,
                   one_hot_penalty_weight, normalization_scheme)
```

| preset | initial samples | iterations | levels | Optuna trials | SA reads | SA sweeps | default seeds | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test` | 10 | 2 | 8 | 0 | 32 | 12 | 1 | 烟测 |
| `quick` | 500 | 150 | 25 | 3 | 100 | 500 | 1 | 项目验收 |
| `paper` | 500 | 1000 | 25 | 20 | 1000 | 3000 | 1 | 参数参考，不验收 |

- 配置先由 Figure 5 的 `preset_config` 解析，再由 `resolve_experiment_config` 应用显式覆盖；Figure 4 和 Figure 5 的 preset 名相同，但数值彼此独立。
- 数值、device、seed、SA 和 QUBO 权重遵循与 Figure 4 相同的严格类型及派生范围规则；CLI 首选 `--sa-reads`，三个 QUBO 权重使用相同的显式 override 名称，normalization scheme 不开放 CLI。
- QUBO preset 契约同样是 `paper` 使用 one-hot penalty `gamma=1.0`，`quick/test` 使用 `gamma=2.0`；CLI 显式权重优先。
- `--settings` 必填；objectives 固定为 `kappa E rho`，Figure 5 CLI 不接受 objective 子集。
- Figure 5 使用四相直接 one-hot 编码和 system penalty，不接受 CGFM setting。
- 默认运行 seed 0；显式传入 `--num-seeds N` 时每个 setting 都运行 N 条 trajectory。
- 多 seed 和数值覆盖是受支持的开发配置，但 canonical `quick` 验收要求表中精确配置、seed 0 和两个 setting。
- validator 直接使用 `FIGURE5_SETTINGS`，并由 `preset_config("quick")` 和 `FIGURE5_PRESET_NUM_SEEDS` 派生验收规模。
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
  --summary figure5_quick\figure5_summary.json
```

未传 `--output` 时，报告写入 `figure5_quick\figure5_validation.json`。

### Figure 5 输出规则

```text
<output-dir>/
  figure5_summary.json
  figure5_manifest.json
  figure5.png
  figure5_validation.json
  logs/
    figure5_runner.log
    plot_figure5.log
  trajectories/
    {setting}_seed_{seed}.json
```

| 输出 | 规则 |
| --- | --- |
| `figure5_manifest.json` | schema v2；记录 `figure=5`、`mixed_radix_v1`、命令、preset、含完整 QUBO 契约的 resolved config、runtime、seed/setting、固定 objectives 和标准路径。 |
| trajectory checkpoint | schema v5；绑定 `seed_derivation + setting + seed + config`，并持久化新的 `QuboStats`。每轮仍写完整状态，但使用紧凑流式 JSON；旧 schema、错误 scheme 或损坏状态都不迁移。 |
| `figure5_summary.json` | schema v5；包含 `seed_derivation`、完整 config、trajectories、全部 proposed `solutions` 和每个 `setting × seed` 的 `pareto_fronts`。 |
| `figure5.png` | 由 `plot_figure5.py` 从 schema v5 summary 生成；多 seed 时必须显式传 `--seed`，stored front 必须与该 seed 的 proposed solutions 重算结果一致。 |
| `figure5_validation.json` | report schema v3；validator 默认写在 summary/图片同目录，通过、验收失败和输入损坏均落盘；`--output` 可覆盖。 |
| 日志 | runner 和 plot 分文件记录；验收结论只来自 validation report。 |

Figure 5 的 `trajectories/*.json` 与 Figure 4 相同，也是每轮覆盖的单行完整 checkpoint 对象，不是 JSONL；同样可以交给 `python -m json.tool` 格式化查看。每条轨迹必须满足：`completed_iterations == iterations`、`len(iteration_records) == iterations`、`final_dataset_size == num_samples + iterations`，并且 `accepted_sa_candidates + duplicate_replacements == iterations`。每条 `IterationRecord` 必须同时保留 setting、seed、iteration、weights、scalarization method、decision status、proposed/added solution 和 SA 诊断字段。

### Figure 5 验收规则

Figure 5 canonical `quick` 严格检查：

- summary、trajectory、iteration record、solution、Pareto front、FM metadata 和 QUBO stats 使用精确字段集合；完整配置精确等于 quick preset（device 允许 CPU/CUDA），seed/objective/setting 的值和顺序均为 canonical 契约。
- 验收器按 seed 重新生成并离散化 500 条初始数据，再用 150 条 iteration records 重建 seen compositions 和实际 added rows；每轮核对确定性 preference weights、scalarization method、SA rank/skipped、sample id、网格 composition，以及由 `build_dataset_row` 重算的 `kappa/E/rho`。
- `accepted` 必须 proposed 等于 added 且 composition 新颖；`duplicate_replacement` 必须 proposed 已出现、added 新颖且 draws 合法。全部 counters、latest weights、最后一轮 scalarization metadata、FM seed plans 和 QUBO stats 都从 records/rows 重新推导。
- 顶层 `solutions` 必须精确等于 records 中 proposed solutions 的顺序展平结果，因此 random replacement 不可能混入；每个 `setting × seed` stored front 再从这些 proposals 按混合方向重算并逐项核对。
- validation report 使用 `report_schema_version: 3`；数据完整性不足时 metrics/diagnostics 不运行并写入 `skipped_reason`，仅 quick 规模不同但结构有效的开发输出仍可生成非阻断 diagnostics。

验收器仍会枚举 25-level 离散设计空间。`metrics.by_setting_seed` 按 `seed_list` 外层、canonical setting 内层的固定顺序，分别报告每个 `setting × seed` 的 unique proposals、exact-front hits、coverage、precision 和 spacing CV；composition 只在当前 pair 内去重，random replacement 不参与。`metrics.setting_aggregates` 再对各 seed 计算 mean/std，绝不先汇池 proposal。

`diagnostics.ddts_comparison.by_seed` 逐 seed 保存 `w_ddts / wo_ddts` 的 coverage 与 spacing 比值，`aggregate` 保存这些比值的 mean/std。分母为零或 spacing 不可定义时写 JSON `null`，不使用 epsilon 伪造比值；这些 diagnostics 不设置通过阈值。
