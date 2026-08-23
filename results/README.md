# Reproduction evidence

该目录只保存可审查的小型状态与验收 JSON，不保存原始 checkpoint、日志或 PNG。

通过 `validate_reproduction.py --output results\<name>.json` 生成 `report_schema_version: 3` 报告。顶层 `scope=quick_workflow` 且 `passed=true` 表示项目的小规模流程验收通过；失败 check 的 `detail.errors` 由 `{path, rule, expected, actual}` 组成。论文趋势和 Pareto 表现只属于非阻断 diagnostics，本项目不验收 paper scale。

Figure 4 report 会重算 stored aggregate；Figure 5 report 会从确定性初始数据和 iteration records 重建 added rows、proposed solutions、Pareto fronts 与 per-seed metrics。输入无法读取、不是 JSON object 或 JSON 损坏时也会输出合法 report v3。旧 report v2 不迁移，也不作为当前通过证据。

`legacy_figure4_quick_audit.json` 和 `legacy_figure5_quick_audit.json` 是代码修复前输出的审查快照；其中的旧目录名保留为历史来源标识，它们不能作为当前 schema 的通过证据。
