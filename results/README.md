# Reproduction evidence

该目录只保存可审查的小型状态与验收 JSON，不保存原始 checkpoint、日志或 PNG。

通过 `validate_reproduction.py --output results\<name>.json` 生成报告。顶层 `scope=quick_workflow` 且 `passed=true` 表示项目的小规模流程验收通过；本项目不验收 paper scale，也不把论文趋势作为通过条件。

`legacy_figure4_quick_audit.json` 和 `legacy_figure5_quick_audit.json` 是代码修复前输出的审查快照；其中的旧目录名保留为历史来源标识，它们不能作为当前 schema 的通过证据。
