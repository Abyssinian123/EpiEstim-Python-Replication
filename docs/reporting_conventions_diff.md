# A 版与当前项目规范差异记录（待团队确认）

本记录仅供集成审查，不替换任何 `CONVENTIONS.md`。规范来源为 A 复现包
`python_reproduction/CONVENTIONS.md` 与当前项目未跟踪的 `CONVENTIONS.md`。

| 项目 | A 版规范 | 当前项目规范 | 集成处理 |
|---|---|---|---|
| SI | 后验用 shifted Gamma `discr_si`；EM 的增长率链接默认 `si_variant="distcrete"`，为未平移 Gamma 的中心离散化 | 未区分两种 SI，统一描述为 `discr_si` | EM 保留 A 默认 `distcrete`；后验由 A 引擎的 `discr_si` 计算 |
| 时间窗口 | 1-based 闭区间；默认周滑窗 `t_start=2..T-6`、`t_end=8..T`；丢弃 `t_end <= mean_si` | 1-based 闭区间及默认窗口相同，但未规定窗口丢弃 | 使用 A 的窗口构造和丢弃逻辑 |
| 聚合余数 | `aggregate_inc` 以完整 `dt` 窗求和，末尾余数不参与结果 | 要求 T 可被 dt 整除，否则拒绝 | B 的 `aggregate_incidence` 预先拒绝余数 |
| 70 天/初始病例 | 先以 7 天×10例作为历史，再生成并返回 70 天 | 对“种子 7 天”是否计入返回长度未明确 | 正式真值采用 A 的返回数组；前置 7 天不计入输出的 70 天 |
| Rt 变化语义 | 变化点在前置历史后推进；模拟与画图用的理想化真值曲线有区别 | 简述 sudden 和 gradual 日期区间 | 本预实验只用恒定 Rt=1.25；未来变 Rt 需另行明确窗口真值定义 |
| 随机种子 | 常数 Rt=6，时变 Rt=5；使用 `default_rng` | 相同 | 本实验疫情 seed=6；报告抽样使用独立 seed |
| 周末转移 | 周末病例的 80% 移至后续周一/周二 | 写为 20% 移至工作日 | 与本次 S0–S3 无关，后续相关实验需全组确认 |

本次数值选择以 A 版规范为准；此处没有改动共享规范文件。
