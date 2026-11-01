# 报告率创新实验结果格式

本说明对应 `results/reporting_nash2023/formal/` 的正式 100 轨迹批次。病例报告模型为 `Y_t | I_t ~ Binomial(I_t, p_t)`；真实 Rt 为恒定 1.25。A 模拟器的 70 个返回日不包含前置 7 天、每日 10 例的历史种子。

## 文件清单

| 文件 | 内容 |
|---|---|
| `S0.npz` … `S3.npz` | 每个情景的真实日病例、报告日病例、周病例、报告率向量、真实 Rt 向量和 EM 重建日病例。 |
| `estimates_all.csv` | 日度报告数据估计与周数据 EM 估计的合并长表，保留完整 11 个 R 输出字段。 |
| `estimates_daily.csv` | `estimates_all.csv` 中 `estimation_method=daily_reported` 的子集。 |
| `estimates_weekly_em.csv` | `estimates_all.csv` 中 `estimation_method=weekly_em` 的子集。 |
| `run_status.csv` | 每个“模拟轨迹×报告情景”的运行状态、计时、异常计数、病例总数及 EM 周总量误差。失败轨迹也保留记录。 |
| `boundary_cases.csv` | 全零、初始病例后灭绝、低病例三类边界输入在日度与 EM 估计中的状态及错误。它们是独立边界检查，不计入 100 条疫情模拟。 |
| `metadata.json` | 模型参数、SI 设定、随机种子、软件版本、引擎提交与校验值、耗时和情景诊断。 |
| `rt_estimates.png` | 各情景、两种估计方法的平均后验均值/区间与恒定真 Rt 的描述性图。统计检验应由分析代码完成。 |
| `formal_check_report.md` | 正式批次检查摘要。 |

## NPZ 数组

| 键 | 形状 | dtype/单位 | 含义 |
|---|---:|---|---|
| `true_incidence` | `(100, 70)` | 整数计数/病例 | A 的 Poisson renewal 模拟器生成的真病例。第 2 维为 1-based 第 1–70 模拟日。 |
| `observed_incidence` | `(100, 70)` | 整数计数/病例 | 对应真病例经 `Binomial(I_t,p_t)` 抽样所得。 |
| `weekly_incidence` | `(100, 10)` | 整数计数/病例/周 | 日报告病例以 day 1 为起点，每 7 天求和。 |
| `reporting_rates` | `(70,)` | 概率 `[0,1]` | 对应每个模拟日的 `p_t`。 |
| `true_rt` | `(70,)` | 无量纲 | 所有 70 天恒为 1.25，是传给 A 模拟器的生成参数，不是估计值。 |
| `em_reconstructed_incidence` | `(100, 70)` | 浮点病例数/日 | 周报告病例的 EM 日度重建；若某轨迹 EM 失败，该行保留为 NaN，并在 `run_status.csv` 标记。 |

## 估计 CSV 字段

以下 11 列完整保留 A 的单次 Rt 输出：

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `t_start`, `t_end` | 整数/日 | 1-based 闭区间估计窗。仅当 `t_end > mean_si` 时输出。 |
| `Mean(R)`, `Std(R)` | 浮点/Rt | 后验 Rt 均值、标准差。 |
| `Quantile.0.025(R)`、`Quantile.0.05(R)`、`Quantile.0.25(R)` | 浮点/Rt | 后验分位数。 |
| `Median(R)` | 浮点/Rt | 后验中位数。 |
| `Quantile.0.75(R)`、`Quantile.0.95(R)`、`Quantile.0.975(R)` | 浮点/Rt | 后验分位数；2.5% 与 97.5% 界用于 95% CrI 覆盖和 `R_t < 1` 趋势分类。 |

每行还带有：

| 字段 | 类型 | 含义 |
|---|---|---|
| `simulation_id` | 整数 1–100 | 同一 id 在 S0–S3 中表示同一真实疫情轨迹，形成配对比较。 |
| `paired_truth_id` | 整数 1–100 | 与 `simulation_id` 相同的配对真值键。 |
| `rt_scenario` | 字符串 | 本批均为 `constant_1.25`。 |
| `reporting_scenario` | 字符串 | `S0`、`S1`、`S2` 或 `S3`。 |
| `estimation_method` | 字符串 | `daily_reported` 或 `weekly_em`。 |
| `true_rt` | 浮点/Rt | 恒定真值 1.25。 |

日度和 EM 结果已按 `simulation_id`、情景、`t_start`、`t_end` 对齐。成员 C 应使用这些键连接结果，不能按 CSV 行序配对。S1–S2 与 S1–S3 的比较均可在相同 `paired_truth_id` 下进行；S0 是完全报告基准。

## EM 回顾性重建的时间信息限制

周度 EM 的 `t_end` 仅表示**最终 Rt 估计滑动窗口的结束日**，不等于该估计所使用的原始观测信息截止日。EM 先利用完整周聚合病例重建日病例，再对重建序列估计 Rt；因此，重建后的某一天可能受到同一聚合周内较晚日期观测的影响。这是基于完整聚合数据的回顾性重建，不是仅使用当日及过去信息的实时滤波或预测。

具体到本实验的 S2，报告率从第 35 天起由 0.8 降至 0.4。第 29–35 天构成一个完整的 7 日聚合周，其中包含第 35 天的低报告率观测。该周总数参与周内日病例重建，因此即使 `t_end=34` 的最终 Rt 窗口结束于第 34 天，重建病例和相应 Rt 估计也可能已受到这份完整周数据的影响。这种提前出现的估计变化**不是时间索引错误**。

成员 C 不应把 `t_end=34` 的变化解释为方法具有提前识别疫情变化的能力、实时预测能力，或报告率下降在第 34 天已经发生。汇报时应说明：观测信息以周为单位提供，EM 使用完整周总数回顾性重建周内病例，窗口结束日早于报告率切换日的估计可能受该周后续观测影响。因此，本实验应定位为**回顾性 Rt 估计方法在报告率变化下的稳健性评价**，而非实时预警或预测评估。

## 运行状态与边界记录

`run_status.csv` 每行是一个模拟轨迹与报告情景组合。`status` 为 `ok`、`anomaly` 或 `failed`；`anomaly_count` 记录 Rt 数值列中的非有限值数。`trajectory_pipeline_seconds` 为该组合从报告抽样到估计结束的耗时，`simulation_batch_seconds` 记录共享真病例矩阵的整体生成耗时。`em_max_weekly_total_error` 应接近浮点误差量级。

`boundary_cases.csv` 中的边界样例不参与主实验统计。它用于明确记录估计器对全零、灭绝和低病例输入是否正常返回，不得将其混入 100 条正式模拟样本。

## 建议的 C 端指标计算

- **偏差**：逐行按 `Mean(R) - true_rt` 计算；聚合前保留配对键和窗口键。
- **RMSE**：按情景、估计方法、窗口或预先定义的汇总口径计算平方误差后开方。
- **95% CrI 覆盖率**：检查 `Quantile.0.025(R) <= true_rt <= Quantile.0.975(R)`。
- **错误趋势分类**：真 Rt=1.25 属于增长；若 `Quantile.0.975(R) < 1` 则估计误判为下降。其他分类规则按组内规范实施。
- 对后续变化 Rt，必须先由团队定义估计窗口真值（例如窗内 Rt 汇总规则）；当前恒定 Rt 可直接填入 1.25。
