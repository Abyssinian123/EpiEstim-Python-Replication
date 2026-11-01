# Nash2023 引擎迁移来源记录

- 来源文件：`C:\Users\Shaira\Desktop\Nash2023_Python复现包\python_reproduction\epiestim.py`
- 原始来源文件：`C:\Users\Shaira\Desktop\Nash2023_Python复现包\python_reproduction\epiestim.py`
- 原始来源 SHA-256：`CEA5D440D554BB1C2DEAE1E372DFBA478EF91698F7B184BFF7238A14D1E1909B`
- 正式集成位置：`src/epiestim/epiestim.py`，由 A 提交 `079c67a` 提供。
- 正式文件 SHA-256：`4DB99C1179E2B6BCAC4C287D8E89E5F928CD85206C07F3C3C99175826EF242C8`
- 来源说明：A 提供包 README 声明估计引擎与 R 存储结果校验至约 `1e-14`。包内没有单独的 Git 版本号。
- 状态：B 曾使用临时 `src/epiestim/nash2023_engine.py` 和 `src/epiestim/engine.py`。A 的正式实现合入后，实验改为从 `epiestim` 包导入；两份临时文件已删除。
- 行为核查：两版 SHA 不同，但本次同种子 10 轨迹回归的模拟/报告数组、窗口及完整 Rt 输出逐值相同。
