# 临时 Nash2023 引擎来源记录

- 来源文件：`C:\Users\Shaira\Desktop\Nash2023_Python复现包\python_reproduction\epiestim.py`
- 集成副本：`src/epiestim/nash2023_engine.py`
- SHA-256（源文件与副本相同）：`CEA5D440D554BB1C2DEAE1E372DFBA478EF91698F7B184BFF7238A14D1E1909B`
- 来源版本：A 提供的完整 Python 复现包；该包 README 声明估计引擎与 R 存储结果校验至约 `1e-14`。包内没有单独的 Git 版本号。
- 状态：为避免模块命名冲突而临时置于 `epiestim.nash2023_engine`，通过 `epiestim.engine` 适配层导入；待成员 A 将官方实现合入团队包后替换此临时副本。
- 算法内容：副本未修改；包装层仅重导出必需接口。
