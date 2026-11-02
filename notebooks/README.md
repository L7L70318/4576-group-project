# 手动运行顺序

01_futures_preparation → 02_auxiliary_preparation → 03_data_checks → 04_overlap_validation。

在 project 根目录安装 `python -m pip install -e ".[dev]"`，选择同一 Python kernel。默认读取缓存；下载开关在01和02。测试断言保留在03和04；没有可选的作者benchmark时，相应比对会明确跳过。计算逻辑位于 src/project，完整CLI为 `python run_all.py`，缓存复跑可使用 `--offline`。

只保留 Table Internet-I 直接惯例这一主版本；输出路径见项目 README。营业日日历和近似日期的限制见方法说明。

GitHub 版本清除执行输出，避免发布 WRDS 数据。探索性验证写入Git忽略的processing_exploration目录；本地运行可重新生成。
