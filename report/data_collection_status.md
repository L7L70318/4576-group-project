# 数据准备状态（整理后）

唯一主版本：vendor逐合约FND/LTD优先，缺失时回退Table Internet-I通知日惯例。原始行情不改动，旧convention-only版本仅在Carry专项审计中重建用于before/after比较。

- WRDS期货原始记录5,017,880条、30个contrcode；CFTC275,975条；国家股指201,306条。
- 主月度面板15,076条，Barley核实纳入后有效收益14,937条；Proposal子样本另存。
- 论文同品种同月有效配对10,373条、27品种、1973-02至2011-09；MAE 30.5459bp。与整理前最后采用版本一致。
- 四个主notebook均逐格执行成功；同合约收益、日期边界、动量缺月和CFTC延迟发布测试通过。
- 总体/逐品种基础统计及逐笔误差只保留一套，见results/replication/overlap_validation。

仍未解决：原Propane不可得；联邦假日代理不是完整交易所日历；缺到期日使用交割月末假设；CFTC实际公布日未完整，不生成预测版；G20为代表性指数proxy，FRED为修订后数据。Barley已核实为WRDS 3160 WESTERN BARLEY、CAD。严格C5为318月/10.06%，rank诊断为447月/13.42%；两者都不指定为正式复刻。差额主要来自long leg，详见carry_replication_audit.md。

手动运行notebooks 01→02→03→04。或安装项目依赖后执行 `python run_all.py --data-only --offline`。新环境无需临时依赖目录，使用pyproject.toml安装。凭据继续保存在外部pgpass。

清理路径及删除体积记录在cleanup_manifest.json；父目录论文/Proposal和project/.git未动。

清理例外：旧 results/replication/tables/strict 下 author_summary.csv、wrds_original_sample_summary.csv、wrds_proposal_sample_summary.csv 被其他进程占用，暂不能删除；主流程不再读取它们。其余清单项已删除，约释放162.4MB。未强制关闭占用程序。
