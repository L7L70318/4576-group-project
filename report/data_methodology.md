# 数据方法：唯一主版本

## 范围与来源

期货请求1969–2025，原文1970-01至2011-09，Proposal2000–2025；实际历史一般更短。WRDS tr_ds_fut.wrds_fut_contract + wrds_contract_info。按configs/commodity_map.csv的contrcode映射，不按可能重码的ticker。BO/RR按固定priority选择系列。WRDS contrcode 3160 明确标为 WESTERN BARLEY、CAD、metric tonne，且覆盖作者1994-07至2003-05样本，故纳入有效收益。原Propane作者样本为1987-09至2006-10；WRDS可识别候选均为2007年后OPIS/Argus掉期或更晚电子合约，不作替代。旧Lumber不接2022新版，HU/RB分开，HO/ULSD规格差异保留。

## Table Internet-I 日期惯例

逐合约vendor firstnoticedate优先；Feeder cattle和Lean hogs等现金结算合约使用vendor lasttrddate。只有相应实际日期缺失时才回退Table Internet-I：谷物、油籽和金属等按前月最后营业日；可可/咖啡分别为交割月首个营业日前10/7个营业日；棉花/木材取15号，原油取前月25号；橙汁/糖取首个营业日。表中前两营业日取第1营业日，4–10日取4日，范围另一端保存为convention_range_end。notice_source和notice_is_exact逐笔披露；旧convention-only政策仅用于before/after审计。

Feeder cattle按交割月最后星期四，Lean hogs按10日作为最后交易日惯例。营业日使用美国联邦假日代理，不是完整历史CME/ICE/加拿大交易所日历；表中usually/around无法唯一恢复真实FND。约某日直接取该日，不额外前移安全缓冲。

有效截止界限优先vendor expirationdate，其次lasttrddate；都缺失时用交割月末并明确标记为独立假设。现金结算品种使用表中推导的最后交易日。

## 报价、选约与收益

研究字段完全相同报价去重；同futcode/日期冲突组整组排除，审计计数。非正/缺失结算价留raw，不用于月收益或log曲线。零成交/连续相同价格不自动判错。

每系列每月共同最后报价日，距自然月末最多7天；不跨日期拼曲线。按交割月份排仍有效合约，从第二近开始找FND及有效截止界限晚于下月末者。若第二近不安全则向后选，是当前实现的边界补充，论文例子未明确覆盖此情形。

同一futcode在t和t+1月末价格比减一，得到超额收益，不再次减无风险利率。退出价缺失不换约不回填。date为收益月，formation_date为进入月；slope为形成月原始前两近报价log比；动量为此前完整六个日历月累计收益，缺月不压缩。

## 组合

按形成时可知状态选择成员。C1–C5分别要求负/正slope两侧足k，平坦曲线不进carry；P1/P2分负曲线、P3/P4分正曲线，M1–M5按动量两端，Q1–Q5五分位，另存AVG和四类Proposal行业组合。保存权重；未来退出收益缺失时不事后重新加权。CARRY不足一侧的处理和原文信号时点仍未完全匹配，不宣称定价表复现。

## 辅助数据

CFTC官方Legacy futures-only年度快照1986–2025；全市场原始记录保留，configs/cftc_map.csv映射。净非商业持仓=(long-short)/OI，Proposal非商业占比=(long+short)/(2OI)，Working T和hedging pressure分别按附录计算，不混称。报告日月度版只用于描述；完整官方release_date缺失，预测版不生成，不能用周二+3天补历史。positioning.align_releases要求完整日历，按实际公布日as-of。

Cboe VIX保存原始日值和月末观察日，不全样本标准化。WRDS 19国代表性本币指数按日绝对收益跨国平均再月平均生成G20 proxy，另存等权日收益平方和开方的Proposal proxy；并非作者精确指数集合。DAX含分红、国家覆盖不齐、俄罗斯停牌已披露，超过7日断点不计算收益，不填停牌价。

FRED USREC/INDPRO/DTWEXBGS/TEDRATE/TB3MS/BAA/AAA是修订后快照，月末取最后非缺失观察不前向填充。USREC仅作事后周期分类，INDPRO仅美国，不能替代G7工业生产或实时数据。

官方来源：https://publicreporting.cftc.gov/w/6dca-aqww/default 、https://www.cboe.com/tradable_products/vix/vix_historical_data 、https://fred.stlouisfed.org/ 。

## 比对与统计

作者.xls实际为TSV，12,207行、29品种、501月、两处slope缺失，仅作benchmark。按同品种同月且两边收益非缺失配对，保存基点偏差/MAE/RMSE、逐行价格和合约、极端误差、日期±1月诊断。总体与逐品种均值/标准差/最小最大/分位数使用完全相同样本，百分比月收益未年化，合并不是组合收益。

Internet-I核对覆盖，Internet-II 29行目标在configs/internet_II_targets.csv。作者N、年化均值/SD/SR匹配原表舍入；contango频率须把零计入(>=0)才能匹配，策略仍用严格正负。附件无OI和原始日价/合约ID，不能直接验证这些字段。Internet-III/GMM等属于后续估计，未实现。

## 重跑和存储

四个notebook依次运行，测试在03/04；可复用逻辑在src，CLI `python run_all.py --data-only --offline`。raw含来源与SHA256 sidecar，data/interim/processed和results均可重建，Git忽略数据和结果。凭据只从外部pgpass读取。旧严格日期/月初下界/范围末端版本及探查缓存已清理，清理清单见cleanup_manifest.json。
