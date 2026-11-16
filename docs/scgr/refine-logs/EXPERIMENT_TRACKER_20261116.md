# 执行跟踪

|组|状态|说明|
|---|---|---|
|CPU/GPU预检|通过|14项CPU检查、真实四图导出、三组512/batch16三次成功AMP|
|V|待启动|无文本，无评分损失|
|T|排队|只启用固定文本条件|
|S|排队|T+评分BCE0.1|

启动后以outputs/scgr_postfusion_20261116/STATUS.json和各组progress.json为实时状态；本文件为启动前计划。
