# 权重保存策略：仅保留最佳模型

按用户要求，后续新训练默认 `save_last_checkpoint=False`：
只在原选模分数严格改善时原子更新 `best.pt`，不生成 `last.pt`。
不改原选模规则、模型、优化器更新、EMA/AMP、学习率或早停。
`best.pt`的原有格式与raw/EMA选择语义保留，可照常评价及离线建图。

已有历史run的last.pt不批量删除。当前v2正在使用旧保存逻辑运行，切换时在epoch边界
读取完整last.pt恢复训练状态，验证继续训练后移除这份过渡last.pt；旧源baseline不动。

没有last.pt后，中断不能从最后一轮完整续训。best.pt不是最后一轮的恢复断点，
不会把它伪装成last.pt继续同一次实验。若未来明确需要恢复断点，可显式
`--save_last_checkpoint`；恢复兼容性测试使用此选项，正式新实验默认关闭。

```bash
# 默认仅best.pt
bash scripts/train_joint_aug.sh MAP_DIR NEW_RUN_DIR --crop_guidance dual_state

# 历史完整断点只用于本次恢复，后续不再写last.pt
python -u run_staged.py --resume EXISTING_RUN/last.pt --no-save_last_checkpoint --device cuda

# 评价最佳权重
python -u run_staged.py --evaluate RUN_DIR/best.pt --device cuda
```

队列支持 `--continue_existing` 接续状态为paused的同一套实验；会校验原run路径、
顺序、地图身份和采样方式，当前run从last.pt恢复，后续run按best-only新建。
这是存储策略变更，不能借此修改科学实验参数。
