# TSBSE temporal revision: one authorized rerun

Parent: scgr_T_tsbse_20261117, source3f7d413, best epoch6 EMA Dice78.4871396%,HD9516.6019533. Original confirmed T78.9492686%/16.1691071 retained. Parent completed9epochs; no existing process or source modified.

Observed parent: both best FP/FN worse than original T, official empty-pred count44 vs21. Early logged extra/base logit gradient median30.62% epoch1 and21.75% epochs2-3, versus9.52% first-batch audit. Later output gradient cosines positive but weakly aligned; not evidence of opposing parameter gradients or established causation. A single stochastic run cannot isolate causal loss effects.

Hypothesis: early thresholded distances and hard-negative sets based on immature predictions exert disproportionate auxiliary influence. Test only timing. Formula, detached exact EDT, radius, bounded weights, cover and background top-K, common peak0.005 all unchanged. Coefficient0 epochs1-3;0.001666667 epoch4;0.003333333 epoch5;0.005 epoch6 onward. No adaptive validation-driven weight changes.

All parameters train from batch1 with original BCE+Dice. Generic encoder pretraining only, fresh M1/TCSR/GN, no task checkpoints or expert models. No model warm start or frozen stage. Seed2023,512,batch16,AdamW lr8e-5 wd1e-4, original scheduler/AMP/EMA, best-only,max60,patience3,myenv. Early stopping remains active; if it stops before ramp reaches full coefficient, report that fact rather than claim full-strength mechanism was tested.

Implementation: optional epoch-start callback in shared engine defaultsNone; existing runners retain behavior. New runner sets transient weight/epoch attributes on raw and EMA models after any EMA reset, without mutating peak config or state_dict. Checkpoint/inference format remains original T. Logs record effective coefficient, raw/weighted terms, two gradient contributions and same-pass error partitions. Zero coefficient still allows recording raw geometric risk; exact added loss/gradient zero verified. Diagnostic loss computation incurs EDT overhead even during zero-weight epochs; learning objective remains base.

Selection: one full validation per epoch, original Dice-best rule. Compare paired Dice/HD95/precision/sensitivity to originalT and parent; log FP/FN,zero-overlap,empty-pred,far-FP and area groups. No favorable epoch mixing. Development set repeatedly reused, not independent test. Same-seed variation exists; a gain would motivate controls, not establish third core novelty.

Verification: coefficient endpoints and raw/EMA synchronization; exact objective/logit gradient equality to BCE+Dice at zero weight; nonzero weight effect; one actual16x512 forward/backward and discarded optimizer step. No new full checkpoint saved by audit. Best-only training retains historical runs. One run only, no automatic successors or replacement of confirmed mainline.
