# TSBSE common-strength check after completed ramp run

Parent:scgr_T_tsbse_ramp_20261117, commit345c4bc, completed14epochs. Best epoch11EMA Dice78.6314135%,HD9516.2144415. Relative original T, FP575034->497968(-13.40%),FN782334->849150(+8.54%),empty predictions21->84,precision81.5717->83.2701%,sensitivity76.4902->74.4823%. Smaller GT group pooled Dice improved but medium/large groups worsened. This supports testing conservative-prediction tradeoff; does not establish a causal defect from one stochastic run.

Early-strength issue largely controlled: after delay/ramp, logged aux/base output-gradient median around1.8-4.5% in later epochs, both branches similar L2. These are snapshots of logit gradients, not parameter gradients or evidence that nominally equal branches exert equal task effects. Whole-slice normalization, hard-negative tail and thresholded coverage remain potential structural limitations.

Only experimental factor changed: common peak coefficient0.005 ->0.0025. Delay3/ramp3 unchanged, so epochs1-3 coefficient0;epoch4 .000833333;epoch5 .001666667;epoch6 onward .0025. Same loss numerator/denominator, detached EDT,weights,K,all training/eval settings,seed and original T architecture. No new loss, head, branch-specific coefficient or expanded sweep. Training from fresh task modules and generic encoder pretraining, not parent checkpoint. One full validation per epoch,myenv,best-only,patience3,max60.

Decision: inspect both same-Dice-best metrics and changes in FN/FP/empty/zero-overlap/far-FP/GT-area groups, and raw/weighted gradients. If reduced strength only returns toward base without target gains, do not continue halving blindly. Next change, if justified, should repair an identified cover/spill mechanism rather than add losses or parameters. No causal assertion from just a better random trajectory.

Goal requires same Dice-best Dice>=79.5% andHD95<16.169107096291484 plus checkpoint reload verification. Numerical dev gain is neither independent-test evidence nor third-innovation proof. Original T and all prior best/logs retained; source immutable per run. No auto replacement of mainline. User stop cancels successors.
