# Diagnostic matched-harness originalT control, not target candidate

Four completed added-loss experiments failed the numerical target. First3 auxiliary-off epochs already diverge across runs. RAM replay restored exact fresh initial tensors,RNG and3 materialized training batches. Even base-vs-base repeat has427 parameter-gradient tensors differ at firstbackward(relL2~1.36e-4);logits initially equal, then diverge by step3(max~.00439). Zero-weight TSBS drift is same order(relL2~1.74e-4;step3max~.00586). Original AMP65536 updates finite. This is evidence of backward nondeterminism,not a TSBSE-specific zero-weight defect,and does not quantify final Dice variance.

Warning-only separate probe identifies memory-efficient attention,adaptive-average-pooling andreflection-padding backward nondeterminism. Original runtime is retained. No custom deterministic backwards or additional optimization machinery justified by this evidence.

Four fixed materialized training batches at saved ramp/local checkpoints show cover-vs-spill parameter opposition, but combined auxiliary gradients generally align with base in fusion/refiner/decoder body. Head-weight conflicts exist and exceed repeated-backward noise; head-bias opposition is trivial foreground/background competition. Raw gradient conflict is not evidence of harmful AdamW updates or a validation failure cause. No gradient projection or selective head routing is introduced.

Named parent ramp0.005, same globalcover. Only experimental factor: existing common coefficient becomes0. All epochs optimize originalBCE+Dice. RawTSBSE risks and zero-gradient diagnostics remain in the current harness. No new parameters/losses/thresholds/seeds. Architecture originalT M1+textTCSR+GN unchanged. Fresh task modules,generic encoder pretraining,myenv,seed2023,512,b16,lr8e-5,AdamW/scheduler/EMA/AMP unchanged,patience3,max60,best-only,one full validation each epoch. OriginalT old mainline and all trial weights/logs retained.

Purpose: current-harness reference before another loss revision. One control still does not estimate a full distribution of run variance or establish causality; matched comparisons must remain qualified. Local-cover best78.1491%/16.9768 does not beat originalT78.9493%/16.1691.

This run explicitly has eligible_for_goal=False. Its score cannot satisfy the third-loss goal even if Dice>=79.5% andHD95<16.1691071. No third loss is active. Goal remains active; candidate goal completion requires an active cover/spill loss and saved same-best checkpoint evidence. No change to success threshold, no automatic replacement of original confirmed mainline, no subsequent loss formula preselected.

Diagnostics and reviews stored under analysis/tsbse_replay_20261118,analysis/tsbse_parameter_gradients_20261118,analysis/tsbse_completed_trials_review_20261118. Independent result review requestedgpt-6.1-sol/high,actual identity unreported,same-family/provisional. No efficacy or innovation claim.
