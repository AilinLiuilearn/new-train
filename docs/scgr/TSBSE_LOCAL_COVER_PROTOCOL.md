# One-factor coverage-scale repair (candidate, not established innovation)

## Parent and decision
Named parent:scgr_T_tsbse_ramp_20261117,commit345c4bc,peak0.005/delay3/ramp3,best11EMA78.631414%/16.214441. Latest common-strength check0.0025 completed11epochs,best8EMA77.866171%/16.991613 and did not reach target. Do not halve coefficients further without new evidence. A single run does not exclude all parameter effects: first3 base-only epochs already differ across runs.

Use preserved better parent0.005, not latest failed half. Exactly one experimental change relative that parent: cover's spatial scale. Peak0.005,delay3/ramp3 and spill are unchanged. No task weight loading from any parent. All task modules fresh, generic encoders pretrained, all visual parameters trainable.

## Exact change
Old cover weight=dP/(dP+rGT),rGT=max(1,sqrt(totalGTarea/pi)).
New cover weight=dP/(dP+max(1,dGTinside)). dGTinside is GT foreground EDT after zero-padding1pixel and cropping back; this defines a finite computational boundary for full/edge-touching masks. It is a local mask-interior distance, not a measured anatomical radius or confidence score. All geometry detached. dP uses current detached predicted foreground(logit>=0); no prediction uses finite image diagonal. Cover remains meanGT(weight*softplus(-logit)).
Spill remains original globalGT-scaled risk wG*softplus(logit),largestK background risk mean,K=min(totalGTarea,Nbg). EmptyGT skips added term,noBG gives zero spill. Both branches averaged, then full-batch mean including skipped zeros. Original BCE+Dice plus common schedule coefficient times that risk. No new loss/head/numeric hyperparameter or inference cost.

Global mode remains default for all existing configs. New config specifies gt_local_thickness. Original T43.8M-parameter model and checkpoint format unchanged; only training risk geometry changes.

## Mechanism evidence
Fixed64 systematic indices across unchanged dev order, FP32 inference of three saved models, no sample/model selection or coefficient tuning. At parent_ramp:near-FN mean weight~0.249->0.722; raw cover output-gradient L1 sum8.683->14.423 (~1.66x). Similar direction at originalT and failed half. LargerGT near-boundary weights especially small under globalradius. This supports testing the repair, not causal efficacy.

This broadly strengthens cover, not only a boundary-only mechanism. Fragment1-4px component gradient fraction decreases, but its absolute gradient increases. No claim of reduced ambiguous-label harm. RawPNG audit finds many tiny/weak-display zero-overlap cases; no clinical visibility or annotation correctness conclusion, no sample exclusions.

## Review and numerical scope
Fresh reviewer requestedgpt-6.1-terra/medium per user route; actual model identity not independently reported. Same-family review, provisional. Review found no blocking math or implementation defect; insisted on exact spill invariance, finite zero-padded/full-mask behavior and AMP checks. Risks:stronger absolute fragment gradients,threshold jumps,artificial outside-image background,possible digital-scale exceptions at deep locations,and insufficient visual information. No claim of new Wasserstein/HD distance or proven third contribution.

Checks: defaultglobal values/gradients match original345c4bc; exact spill values/gradients remain unchanged across correct,missed,all-pred,empty/fullGT,edge,fragmented and threshold-near masks; local one-pixel missed-edge gradient rises; finite difference; output-logit gradient L1 bound; same-parameter/single-config-factor check; actual16x512 fresh peak-strength AMP forward/backward and discardedAdamW step. Formal Scaler/scheduler/EMA unchanged.

## Execution/evidence
Seed2023,512,batch16,lr8e-5,AdamW wd1e-4,original aug/scheduler/EMA/AMP,patience3,max60,myenv,best-only. One full fusion validation each epoch, original threshold and PNG HD95. Log cover/spill contributions,coverage FN weight sums/counts,GT scale,per-scale features/gates/residuals and AMP diagnostics. New output paths/commit fixed at launch, old bests preserved.

Goal remains same Dice-best>=79.5%,pairedHD95<16.169107096291484,requiring checkpoint reload evidence before completion. Compare originalT and named parent,FP/FN,zero-overlap/empty,farFP andGT-area groups. Do not mix epochs or pick thresholds/subsets. Reused dev and stochastic training limit causal/generalization claims. No automatic new design until completed results analyzed; user stop cancels successors.
