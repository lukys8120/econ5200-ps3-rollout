# AI Appendix: Phase 4

*Model and date:*
Claude Opus 5.5 (High), 2026-11-05

## Task 4.1: Wild cluster bootstrap

*Version 1 prompt (as given, sent once as written):*

[Prep] Act as an applied microeconometrician who has implemented
cluster-robust inference for panels with very few clusters.

[Request] Implement the wild cluster bootstrap exactly to the
specification below, for a county-year panel with treatment assigned
at the level of eight regional boards. Impose the null, apply
Rademacher weights at the cluster level, enumerate all 256 weight
vectors rather than resampling, and return the bootstrap p-value for
the treatment coefficient.
<paste the six numbered steps here>

[Iterate] numpy, pandas and statsmodels only. Reuse my panel's column
names. The function must accept the cluster variable as an argument.

[Mechanism Check] Explain why the weights are applied to residuals at
the CLUSTER level rather than the observation level, why imposing the
null matters, and what changes when the number of clusters drops
below roughly ten.

[Evaluate] Run it on my panel and compare the p-value against the
naive cluster-robust one. Tell me which conclusion changes.

*Reply (raw, pasted):*

Three gaps first:

The six steps weren’t pasted (placeholder still in your prompt). I built it to the standard WCR spec (Cameron, Gelbach & Miller 2008; Roodman et al. 2019). Send the steps and I’ll align any deviations.
No panel was uploaded, so I couldn’t run the evaluation or reuse your column names. Column names are arguments instead.
statsmodels isn’t installable in my sandbox. I verified the numpy core on a synthetic 8-board panel (it runs, p_boot = 0.2656). The statsmodels calls are standard but untested here.

Usage (wcb.py):
wild_cluster_bootstrap(df, y="outcome", treat="treated", cluster="board", fe=("county","year"), controls=(...))
It returns the coefficient, t, p_boot (from 256 enumerated vectors) and p_naive (CR1, t with G−1 df).

Mechanism check

Why cluster-level weights: Errors are correlated within boards. Flipping each board’s whole residual vector by one ±1 draw preserves that within-cluster correlation in every bootstrap sample. Observation-level weights would destroy it and reproduce the iid SE you’re trying to avoid.

Why impose the null: Residuals from the restricted model mean the bootstrap data are generated under H0, so the t* distribution is the correct null reference. Unrestricted residuals give overdispersed t* and weaker size control, which is worst when G is small.

Below ~10 clusters: The CRVE is downward-biased, so naive t-tests over-reject, often badly. This is especially true when only a few clusters are treated (here, likely 2–3 of 8). The bootstrap also becomes coarse. With 8 clusters, Rademacher gives only 128 distinct |t*| values, so the p-value takes values in steps of 1/128, and p < 0.01 is nearly unattainable. Webb’s 6-point weights (6⁸ vectors) fix the granularity. If very few clusters are treated, even WCR can be unreliable. In that case, consider randomization inference or the MacKinnon–Webb subcluster variants.

Evaluate: Upload the panel and tell me the y, treatment and FE columns, and I’ll run it and compare. Typically, a naive p below 0.05 rises above 0.05 under WCR, so “significant” flips to “not significant.”

*Changes I made to the code, one line each:*

- Called the function on `panel` with `cluster="board"` and `fe=("county", "year")`, so it runs the agency's TWFE regression, and added the print lines for both p-values.

## Task 4.3: Hostile read

*Prompt (with my §3.2 paragraph pasted in):*

Before the treatment takes place we can see that the true effect lies flat at 0 until the treatment period begins. There is some noise in the study itself that creates the study estimate to be under 0 before the treatment begins, the coefficients before treatement are: `-1.58 -2.17 -0.46 -1.29  0.0`. The assumption built in is that both of the groups would be moving parallel and coeafficient would stay around 0 if there was no treatment taking place. The assumption in this case does hodl though since the implementation wasn't taking place due to some grander economic changes as each county implemented the treatment at different points and time, and also in these counties nothing else has changed, additionally the effect that happen each year are random and affect each county in the same manner. As such, the pre-trends are consistant with the design itself, but they are not the evidance of the actual design.
[Prep] Act as the state agency's lead econometrician. You believe the
programme works and you are reading an external review that says your
TWFE estimate is uninformative.

[Request] Here is the reviewer's paragraph on parallel trends and
identification: <paste your §3.2 paragraph>. Give the three strongest
objections you would raise in a written response.

[Evaluate] Rank them by how much damage each does to the reviewer's
conclusion.

*Reply (raw, pasted):*

In character: the agency’s response

1. The reviewer’s own pre-trends undermine the reviewer’s alternative (most damaging).
Every pre-adoption coefficient is negative (−1.58, −2.17, −0.46, −1.29), and the reviewer waves them away as “noise” without reporting standard errors or a test. Either they are noise, in which case the event study is too imprecise to support confident claims about anything, including the reviewer’s preferred estimate; or they reflect adopting counties catching up before adoption, in which case the reviewer’s “clean” estimate is biased upward too. The reviewer can’t use the event study to discredit us while ignoring what it shows about their own estimate.

2. The reviewer concedes identification.
The paragraph states the parallel trends assumption “does hold”. Our estimate rests on the same assumption. If it holds, the dispute is about how comparisons are weighted, not whether the data identify a programme effect, so calling our estimate “uninformative” overstates the case. It is positive, as is every true effect the reviewer assumes.

3. The reviewer misdescribes the rollout.
The paragraph says yearly shocks “affect each county in the same manner” and that “each county” adopted at different times. Elsewhere the review relies on board-level shocks to reject our standard errors, and adoption was decided by board. A review that is inconsistent about the basic structure of the data shouldn’t be taken at face value.