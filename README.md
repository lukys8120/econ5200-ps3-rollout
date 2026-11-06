# The Rollout That Looked Like It Worked: Staggered Difference-in-Differences

**ECON 5200 · Causal Machine Learning & Applied Analytics · Problem Set 3**

## Objective

A state workforce agency rolled out a training programme county by county over four years and reported a positive, significant effect from a two-way fixed effects (TWFE) regression. As an independent reviewer, I tested whether that number means what the agency says it means.

## Methodology

- Re-ran Lab 11's Card and Krueger (1994) replication and recovered the same two-by-two DiD as an interaction coefficient in a long-form regression.
- Simulated the agency's panel: 40 counties in 8 regional boards, adoption at the board level between years 3 and 6, one board that never adopts, 12 years, an effect that grows with the square of years of exposure, and a board-by-year shock. The true effect is recorded for every treated county-year.
- Ran the agency's TWFE regression and decomposed it into its two-by-two comparisons and their weights (Goodman-Bacon).
- Re-estimated the effect one cohort-year cell at a time, using only never-treated and not-yet-treated counties as controls, and aggregated the cells.
- Built an event study around adoption with the year before adoption (e = −1) as the omitted period.
- Compared default, county-clustered and board-clustered standard errors.
- Had an AI implement the wild cluster bootstrap with all 256 Rademacher weight vectors, and used an AI as a hostile second reviewer of my parallel-trends paragraph.

## Key Findings

**The replication carries over.** The two-by-two DiD is +2.75 FTE, and the regression's interaction coefficient is 2.754.

**TWFE badly understates the effect.** The true average effect on the treated is 14.32, but TWFE gives 2.43, a bias of −11.89.

**Almost half of TWFE's weight sits on forbidden comparisons.** The weighted sum of the two-by-twos equals TWFE exactly (2.429). Comparisons that use already-treated counties as controls carry 48.5% of the weight and are all negative. The largest, cohort 6 against cohort 3, carries 24.4% of the weight and estimates −9.32. As a result, TWFE lies below every true effect in the panel (3.0 to 43.5).

**The clean estimator recovers the truth.** Using only not-yet-treated and never-treated controls gives 15.55. On the noise-free outcome it gives 14.32, exactly the true effect, so the remaining gap is noise.

**The agency's significance comes from the wrong standard errors.** The estimate is 2.43 in every version. The p-value is 0.017 with default SEs, 0.307 clustered by county and 0.673 clustered by board, the level at which treatment was assigned. The wild cluster bootstrap over all 256 weight vectors gives p = 0.977, so the agency's conclusion does not survive.

## Repository Contents

| File | Description |
|---|---|
| `Econ_5200_PS3.ipynb` | Full problem set notebook with rendered output |
| `src/did_toolkit.py` | Extends Lab 11's `did_utils.py`: two-by-two DiD, event study, clean ATT, a clustered-inference wrapper and the wild cluster bootstrap |
| `src/did_utils.py` | Lab 11's DiD helpers, used for the cohort-year cells in Step 3.1 |
| `ai-appendix.md` | AI prompts and replies for Tasks 4.1 and 4.3 |

## Data

- Card and Krueger (1994) NJ/PA fast-food survey, downloaded from David Card's archive (`njmin.zip`).
- The staggered panel is simulated with a fixed seed (`np.random.default_rng(42)`).
