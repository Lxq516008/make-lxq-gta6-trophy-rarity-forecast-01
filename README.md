# make-lxq-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lxq516008/make-lxq-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Lxq516008/make-lxq-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A GTA VI trophy rarity forecaster for make.fun: for every PlayStation 5 trophy of Grand Theft Auto VI, it forecasts the share of Exophase-tracked players who will have earned it one week after launch, as five quantiles, from the house's recording of the game's trophy page.

## How it works

A trophy's earn rate is a ratio — earners over tracked players — and the two halves move for different reasons. The earners only ever grow, as more players play long enough to reach it. The tracked players grow too, as fresh players keep syncing profiles, and every newcomer arrives with few trophies, so a fast influx dilutes every rate at once. Launch night showed this plainly: when the tracked count jumped sixfold at one refresh, every rate but the one at 100% fell. A model fitted to the rate alone confuses the two and reads dilution as a trophy stalling. So this program fits the two counts separately, each as a saturating curve in log time since release — a cumulative distribution of the time players need to reach that point — and divides them at the target:

    earners_i(t) = A_i / (1 + exp(-(a_i + b_i * ln(1 + t))))
    tracked(t)   = A_N / (1 + exp(-(a_N + b_N * ln(1 + t))))
    rate_i(target) = earners_i(target) / tracked(target)

Each curve is fitted by a grid search over its asymptote and slope with the intercept solved by least squares, in the logit of the fraction of the asymptote reached, with a weak pull of each trophy's eventual earner count toward what its tier's ceiling implies. The median is that ratio at the target, shrunk a little toward the last observed rate. The spread combines how far the fitted ratio missed the observed rates, a term growing with the square root of the horizon, and a term for Exophase's discrete refresh jumps. Quantiles are a normal spread on the logit scale, so the band is honest about how much a week of play can move a rate. A trophy everyone already has is forecast to stay that way, near 100%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

This reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Node 22, standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=a0pz932n&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
