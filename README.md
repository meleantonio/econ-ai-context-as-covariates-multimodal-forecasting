# Forecasting with context as covariates on SWARM-Lab/CoCo-Eval

## Named object
The named empirical object is **SWARM-Lab/CoCo-Eval**, the Hugging Face dataset behind the CoCo paper.

Use the **`fnspid`** split and start with one stock series, for example **`fnspid/ba`** (Boeing). The dataset card shows the first two observations in that series as:

- 2021-12-21: 199.520004
- 2021-12-22: 201.690002

That makes the object concrete enough to audit. You can see the text, the target, and the timestamp, which is exactly what economists need before they talk about forecasting gains.

## Stake
The stake is simple: if you treat text as decoration, you will misstate the next-day move and then tell a neat story about “context” that is numerically false.

In equity work, that false number affects event studies, trading rules, and any claim that a narrative signal adds information beyond the lagged price. In macro work, the same mistake shows up as a bad nowcast or an overconfident policy brief.

## Naive path
The obvious path is carry-forward forecasting: predict tomorrow with today’s price.

On the Boeing slice above, that gives **199.520004** for 2021-12-22, but the realised value is **201.690002**. The wrong number is therefore **2.169998**.

That is the failure mode CoCo is trying to repair: not “more words,” but a better-conditioned forecast object that can absorb text without pretending the lag alone is enough.

## Technique
Read CoCo as a **context-as-covariates** design, not as “LLM magic.” The method is:

1. take a historical target series;
2. align each timestamp with its contemporaneous text;
3. build a supervised one-step-ahead forecasting problem;
4. let a model see both lagged prices and text features;
5. compare the result to persistence.

The tutorial script does exactly that on **SWARM-Lab/CoCo-Eval**:

- it loads `fnspid/ba`;
- constructs lagged numeric features;
- adds a text representation;
- fits a regularised model rather than toy OLS;
- prints the naive carry-forward forecast next to the covariate-aware forecast;
- saves a small diagnostic plot.

That is the RELAI loop in plain English:

- **Examine** the named object row by row;
- **eXplain** why persistence gives the wrong number;
- **Probe** whether text actually helps out of sample;
- **Link to economics** through a forecasting problem economists already know;
- **Output prediction** before running the code;
- **Recreate** with `python tutorial.py`;
- **Extend** by swapping the stock series, the horizon, or the text encoder.

## You will get this wrong
Common first-week mistakes:

- using text that arrives after the forecast cutoff;
- mixing dates across tickers and accidentally creating look-ahead leakage;
- evaluating on the same rows used to fit the text vectoriser;
- forgetting that a text feature can proxy for the market move rather than explain it;
- reading one good backtest as proof of a stable forecasting edge;
- assuming that missing text means missing information rather than a preprocessing choice;
- comparing against a weak baseline and calling the result identification.

## Controversy
Applied researchers disagree on three things here.

First, is text a genuine covariate or just a noisy proxy for latent fundamentals and risk premia? Second, do gains survive strict timestamp discipline, or do they vanish once you enforce publication timing and realistic lag structure? Third, do foundation-model pipelines beat a disciplined linear or tree baseline once you equalise leakage controls and feature engineering effort?

That disagreement is productive. It is also why this tutorial keeps the naive forecast visible instead of hiding it behind prose.

## The technique in one sentence
Use text as an additional covariate in a properly lagged forecasting design, then test whether it improves the one-step prediction relative to persistence on a real dataset.

## What to run
Run `python tutorial.py` after installing the requirements. The script loads the real CoCo-Eval release from Hugging Face, prints the exact carry-forward miss on the first Boeing observation, fits a covariate-aware baseline, and writes a small plot so you can inspect the forecast path.

## What to look for
Do not look for a miracle. Look for a smaller error than persistence, stable code, and a model object you can audit. If the text features do not help on a small slice, that is still a useful result: it tells you the signal is thinner than the paper title suggests.