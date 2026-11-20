// GTA VI trophy rarity forecast — an earner-count growth forecaster.
//
// For every PlayStation 5 trophy of Grand Theft Auto VI, this program forecasts
// the share of Exophase-tracked players who will have earned it at the case's
// target, as five quantiles (5%, 25%, 50%, 75%, 95%). It reads only the case
// file the grader hands it: the trophy list and the history of reads of the
// game's Exophase trophy page before the checkpoint.
//
// The idea: a trophy's earn rate is a ratio, earners over tracked players, and
// the two halves move for different reasons. The earners only ever grow —
// players do not un-earn a trophy — as more of them play long enough to reach
// it. The tracked players grow too, as fresh players keep syncing profiles,
// and every newcomer arrives with few trophies, so a fast influx dilutes every
// rate at once. Launch night showed this plainly: when the tracked count jumped
// sixfold at one refresh, every rate but the one at 100% fell. A model fitted
// to the rate alone confuses the two and reads dilution as a trophy stalling.
// So this program fits the two counts separately, each as a saturating curve
// in log time since release — a cumulative distribution of the time players
// need to reach that point — and divides them at the target:
//
//     earners_i(t) = A_i / (1 + exp(-(a_i + b_i * ln(1 + t))))
//     tracked(t)   = A_N / (1 + exp(-(a_N + b_N * ln(1 + t))))
//     rate_i(target) = earners_i(target) / tracked(target)
//
// Each curve is fitted by a grid search over its asymptote and slope with the
// intercept solved by least squares, in the logit of the fraction of the
// asymptote reached, with a weak pull of each trophy's eventual earner count
// toward what its tier's ceiling implies. The median is that ratio at the
// target, shrunk a little toward the last observed rate. The spread combines
// how far the fitted ratio missed the observed rates, a term that grows with
// the square root of the horizon, and a term for Exophase's discrete refresh
// jumps. Quantiles are a normal spread on the logit scale, so the band is
// honest about how much a week of play can move a rate. A trophy everyone
// already has is forecast to stay that way, near 100%.
//
// Everything is deterministic: fixed grids, no randomness, no clock reads,
// inputs processed in name order. Node 22, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** The game's release, the zero of log time. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** How many history reads a fit sees at most (evenly spaced in time). */
const MAX_READS = 120;
/** A rate at or above this is treated as saturated (everyone has it). */
const SATURATED = 99.5;
/** The spread, in logit units per square-root day, of a rate's wander. */
const HORIZON_SIGMA = 0.35;
/** The spread, in logit units, one Exophase refresh jump can add. */
const REFRESH_SIGMA = 0.25;
/** How strongly an asymptote is pulled toward its prior (pseudo-observations). */
const PRIOR_WEIGHT = 2.0;
/** How much of the median comes from the fitted ratio rather than the last rate. */
const MODEL_WEIGHT = 0.85;
/** The long-run ceiling a trophy of each tier settles toward, in percent. */
const TIER_CEILING: Record<string, number> = { platinum: 3, gold: 20, silver: 35, bronze: 55 };

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

/** A percent clamped into the range the logit is defined on. */
const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
/** logit of a percent. */
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
/** logit of a fraction in (0, 1). */
const logitFrac = (f: number): number => {
  const c = Math.min(0.9995, Math.max(0.0005, f));
  return Math.log(c / (1 - c));
};
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from the game's release to an ISO timestamp, floored so log time is defined. */
const hoursSinceRelease = (iso: string): number =>
  Math.max(0.05, (Date.parse(iso) - RELEASE_MS) / 3600000);

/** Evenly spaced indices from 0..n-1, at most m of them, first and last always kept. */
function spacedIndices(n: number, m: number): number[] {
  if (n <= m) return Array.from({ length: n }, (_, i) => i);
  const out: number[] = [];
  for (let i = 0; i < m; i++) out.push(Math.floor((i * (n - 1)) / (m - 1)));
  return [...new Set(out)];
}

/** Geometric grid from lo to hi, count points. */
function geoGrid(lo: number, hi: number, count: number): number[] {
  const out: number[] = [];
  for (let k = 0; k < count; k++) out.push(lo * Math.pow(hi / lo, k / (count - 1)));
  return out;
}

/** Linear grid from lo to hi, count points. */
function linGrid(lo: number, hi: number, count: number): number[] {
  const out: number[] = [];
  for (let k = 0; k < count; k++) out.push(lo + ((hi - lo) * k) / (count - 1));
  return out;
}

interface Curve {
  /** The asymptote the count settles toward. */
  asymptote: number;
  /** The intercept of the logit curve. */
  intercept: number;
  /** The slope of the logit curve in log time. */
  slope: number;
}

/**
 * Fit y(t) = A * sigmoid(a + b * x) by grid search over A and b, with a solved
 * by least squares for each pair, in the logit of the fraction of the
 * asymptote reached. priorLogit is the logit of the asymptote's prior as a
 * fraction of the grid's top, pulled with priorWeight pseudo-observations.
 */
function fitCurve(
  xs: number[],
  ys: number[],
  aGrid: number[],
  bGrid: number[],
  priorLogit: number,
  priorWeight: number,
): Curve {
  const n = ys.length;
  const top = aGrid[aGrid.length - 1];
  const score = (A: number, b: number): { sse: number; a: number } => {
    let sumY = 0;
    const yv: number[] = new Array(n);
    for (let j = 0; j < n; j++) {
      yv[j] = logitFrac(ys[j] / A);
      sumY += yv[j] - b * xs[j];
    }
    const a = sumY / n;
    let sse = priorWeight * (logitFrac(A / top) - priorLogit) ** 2;
    for (let j = 0; j < n; j++) {
      const d = yv[j] - a - b * xs[j];
      sse += d * d;
    }
    return { sse, a };
  };
  let best = { sse: Infinity, A: aGrid[0], a: 0, b: 0 };
  for (const A of aGrid) {
    for (const b of bGrid) {
      const { sse, a } = score(A, b);
      if (sse < best.sse) best = { sse, A, a, b };
    }
  }
  // One finer pass around the winner.
  for (const dA of [0.97, 0.99, 1.01, 1.03]) {
    for (const db of [-0.05, -0.02, 0.02, 0.05]) {
      const A = Math.min(top, Math.max(aGrid[0], best.A * dA));
      const b = best.b + db;
      const { sse, a } = score(A, b);
      if (sse < best.sse) best = { sse, A, a, b };
    }
  }
  return { asymptote: best.A, intercept: best.a, slope: best.b };
}

/** Five quantiles for one trophy at the case's target. */
function forecastTrophy(
  t: Trophy,
  history: Read[],
  picks: number[],
  xs: number[],
  tracked: Curve,
  nTarget: number,
  xTarget: number,
  days: number,
): number[] {
  const idx = t.id - 1;
  // The trophy's most recent listed rate, scanning back past any nulls.
  let lastRate = 0;
  for (let j = history.length - 1; j >= 0; j--) {
    const r = history[j].rates[idx];
    if (r !== null && r !== undefined) {
      lastRate = clampPct(r);
      break;
    }
  }
  // A trophy everyone already has stays that way: near 100 with a tight band.
  // Launch night it held 100% while the tracked count grew sixfold.
  if (lastRate >= SATURATED) {
    const m = logit(99.99);
    return Z.map((z) => round4(clampPct(100 * sigmoid(m + z * 0.3))));
  }

  // This trophy's earner counts at the picked reads.
  const eXs: number[] = [];
  const eYs: number[] = [];
  const eNs: number[] = [];
  const eRates: number[] = [];
  for (let k = 0; k < picks.length; k++) {
    const r = history[picks[k]].rates[idx];
    if (r === null || r === undefined) continue;
    const n = history[picks[k]].players_tracked;
    eXs.push(xs[k]);
    eYs.push(Math.max(0.01, clampPct(r)) * n / 100);
    eNs.push(n);
    eRates.push(clampPct(r));
  }
  if (eXs.length === 0) {
    // No read ever listed this trophy: say so with a wide band around 1%.
    const m = logit(1);
    return Z.map((z) => round4(clampPct(100 * sigmoid(m + z * 2.0))));
  }

  // Fit the earner count's curve, its asymptote pulled toward the tier ceiling.
  const maxE = Math.max(...eYs);
  const eTop = maxE * 50;
  const priorEarners = tracked.asymptote * ((TIER_CEILING[t.tier] ?? 50) / 100);
  const earnerCurve = fitCurve(
    eXs,
    eYs,
    geoGrid(Math.max(maxE * 1.02, maxE + 1), eTop, 22),
    linGrid(-0.5, 3.1, 25),
    logitFrac(Math.min(0.999, Math.max(0.001, priorEarners / eTop))),
    PRIOR_WEIGHT,
  );
  const eTarget = earnerCurve.asymptote * sigmoid(earnerCurve.intercept + earnerCurve.slope * xTarget);
  const modelRate = clampPct((eTarget / nTarget) * 100);

  // How far the fitted ratio missed the observed rates, for the band.
  let sq = 0;
  for (let k = 0; k < eXs.length; k++) {
    const pred = (earnerCurve.asymptote * sigmoid(earnerCurve.intercept + earnerCurve.slope * eXs[k])) / eNs[k] * 100;
    const d = logit(eRates[k]) - logit(pred);
    sq += d * d;
  }
  const sigmaFit = Math.min(1.5, Math.max(0.15, Math.sqrt(sq / eXs.length)));

  const medianLogit = MODEL_WEIGHT * logit(modelRate) + (1 - MODEL_WEIGHT) * logit(lastRate);
  const sigma = Math.min(
    2.5,
    Math.sqrt(sigmaFit ** 2 + (HORIZON_SIGMA * Math.sqrt(days)) ** 2 + REFRESH_SIGMA ** 2),
  );
  const qs = Z.map((z) => round4(clampPct(100 * sigmoid(medianLogit + z * sigma))));
  for (let k = 1; k < qs.length; k++) qs[k] = Math.max(qs[k], qs[k - 1]);
  return qs;
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case ${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`case ${c.id} has no history`);
  const xTarget = Math.log(1 + hoursSinceRelease(c.target));
  const days = c.horizon_minutes / 1440;

  // The reads the fits see: evenly spaced in time, first and last always kept.
  const picks = spacedIndices(c.history.length, MAX_READS);
  const xs = picks.map((j) => Math.log(1 + hoursSinceRelease(c.history[j].at)));
  const ns = picks.map((j) => c.history[j].players_tracked);

  // Fit the tracked-player count's curve, then read it at the target.
  const maxN = Math.max(...ns);
  const tracked = fitCurve(
    xs,
    ns,
    geoGrid(maxN * 1.01, maxN * 10, 20),
    linGrid(-0.2, 2.0, 15),
    logitFrac(0.2),
    0.5,
  );
  const nTarget = tracked.asymptote * sigmoid(tracked.intercept + tracked.slope * xTarget);

  const trophies = c.trophies.map((t) => ({
    id: t.id,
    q: forecastTrophy(t, c.history, picks, xs, tracked, nTarget, xTarget, days),
  }));
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
