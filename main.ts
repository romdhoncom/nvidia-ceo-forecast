// Forecast of every market's Yes price minute by minute, 4 hours ahead, for the "Next CEO of NVIDIA?" event.
//
// The model is an empirical, level-local random walk in price space, built only from the case's own history:
//
//   1. For a ladder of horizons h (1, 2, 3, 4, 6, 8, ... 240 minutes) it pools the market's own h-minute price
//      changes, p[t + h] - p[t], over two windows: the last WINDOW_RECENT minutes, which carry the regime the
//      market is in now, and its whole history, which carries the tail it is capable of. Every sample is
//      additionally weighted by a Gaussian on how close the price was to the checkpoint price, so a change is
//      learned at the level the market trades at now rather than at a level it left two days ago. Minutes that
//      did not move are samples too - their change is exactly 0 - so long still stretches and rare jumps stay
//      in one distribution, which is what a market that ticks every few minutes and jumps now and then needs.
//   2. The five sample quantiles at each horizon become spreads around the sample median, so the forecast's
//      median is the market's last price, and the spreads are blended (geometrically) between the two windows
//      and interpolated log-log between the ladder's rungs, so each spread keeps its shape as it grows.
//   3. The spreads are shaped by SCALE, floored by FLOOR * sqrt(h), added to the checkpoint price and clamped
//      to [0, 1]; the rows are forced non-decreasing and the median is pinned to the checkpoint price exactly.
//
// Nothing here is random, nothing is fetched, and the work is linear in the case's size: a case of 24 markets
// with 4,320 minutes each takes a fraction of a second.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** How the pooled spreads are shaped: the tail below is widened, the 75% side is pulled in, the median is pinned. */
const SCALE = [1.15, 1.15, 1, 0.2, 0.85];
/** Minutes of recent history pooled into each market's spread estimate: the regime it is in now. */
const WINDOW_RECENT = 360;
/** Weight of the whole-history estimate in the blend, against the recent window's. */
const MIX_HISTORY = 0.35;
/** Width of the Gaussian on price level: samples taken far from the checkpoint price count less. */
const LEVEL_BAND = 0.15;
/** Spread floor in price units per square-root minute, so a dead market still gets a band. */
const FLOOR = 0.0001;
/** The normal's quantiles, used only when a market has too little history to measure a spread from. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** Horizons (minutes) whose spread is measured directly; the rest are interpolated between them. */
const ANCHORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 240];
/** Below this many weighted samples a rung is not measured. */
const MIN_SAMPLES = 8;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  at: string;
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** One pooled price change and the weight it carries. */
interface Sample {
  d: number;
  w: number;
}

/** The weighted sample quantile: the change whose cumulative weight first reaches p of the total. */
function sampleQuantile(sorted: Sample[], totalWeight: number, p: number): number {
  const target = p * totalWeight;
  let acc = 0;
  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i] as Sample;
    acc += s.w;
    if (acc >= target) {
      if (i === 0) return s.d;
      const before = acc - s.w;
      const frac = (target - before) / s.w;
      const prev = sorted[i - 1] as Sample;
      return prev.d + (s.d - prev.d) * frac;
    }
  }
  return (sorted[sorted.length - 1] as Sample).d;
}

/**
 * The pooled h-minute changes of one market over the last `window` minutes (Infinity: all of it), as spreads
 * around the sample median at each rung: spreads.get(h) = [below 5%, below 25%, 0, above 75%, above 95%].
 */
function measureSpreads(prices: number[], window: number): Map<number, number[]> {
  const n = prices.length;
  const checkpoint = prices[n - 1] as number;
  const from = Number.isFinite(window) ? Math.max(0, n - 1 - window) : 0;
  const byHorizon = new Map<number, number[]>();
  for (const h of ANCHORS) {
    const samples: Sample[] = [];
    let total = 0;
    for (let t = from; t + h < n; t++) {
      const u = ((prices[t] as number) - checkpoint) / LEVEL_BAND;
      const w = Math.exp(-u * u);
      if (w < 1e-6) continue;
      samples.push({ d: (prices[t + h] as number) - (prices[t] as number), w });
      total += w;
    }
    if (samples.length < MIN_SAMPLES) continue;
    samples.sort((a, b) => a.d - b.d);
    const q = LEVELS.map((p) => sampleQuantile(samples, total, p));
    const median = q[2] as number;
    byHorizon.set(h, [
      Math.max(0, median - (q[0] as number)),
      Math.max(0, median - (q[1] as number)),
      0,
      Math.max(0, (q[3] as number) - median),
      Math.max(0, (q[4] as number) - median),
    ]);
  }
  return byHorizon;
}

/** One level's spread at horizon h, interpolated (log-log) between the measured rungs, or sqrt-grown past them. */
function spreadAt(rungs: number[][], rungHorizons: number[], h: number, level: number): number {
  const m = rungHorizons.length;
  if (m === 0) return 0;
  const first = rungHorizons[0] as number;
  const last = rungHorizons[m - 1] as number;
  if (m === 1 || h <= first) return (rungs[0] as number[])[level] as number;
  if (h > last) {
    // Past the longest rung the spread keeps widening like a random walk's: sqrt(h / lastRung).
    return ((rungs[m - 1] as number[])[level] as number) * Math.sqrt(h / last);
  }
  let lo = 0;
  let hi = m - 1;
  for (let i = 0; i < m - 1; i++) {
    if ((rungHorizons[i] as number) <= h && h <= (rungHorizons[i + 1] as number)) {
      lo = i;
      hi = i + 1;
      break;
    }
  }
  const a = rungHorizons[lo] as number;
  const b = rungHorizons[hi] as number;
  const va = (rungs[lo] as number[])[level] as number;
  const vb = (rungs[hi] as number[])[level] as number;
  const w = (Math.log(h) - Math.log(a)) / (Math.log(b) - Math.log(a));
  const v = va > 1e-9 && vb > 1e-9 ? Math.exp(Math.log(va) + (Math.log(vb) - Math.log(va)) * w) : va + (vb - va) * w;
  return Math.max(0, v);
}

const round9 = (x: number): number => Math.round(x * 1e9) / 1e9 + 0;

/** The forecast of one market: horizon rows of the five quantiles around its last price. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  const prices = m.prices;
  if (prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = prices[prices.length - 1] as number;

  const recent = measureSpreads(prices, WINDOW_RECENT);
  const history = measureSpreads(prices, Number.POSITIVE_INFINITY);
  const recentHorizons = [...recent.keys()].sort((a, b) => a - b);
  const historyHorizons = [...history.keys()].sort((a, b) => a - b);
  const recentRungs = recentHorizons.map((h) => recent.get(h) as number[]);
  const historyRungs = historyHorizons.map((h) => history.get(h) as number[]);
  const measured = recentHorizons.length > 0 || historyHorizons.length > 0;

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const floor = FLOOR * Math.sqrt(h);
    const row: number[] = [];
    for (let level = 0; level < 5; level++) {
      if (level === 2) {
        row.push(x);
        continue;
      }
      let spread: number;
      if (!measured) {
        spread = Math.abs(Z[level] as number) * floor;
      } else {
        const a = spreadAt(recentRungs, recentHorizons, h, level);
        const b = historyHorizons.length === 0 ? a : spreadAt(historyRungs, historyHorizons, h, level);
        const mixed = a > 1e-9 && b > 1e-9
          ? Math.exp((1 - MIX_HISTORY) * Math.log(a) + MIX_HISTORY * Math.log(b))
          : a + (b - a) * MIX_HISTORY;
        spread = Math.max(mixed * (SCALE[level] as number), floor);
      }
      const q = level < 2 ? x - spread : x + spread;
      row.push(round9(Math.min(1, Math.max(0, q))));
    }
    // The median is the checkpoint price; nothing strays past it on its own side.
    row[2] = x;
    row[1] = Math.min(row[1] as number, x);
    row[3] = Math.max(row[3] as number, x);
    // Non-decreasing, as the format requires.
    for (let level = 1; level < 5; level++) {
      if ((row[level] as number) < (row[level - 1] as number)) row[level] = row[level - 1] as number;
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== LEVELS.length || c.quantiles.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(LEVELS)}`);
  }
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
