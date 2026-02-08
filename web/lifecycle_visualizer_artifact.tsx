// Lifecycle Path Visualizer - Claude Artifact
// Interactive visualization for lifecycle investment strategy
// Copy this entire file into a Claude artifact (React type)

import React, { useState, useMemo, useEffect } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine, Cell
} from 'recharts';

// =============================================================================
// Types
// =============================================================================

/**
 * Parameters for the economic environment (VAR structure).
 * Matches Python EconomicParams from core/params.py exactly.
 */
interface EconomicParams {
  rBar: number;           // Long-run mean real rate (Python: r_bar = 0.02)
  phi: number;            // Interest rate persistence (1.0 = random walk)
  sigmaR: number;         // Rate shock volatility (0.3 pp = 0.003)
  muExcess: number;       // Equity risk premium (stock excess return)
  bondSharpe: number;     // Bond Sharpe ratio (replaces fixed mu_bond)
  sigmaS: number;         // Stock return volatility
  rho: number;            // Correlation between rate and stock shocks
  bondDuration: number;   // Duration for HC decomposition and MV optimization
  maxDuration?: number;   // Cap on computed durations (undefined = no cap)
}

/**
 * Default economic parameters matching Python EconomicParams defaults exactly.
 *
 * TEST FIXTURE: Use as the standard baseline for unit tests and simulation verification.
 * All values are economically reasonable and match the Python implementation in core/params.py.
 */
const DEFAULT_ECON_PARAMS: EconomicParams = {
  rBar: 0.02,             // Long-run mean real rate
  phi: 1.0,               // Interest rate persistence (random walk)
  sigmaR: 0.003,          // Rate shock volatility (0.3 pp)
  muExcess: 0.045,        // Equity risk premium (4.5 pp)
  bondSharpe: 0.0,        // Bond Sharpe ratio (no term premium)
  sigmaS: 0.18,           // Stock return volatility
  rho: 0.0,               // Correlation between rate and stock shocks
  bondDuration: 20.0,     // Duration for HC decomposition
  maxDuration: undefined, // No cap on computed durations
};

/**
 * Compute bond excess return from Sharpe ratio.
 * mu_bond = bond_sharpe * bond_duration * sigma_r
 */
function computeMuBondFromEcon(econ: EconomicParams): number {
  return muBond(econ.bondSharpe, econ.bondDuration, econ.sigmaR);
}

/**
 * Parameters for lifecycle model.
 * Matches Python LifecycleParams from core/params.py exactly.
 */
interface LifecycleParams {
  // Age parameters
  startAge: number;             // Age at career start (Python: start_age = 25)
  retirementAge: number;        // Age at retirement (Python: retirement_age = 65)
  endAge: number;               // Planning horizon (Python: end_age = 95)

  // Income parameters (in $000s for cleaner numbers)
  initialEarnings: number;      // Starting annual earnings ($200k)
  earningsGrowth: number;       // Real earnings growth rate (flat = 0.0)
  earningsHumpAge: number;      // Age at peak earnings (at retirement = flat)
  earningsDecline: number;      // Decline rate after peak

  // Expense parameters (subsistence/baseline)
  baseExpenses: number;         // Base annual subsistence expenses ($100k)
  expenseGrowth: number;        // Real expense growth rate (flat = 0.0)
  retirementExpenses: number;   // Retirement subsistence expenses ($100k)

  // Consumption parameters
  consumptionShare: number;     // Share of net worth consumed above subsistence
  consumptionBoost: number;     // Boost above median return for consumption rate

  // Asset allocation parameters
  stockBetaHumanCapital: number; // Beta of human capital to stocks

  // Mean-variance optimization parameters
  // If gamma > 0, target allocations are derived from MV optimization
  // If gamma = 0, use the fixed target allocations below
  gamma: number;                    // Risk aversion coefficient for MV optimization
  targetStockAllocation: number;    // Target stock allocation (used if gamma=0)
  targetBondAllocation: number;     // Target bond allocation (used if gamma=0)

  // Portfolio constraint parameters
  maxLeverage: number;          // Maximum leverage ratio (1.0 = no borrowing, Infinity = unconstrained)

  // Economic parameters (consistent with EconomicParams for DGP)
  riskFreeRate: number;         // Long-run real risk-free rate
  equityPremium: number;        // Equity risk premium

  // Initial financial wealth (can be negative for student loans)
  initialWealth: number;        // Starting financial wealth ($100k, negative allowed)
}

/**
 * Default lifecycle parameters matching Python LifecycleParams defaults exactly.
 *
 * TEST FIXTURE: Use as the standard baseline for unit tests and simulation verification.
 * All values are economically reasonable and match the Python implementation in core/params.py.
 */
const DEFAULT_LIFECYCLE_PARAMS: LifecycleParams = {
  // Age parameters
  startAge: 25,
  retirementAge: 65,
  endAge: 95,

  // Income parameters
  initialEarnings: 200,         // $200k starting earnings
  earningsGrowth: 0.0,          // Flat earnings
  earningsHumpAge: 65,          // Peak at retirement = flat
  earningsDecline: 0.0,         // No decline

  // Expense parameters
  baseExpenses: 100,            // $100k base expenses
  expenseGrowth: 0.0,           // Flat expenses
  retirementExpenses: 100,      // $100k retirement expenses

  // Consumption parameters
  consumptionShare: 0.05,       // 5% of net worth consumed above subsistence
  consumptionBoost: 0.0,        // No boost

  // Asset allocation parameters
  stockBetaHumanCapital: 0.0,   // Bond-like human capital

  // Mean-variance optimization parameters
  gamma: 2.0,                   // Risk aversion coefficient
  targetStockAllocation: 0.60,  // 60% stocks if gamma=0
  targetBondAllocation: 0.30,   // 30% bonds if gamma=0

  // Portfolio constraint parameters
  maxLeverage: 1.0,             // No leverage allowed

  // Economic parameters
  riskFreeRate: 0.02,           // 2% real risk-free rate
  equityPremium: 0.045,         // 4.5% equity premium (matches DEFAULT_ECON_PARAMS.muExcess)

  // Initial wealth
  initialWealth: 100,           // $100k starting wealth
};

// =============================================================================
// Test Fixture Variants
// =============================================================================

/**
 * TEST FIXTURE: Aggressive economic environment for edge case testing.
 * Higher volatility and equity premium to stress-test portfolio strategies.
 */
const AGGRESSIVE_ECON_PARAMS: EconomicParams = {
  rBar: 0.02,             // Same mean rate
  phi: 1.0,               // Random walk persistence
  sigmaR: 0.006,          // 2x rate volatility (0.6 pp vs 0.3 pp)
  muExcess: 0.06,         // Higher equity premium (6% vs 4%)
  bondSharpe: 0.0,        // Same bond Sharpe (no term premium)
  sigmaS: 0.25,           // Higher stock volatility (25% vs 18%)
  rho: -0.2,              // Negative stock-rate correlation (flight to quality)
  bondDuration: 20.0,     // Same duration
  maxDuration: undefined, // No cap on computed durations
};

/**
 * TEST FIXTURE: Conservative lifecycle parameters for edge case testing.
 * Higher risk aversion (gamma=5) and lower consumption share for cautious investors.
 */
const CONSERVATIVE_LIFECYCLE_PARAMS: LifecycleParams = {
  // Age parameters (same as default)
  startAge: 25,
  retirementAge: 65,
  endAge: 95,

  // Income parameters (same as default)
  initialEarnings: 200,
  earningsGrowth: 0.0,
  earningsHumpAge: 65,
  earningsDecline: 0.0,

  // Expense parameters (same as default)
  baseExpenses: 100,
  expenseGrowth: 0.0,
  retirementExpenses: 100,

  // Consumption parameters (more cautious)
  consumptionShare: 0.03,       // Lower consumption share (3% vs 5%)
  consumptionBoost: 0.0,

  // Asset allocation parameters
  stockBetaHumanCapital: 0.0,   // Bond-like human capital

  // Mean-variance optimization parameters (more risk averse)
  gamma: 5.0,                   // Higher risk aversion (5 vs 2)
  targetStockAllocation: 0.40,  // Lower stock target (40% vs 60%)
  targetBondAllocation: 0.50,   // Higher bond target (50% vs 30%)

  // Portfolio constraint parameters
  maxLeverage: 1.0,

  // Economic parameters
  riskFreeRate: 0.02,
  equityPremium: 0.045,

  // Initial wealth
  initialWealth: 100,
};

/**
 * TEST FIXTURE: High-beta human capital lifecycle parameters.
 * For testing scenarios where human capital is stock-like (e.g., tech entrepreneurs).
 */
const HIGH_BETA_LIFECYCLE_PARAMS: LifecycleParams = {
  ...DEFAULT_LIFECYCLE_PARAMS,
  stockBetaHumanCapital: 0.6,   // Stock-like human capital (beta = 0.6)
  gamma: 3.0,                   // Moderately risk averse
  maxLeverage: Infinity,        // Unconstrained leverage for aggressive hedging
};

/**
 * State available to strategy at each time step.
 * Matches Python SimulationState from core/params.py exactly.
 *
 * This provides all the information a strategy needs to make decisions,
 * including wealth measures, cash flows, market conditions, and precomputed
 * hedge components for LDI-style strategies.
 */
interface SimulationState {
  // Time indices
  t: number;                    // Current period (0-indexed)
  age: number;                  // Current age (Python: age)
  year: number;                 // Current year (derived from age - startAge)
  isWorking: boolean;           // True if before retirement (Python: is_working)

  // Market state
  currentRate: number;          // Current interest rate (Python: current_rate)

  // Human capital and expense liability
  humanCapital: number;         // PV of future earnings (Python: human_capital)
  pvExpenses: number;           // PV of future expenses (Python: pv_expenses)

  // Durations
  durationHc: number;           // Duration of human capital (Python: duration_hc)
  durationExp: number;          // Duration of expense liability (Python: duration_expenses)

  // Wealth measures
  financialWealth: number;      // Current FW (Python: financial_wealth)
  netWorth: number;             // HC + FW - PV(expenses) (Python: net_worth)

  // Cash flows
  earnings: number;             // Current period earnings (Python: earnings)
  expenses: number;             // Current period expenses/subsistence (Python: expenses)

  // HC decomposition (for LDI hedge) - Python: hc_stock_component, hc_bond_component, hc_cash_component
  hcStockComponent: number;     // Stock-like portion of human capital
  hcBondComponent: number;      // Bond-like portion of human capital
  hcCashComponent: number;      // Cash-like portion of human capital

  // Expense liability decomposition (for LDI hedge) - Python: exp_bond_component, exp_cash_component
  expBondComponent: number;     // Bond-like portion of expense liability
  expCashComponent: number;     // Cash-like portion of expense liability

  // Target allocations from MV optimization
  targetStock: number;          // Target stock allocation (Python: target_stock)
  targetBond: number;           // Target bond allocation (Python: target_bond)
  targetCash: number;           // Target cash allocation (Python: target_cash)

  // Parameters (read-only reference)
  params: LifecycleParams;      // Lifecycle parameters
  econParams: EconomicParams;   // Economic parameters
}

/**
 * Actions returned by strategy for current period.
 * Matches Python StrategyActions from core/params.py exactly.
 *
 * Contains portfolio allocation weights for the current period.
 * Constraint: stockWeight + bondWeight + cashWeight should sum to 1.0
 * Constraint: consumption should be non-negative (>= 0)
 */
interface StrategyActions {
  // Portfolio weights (should sum to 1.0)
  stockWeight: number;          // Stock allocation weight (Python: stock_weight)
  bondWeight: number;           // Bond allocation weight (Python: bond_weight)
  cashWeight: number;           // Cash allocation weight (Python: cash_weight)

  // Consumption (non-negative)
  consumption: number;          // Total consumption for the period (Python: total_consumption)

  // Target dollar positions (optional, for LDI-type strategies)
  // When present, the engine re-normalizes weights to the investable base (fw + savings)
  targetFinStock?: number;
  targetFinBond?: number;
  targetFinCash?: number;
}

/**
 * Strategy protocol: a function that maps SimulationState to StrategyActions.
 * Matches Python StrategyProtocol from core/params.py.
 *
 * Strategies are simple callables that take the current simulation state
 * and return actions (consumption and portfolio weights). This allows
 * strategies to be easily swapped and compared using simulateWithStrategy.
 *
 * The strategy object can have a `name` property and optionally a `reset()`
 * method that is called at the start of each simulation path.
 */
interface Strategy {
  name: string;
  (state: SimulationState): StrategyActions;
  reset?: () => void;
}

/**
 * Unified, strategy-agnostic simulation result.
 * Matches Python SimulationResult from core/params.py.
 *
 * Works for both:
 * - Single simulation (deterministic median path): arrays are 1D [nPeriods]
 * - Monte Carlo: arrays are 2D [nSims, nPeriods]
 *
 * The strategy is an INPUT to simulation, not part of the OUTPUT type.
 * Any strategy (LDI, RuleOfThumb, Fixed, custom) produces this same result format.
 */
interface SimulationResult {
  // Identification
  strategyName: string;
  ages: number[];                     // [nPeriods] age at each time step

  // Core wealth and consumption paths
  // Shape: [nPeriods] for single sim, [nSims][nPeriods] for MC
  financialWealth: number[] | number[][];
  consumption: number[] | number[][];
  subsistenceConsumption: number[] | number[][];
  variableConsumption: number[] | number[][];

  // Portfolio allocation weights (sum to 1.0)
  stockWeight: number[] | number[][];
  bondWeight: number[] | number[][];
  cashWeight: number[] | number[][];

  // Market conditions used in simulation
  interestRates: number[] | number[][];
  stockReturns: number[] | number[][];

  // Earnings (after wage shocks applied, if any)
  earnings: number[] | number[][];

  // Human capital path (for verification)
  humanCapital: number[] | number[][];

  // Default tracking
  // For single sim: scalar; for MC: [nSims] array
  defaulted: boolean | boolean[];
  defaultAge: number | null | (number | null)[];
  finalWealth: number | number[];

  // Optional metadata for scenarios
  description: string;
}

interface Params {
  // Age parameters
  startAge: number;
  retirementAge: number;
  endAge: number;

  // Income parameters
  initialEarnings: number;
  earningsGrowth: number;
  earningsHumpAge: number;
  earningsDecline: number;

  // Expense parameters
  baseExpenses: number;
  expenseGrowth: number;
  retirementExpenses: number;

  // Portfolio parameters
  stockBetaHC: number;
  gamma: number;
  initialWealth: number;

  // Market parameters (VCV Merton)
  rBar: number;
  muStock: number;
  bondSharpe: number;  // Bond Sharpe ratio (muBond = bondSharpe * bondDuration * sigmaR)
  sigmaS: number;
  sigmaR: number;
  rho: number;
  bondDuration: number;
  phi: number;
  maxDuration?: number;  // Cap on computed durations (undefined = no cap)
  consumptionBoost: number;  // Boost above median return for consumption rate
}

// Raw muBond calculation from primitive values
function muBond(bondSharpe: number, bondDuration: number, sigmaR: number): number {
  return bondSharpe * bondDuration * sigmaR;
}

// Helper function to compute muBond from Params
function computeMuBond(params: Params): number {
  return muBond(params.bondSharpe, params.bondDuration, params.sigmaR);
}

// Convert LifecycleParams + EconomicParams to legacy Params interface
function toLegacyParams(lp: LifecycleParams, ep: EconomicParams): Params {
  return {
    startAge: lp.startAge,
    retirementAge: lp.retirementAge,
    endAge: lp.endAge,
    initialEarnings: lp.initialEarnings,
    earningsGrowth: lp.earningsGrowth,
    earningsHumpAge: lp.earningsHumpAge,
    earningsDecline: lp.earningsDecline,
    baseExpenses: lp.baseExpenses,
    expenseGrowth: lp.expenseGrowth,
    retirementExpenses: lp.retirementExpenses,
    stockBetaHC: lp.stockBetaHumanCapital,
    gamma: lp.gamma,
    initialWealth: lp.initialWealth,
    rBar: ep.rBar,
    muStock: ep.muExcess,
    bondSharpe: ep.bondSharpe,
    sigmaS: ep.sigmaS,
    sigmaR: ep.sigmaR,
    rho: ep.rho,
    bondDuration: ep.bondDuration,
    phi: ep.phi,
    maxDuration: ep.maxDuration,
    consumptionBoost: lp.consumptionBoost,
  };
}

// Compute full portfolio variance with stock-bond correlation
function computePortfolioVariance(
  wS: number, wB: number,
  sigmaS: number, sigmaR: number,
  duration: number, rho: number
): number {
  const sigmaB = duration * sigmaR;
  const covSB = -duration * sigmaS * sigmaR * rho;
  return wS * wS * sigmaS * sigmaS + wB * wB * sigmaB * sigmaB + 2 * wS * wB * covSB;
}

interface LifecycleResult {
  ages: number[];
  earnings: number[];
  expenses: number[];
  pvEarnings: number[];
  pvExpenses: number[];
  durationEarnings: number[];
  durationExpenses: number[];
  humanCapital: number[];
  hcStock: number[];
  hcBond: number[];
  hcCash: number[];
  expBond: number[];
  expCash: number[];
  financialWealth: number[];
  stockWeight: number[];
  bondWeight: number[];
  cashWeight: number[];
  subsistenceConsumption: number[];
  variableConsumption: number[];
  totalConsumption: number[];
  netWorth: number[];
  targetStock: number;
  targetBond: number;
  targetCash: number;
  // Target financial positions ($k) - dollar amounts before normalization
  targetFinStock: number[];
  targetFinBond: number[];
  targetFinCash: number[];
  // Market conditions tracking
  cumulativeStockReturn: number[];  // Cumulative stock return (1 = starting value)
  interestRate: number[];           // Interest rate path per year
  // Fixed Income Hedging Metrics
  netFiPv: number[];      // Net FI PV = bond_holdings + hc_bond - exp_bond
  dv01: number[];         // Dollar Value of 01bp rate change (per percentage point)
}

type PageType = 'base' | 'oneDraw' | 'scenarios';

type ConsumptionRule = 'adaptive' | 'fourPercent';

interface ScenarioParams {
  consumptionRule: ConsumptionRule;
  rateShockAge: number;      // Age when rate shock occurs
  rateShockMagnitude: number; // Change in rate (e.g., -0.02 for 2% drop)
  badReturnsEarly: boolean;   // Force bad returns in first 10 years of retirement
}

// =============================================================================
// Random Number Generation
// =============================================================================

// Seedable random number generator (Mulberry32)
function mulberry32(seed: number): () => number {
  return function() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Box-Muller transform for normal distribution
function boxMuller(rand: () => number): number {
  const u1 = rand();
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Generate correlated normal shocks
function generateCorrelatedShocks(
  rand: () => number,
  rho: number
): [number, number] {
  const z1 = boxMuller(rand);
  const z2 = boxMuller(rand);
  // z1 = stock shock, z2 = rate shock (correlated with z1)
  const stockShock = z1;
  const rateShock = rho * z1 + Math.sqrt(1 - rho * rho) * z2;
  return [stockShock, rateShock];
}

/**
 * Update interest rate following random walk process (phi=1.0).
 *
 * This implements the discrete-time AR(1) process for interest rates:
 *   r_{t+1} = r_bar + phi * (r_t - r_bar) + sigma_r * epsilon_t
 *
 * When phi = 1.0 (random walk), this simplifies to:
 *   r_{t+1} = r_t + sigma_r * epsilon_t
 *
 * The function tracks both:
 * - latentRate: The unbounded rate following the pure random walk
 * - observedRate: The capped rate used for pricing (floor -2%, cap 15%)
 *
 * This matches Python core/economics.py::simulate_interest_rates() exactly.
 *
 * @param latentRate - Current latent (unbounded) interest rate
 * @param sigmaR - Rate shock volatility (standard deviation)
 * @param rateShock - Standard normal shock (epsilon_t)
 * @param rateFloor - Minimum observed rate (default -0.02 = -2%)
 * @param rateCap - Maximum observed rate (default 0.15 = 15%)
 * @returns [newLatentRate, observedRate] - Updated latent rate and capped observed rate
 */
function updateInterestRate(
  latentRate: number,
  sigmaR: number,
  rateShock: number,
): [number, number] {
  // Random walk update: r_t = r_{t-1} + sigma_r * shock
  const newLatentRate = latentRate + sigmaR * rateShock;
  // No floor/cap — matches Python core/economics.py
  return [newLatentRate, newLatentRate];
}

// =============================================================================
// Core Calculation Functions
// =============================================================================

/**
 * Effective duration of a zero-coupon bond under mean-reverting rates.
 *
 * For the AR(1) process r_{t+1} = r_bar + phi(r_t - r_bar) + eps, the sensitivity
 * of a tau-year zero-coupon bond price to the current short rate is:
 *
 *     B(tau) = (1 - phi^tau) / (1 - phi)
 *
 * This is LESS than tau because mean reversion anchors long-term rates.
 *
 * As tau -> infinity, B(tau) -> 1/(1-phi). With phi=0.85, max duration ~= 6.67 years.
 *
 * @param tau - Time to maturity in years
 * @param phi - Mean reversion parameter (persistence)
 * @returns Effective duration (sensitivity to short rate changes)
 */
function effectiveDuration(tau: number, phi: number): number {
  // Edge case: zero or negative maturity
  if (tau <= 0) return 0.0;
  // No mean reversion case (random walk): duration equals maturity
  if (Math.abs(phi - 1.0) < 1e-10) return tau;
  // Mean reversion case: VCV model formula
  return (1 - Math.pow(phi, tau)) / (1 - phi);
}

/**
 * Price of a zero-coupon bond under the discrete-time Vasicek model.
 *
 * Under the expectations hypothesis, the tau-period spot rate is the average
 * of expected future short rates. Given:
 *     E_t[r_{t+k}] = r_bar + phi^k(r_t - r_bar)
 *
 * The price is:
 *     P(tau) = exp(-tau*r_bar - B(tau)*(r - r_bar))
 *
 * where B(tau) = (1 - phi^tau)/(1 - phi) is the effective duration.
 *
 * @param r - Current short rate
 * @param tau - Time to maturity
 * @param rBar - Long-run mean rate
 * @param phi - Mean reversion parameter
 * @returns Bond price (between 0 and 1)
 */
function zeroCouponPrice(r: number, tau: number, rBar: number, phi: number): number {
  // Edge case: zero or negative maturity returns par value
  if (tau <= 0) return 1.0;
  const B = effectiveDuration(tau, phi);
  return Math.exp(-tau * rBar - B * (r - rBar));
}

/**
 * Compute present value of cashflow stream.
 *
 * If rBar and phi provided (not null), uses mean-reverting term structure.
 * Otherwise uses flat discount rate.
 *
 * @param cashflows - Array of cashflow values over time
 * @param rate - Discount rate (current short rate for VCV, constant rate for flat)
 * @param phi - Mean reversion parameter (null for flat discounting)
 * @param rBar - Long-run mean rate (null for flat discounting)
 * @returns Present value of cashflow stream at time 0
 */
function computePresentValue(
  cashflows: number[],
  rate: number,
  phi: number | null = null,
  rBar: number | null = null
): number {
  if (cashflows.length === 0) return 0;

  let pv = 0;
  if (rBar !== null && phi !== null) {
    // Mean-reverting term structure
    for (let t = 0; t < cashflows.length; t++) {
      pv += cashflows[t] * zeroCouponPrice(rate, t + 1, rBar, phi);
    }
  } else {
    // Flat discount rate
    for (let t = 0; t < cashflows.length; t++) {
      pv += cashflows[t] / Math.pow(1 + rate, t + 1);
    }
  }
  return pv;
}

/**
 * Compute effective duration of cashflow stream.
 *
 * Under mean reversion (rBar and phi provided), duration is the PV-weighted
 * average of effective durations:
 *   D_eff = sum(cf[t] * P(t+1) * B(t+1)) / PV
 *
 * Under flat rate (rBar/phi null), returns traditional Macaulay duration:
 *   D = sum(t * cf[t] / (1+r)^t) / PV
 *
 * @param cashflows - Array of cashflow values over time
 * @param rate - Discount rate
 * @param phi - Mean reversion parameter (null for traditional duration)
 * @param rBar - Long-run mean rate (null for traditional duration)
 * @returns Duration of cashflow stream (sensitivity to rate changes)
 */
function computeDuration(
  cashflows: number[],
  rate: number,
  phi: number | null = null,
  rBar: number | null = null,
  maxDuration?: number
): number {
  if (cashflows.length === 0) return 0;

  const pv = computePresentValue(cashflows, rate, phi, rBar);
  if (pv < 1e-10) return 0;

  let weightedSum = 0;
  if (rBar !== null && phi !== null) {
    // Effective duration under mean reversion
    for (let t = 0; t < cashflows.length; t++) {
      const P_t = zeroCouponPrice(rate, t + 1, rBar, phi);
      const B_t = effectiveDuration(t + 1, phi);
      weightedSum += cashflows[t] * P_t * B_t;
    }
  } else {
    // Traditional Macaulay duration
    for (let t = 0; t < cashflows.length; t++) {
      weightedSum += (t + 1) * cashflows[t] / Math.pow(1 + rate, t + 1);
    }
  }
  const duration = weightedSum / pv;
  if (maxDuration !== undefined) return Math.min(duration, maxDuration);
  return duration;
}

/**
 * Compute Present Value of consumption at time 0.
 *
 * This discounts all future consumption back to the starting age,
 * providing a single metric for lifetime consumption in present value terms.
 * Uses flat discount rate (no term structure).
 *
 * @param consumption - Array of consumption values over time
 * @param rate - Discount rate (typically the risk-free rate)
 * @returns Present value of total lifetime consumption at time 0
 */
function computePvConsumption(consumption: number[], rate: number): number {
  if (consumption.length === 0) return 0;

  let pv = 0;
  for (let t = 0; t < consumption.length; t++) {
    pv += consumption[t] / Math.pow(1 + rate, t);
  }
  return pv;
}

/**
 * Calculate present value of consumption stream using realized interest rate paths.
 * This matches Python's approach: PV_realized = sum(C_t / prod(1 + r_s for s < t))
 *
 * @param consumption - Array of consumption values over time
 * @param rates - Array of realized interest rates over time
 * @returns Present value of total lifetime consumption at time 0
 */
function computePvConsumptionRealized(consumption: number[], rates: number[]): number {
  if (consumption.length === 0) return 0;

  const T = consumption.length;
  let pv = consumption[0];  // t=0 not discounted
  let cumulativeGrowth = 1.0;
  for (let t = 1; t < T; t++) {
    cumulativeGrowth *= (1 + rates[t - 1]);
    pv += consumption[t] / cumulativeGrowth;
  }
  return pv;
}

/**
 * Calculate present value of liability stream (constant consumption annuity).
 *
 * If rBar and phi are provided, uses the mean-reverting term structure:
 *   PV = sum C * P(t) where P(t) = exp(-t*r_bar - B(t)*(r - r_bar))
 *
 * Otherwise falls back to flat discount rate:
 *   PV = C * [1 - (1+r)^(-T)] / r
 *
 * @param consumption - Annual consumption amount (constant)
 * @param rate - Current short rate (for VCV) or discount rate (for flat)
 * @param yearsRemaining - Number of years of consumption remaining
 * @param rBar - Long-run mean rate (null for flat discounting)
 * @param phi - Mean reversion parameter (null for flat discounting)
 * @returns Present value of liability stream
 */
function liabilityPv(
  consumption: number,
  rate: number,
  yearsRemaining: number,
  rBar: number | null = null,
  phi: number | null = null
): number {
  if (yearsRemaining <= 0) return 0;

  // Use mean-reverting term structure if parameters provided
  if (rBar !== null && phi !== null) {
    let pv = 0;
    for (let t = 1; t <= yearsRemaining; t++) {
      pv += consumption * zeroCouponPrice(rate, t, rBar, phi);
    }
    return pv;
  }

  // Fallback to flat rate discounting
  if (rate < 1e-10) {
    return consumption * yearsRemaining;
  }
  return consumption * (1 - Math.pow(1 + rate, -yearsRemaining)) / rate;
}

/**
 * Calculate effective duration of liability stream (constant consumption annuity).
 *
 * If rBar and phi provided, returns EFFECTIVE duration (sensitivity to
 * short rate) under mean reversion:
 *   D_eff = (1/PV) * sum C * P(t) * B(t)
 *
 * where B(t) = (1 - phi^t)/(1 - phi) is the effective duration of a t-year zero.
 *
 * Otherwise returns traditional modified duration.
 *
 * @param consumption - Annual consumption amount (constant)
 * @param rate - Current short rate (for VCV) or discount rate (for flat)
 * @param yearsRemaining - Number of years of consumption remaining
 * @param rBar - Long-run mean rate (null for traditional duration)
 * @param phi - Mean reversion parameter (null for traditional duration)
 * @returns Duration of liability stream
 */
function liabilityDuration(
  consumption: number,
  rate: number,
  yearsRemaining: number,
  rBar: number | null = null,
  phi: number | null = null
): number {
  if (yearsRemaining <= 0) return 0;

  // Use effective duration under mean reversion
  if (rBar !== null && phi !== null) {
    const pv = liabilityPv(consumption, rate, yearsRemaining, rBar, phi);
    if (pv < 1e-10) return 0;

    let weightedSum = 0;
    for (let t = 1; t <= yearsRemaining; t++) {
      const P_t = zeroCouponPrice(rate, t, rBar, phi);
      const B_t = effectiveDuration(t, phi);
      weightedSum += consumption * P_t * B_t;
    }
    return weightedSum / pv;
  }

  // Fallback to traditional modified duration
  const pv = liabilityPv(consumption, rate, yearsRemaining);
  if (pv < 1e-10) return 0;

  let weightedSum = 0;
  for (let t = 1; t <= yearsRemaining; t++) {
    weightedSum += t * consumption / Math.pow(1 + rate, t + 1);
  }
  return weightedSum / pv;
}

function computeFullMertonAllocation(
  muStock: number,
  muBond: number,
  sigmaS: number,
  sigmaR: number,
  rho: number,
  duration: number,
  gamma: number
): [number, number, number] {
  if (gamma <= 0) {
    throw new Error("Risk aversion gamma must be positive");
  }

  // Bond return volatility from duration and rate volatility
  const sigmaB = duration * sigmaR;

  // Covariance: Cov(R_s, R_b) = -D * sigma_s * sigma_r * rho
  const covSB = -duration * sigmaS * sigmaR * rho;

  // Variances
  const varS = sigmaS * sigmaS;
  const varB = sigmaB * sigmaB;

  // Determinant for 2x2 matrix inversion
  const det = varS * varB - covSB * covSB;

  let stockWeight: number;
  let bondWeight: number;

  if (Math.abs(det) < 1e-12) {
    // Near-singular: fall back to single-asset solution
    stockWeight = muStock / (gamma * varS);
    bondWeight = varB > 1e-12 ? muBond / (gamma * varB) : 0;
  } else {
    // 2x2 inverse: [[a,b],[c,d]]^-1 = (1/det)*[[d,-b],[-c,a]]
    const inv00 = varB / det;
    const inv01 = -covSB / det;
    const inv10 = -covSB / det;
    const inv11 = varS / det;

    // w* = (1/gamma) * Sigma^(-1) * mu
    stockWeight = (inv00 * muStock + inv01 * muBond) / gamma;
    bondWeight = (inv10 * muStock + inv11 * muBond) / gamma;
  }

  const cashWeight = 1.0 - stockWeight - bondWeight;
  return [stockWeight, bondWeight, cashWeight];
}

function computeFullMertonAllocationConstrained(
  muStock: number,
  muBond: number,
  sigmaS: number,
  sigmaR: number,
  rho: number,
  duration: number,
  gamma: number
): [number, number, number] {
  let [wStock, wBond, wCash] = computeFullMertonAllocation(
    muStock, muBond, sigmaS, sigmaR, rho, duration, gamma
  );

  // Apply no-short constraint
  wStock = Math.max(0, wStock);
  wBond = Math.max(0, wBond);
  wCash = Math.max(0, wCash);

  // Normalize to sum to 1
  const total = wStock + wBond + wCash;
  if (total > 0) {
    wStock /= total;
    wBond /= total;
    wCash /= total;
  } else {
    wStock = 0;
    wBond = 0;
    wCash = 1;
  }

  return [wStock, wBond, wCash];
}

/**
 * Normalize portfolio weights with no-short constraint.
 *
 * Takes target financial holdings and normalizes to valid weights.
 * Matches Python core/simulation.py normalize_portfolio_weights exactly.
 *
 * When allow_leverage=false (default):
 *   - Clips negative weights to 0
 *   - Normalizes to sum to 1.0
 *   - Preserves relative proportions where possible
 *
 * When allow_leverage=true:
 *   - Returns raw weights (can be negative or >1)
 *   - Allows shorting and leveraged positions
 *
 * @param targetFinStock - Target financial stock holdings ($)
 * @param targetFinBond - Target financial bond holdings ($)
 * @param targetFinCash - Target financial cash holdings ($)
 * @param fw - Current financial wealth ($)
 * @param targetStock - Baseline target stock weight (from MV optimization)
 * @param targetBond - Baseline target bond weight
 * @param targetCash - Baseline target cash weight
 * @param maxLeverage - Maximum leverage ratio (1.0 = no borrowing, 2.0 = 1x borrow, Infinity = unconstrained)
 * @returns [stockWeight, bondWeight, cashWeight] normalized to sum to 1
 */
function normalizePortfolioWeights(
  targetFinStock: number,
  targetFinBond: number,
  targetFinCash: number,
  fw: number,
  targetStock: number,
  targetBond: number,
  targetCash: number,
  maxLeverage: number = 1.0
): [number, number, number] {
  if (fw <= 1e-6) {
    // Clip MV targets (may be negative with unconstrained optimization)
    let ws = Math.max(0, targetStock);
    let wb = Math.max(0, targetBond);
    let total = ws + wb;
    if (total > maxLeverage) {
      const scale = maxLeverage / total;
      ws *= scale;
      wb *= scale;
    }
    const wc = 1.0 - ws - wb;
    if (ws + wb > 0 || wc > 0) return [ws, wb, wc];
    return [0, 0, 1];
  }

  // Clip stocks and bonds at 0 (no shorting risky assets)
  let finStock = Math.max(0, targetFinStock);
  let finBond = Math.max(0, targetFinBond);

  // Cap total long exposure at maxLeverage * FW
  const totalLong = finStock + finBond;
  const maxLong = maxLeverage * fw;
  if (totalLong > maxLong) {
    const scale = maxLong / totalLong;
    finStock *= scale;
    finBond *= scale;
  }

  // Cash is residual: can be negative (= borrowing)
  return [finStock / fw, finBond / fw, (fw - finStock - finBond) / fw];
}

// =============================================================================
// Generic Strategy Simulation Engine
// =============================================================================

/**
 * Generic simulation engine that runs ANY strategy.
 * Matches Python simulate_with_strategy from core/simulation.py.
 *
 * This is the single source of truth for simulation logic. Strategies are
 * simple functions that map state to actions, allowing easy comparison.
 *
 * Key insight: Static calculations (median path) are just dynamic calculations
 * with zero shocks. This unifies the codebase.
 *
 * @param strategy - Strategy implementing Strategy interface (maps state -> actions)
 * @param params - Lifecycle parameters
 * @param econParams - Economic parameters
 * @param rateShocks - Shape [nSims][nPeriods] - interest rate epsilon shocks
 * @param stockShocks - Shape [nSims][nPeriods] - stock return epsilon shocks
 * @param initialRate - Starting interest rate (defaults to rBar)
 * @param description - Optional description for the simulation result
 *
 * @returns SimulationResult containing all simulation outputs with unified field names.
 *          Arrays are 2D [nSims][nPeriods] for Monte Carlo, squeezed to 1D for single sim.
 */
function simulateWithStrategy(
  strategy: Strategy,
  params: LifecycleParams,
  econParams: EconomicParams,
  rateShocks: number[][],
  stockShocks: number[][],
  initialRate: number | null = null,
  description: string = ""
): SimulationResult {
  const effectiveInitialRate = initialRate !== null ? initialRate : econParams.rBar;

  const nSims = rateShocks.length;
  const nPeriods = rateShocks[0].length;
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  if (nPeriods !== totalYears) {
    throw new Error(`Shock periods ${nPeriods} != total years ${totalYears}`);
  }

  // Compute target allocations from MV optimization (unconstrained)
  const muBond = computeMuBondFromEcon(econParams);
  const [targetStock, targetBond, targetCash] = computeFullMertonAllocation(
    econParams.muExcess, muBond, econParams.sigmaS, econParams.sigmaR,
    econParams.rho, econParams.bondDuration, params.gamma
  );

  // Simulate interest rate paths from shocks
  // ratePaths[sim][t] = observed rate at time t for simulation sim
  const ratePaths: number[][] = [];
  for (let sim = 0; sim < nSims; sim++) {
    const rates: number[] = [];
    let latentRate = effectiveInitialRate;
    let observedRate = effectiveInitialRate;
    for (let t = 0; t < totalYears; t++) {
      // For first period, use initial rate, then update
      if (t > 0) {
        [latentRate, observedRate] = updateInterestRate(
          latentRate, econParams.sigmaR, rateShocks[sim][t - 1]
        );
      }
      rates.push(observedRate);
    }
    ratePaths.push(rates);
  }

  // Simulate stock return paths
  // stockReturnPaths[sim][t] = stock return for period t for simulation sim
  const stockReturnPaths: number[][] = [];
  for (let sim = 0; sim < nSims; sim++) {
    const returns: number[] = [];
    for (let t = 0; t < totalYears; t++) {
      // Stock return = r_t + mu_excess + sigma_s * epsilon_t
      // This uses arithmetic mean returns (matching Python simulate_stock_returns)
      const stockReturn = ratePaths[sim][t] + econParams.muExcess
        + econParams.sigmaS * stockShocks[sim][t];
      returns.push(stockReturn);
    }
    stockReturnPaths.push(returns);
  }

  // Compute bond returns using duration approximation
  // bond_return = yield + spread - duration * delta_r
  const bondReturnPaths: number[][] = [];
  for (let sim = 0; sim < nSims; sim++) {
    const returns: number[] = [];
    for (let t = 0; t < totalYears - 1; t++) {
      const deltaR = ratePaths[sim][t + 1] - ratePaths[sim][t];
      const bondReturn = ratePaths[sim][t] + muBond -
        econParams.bondDuration * deltaR;
      returns.push(bondReturn);
    }
    // Last period: no rate change
    returns.push(ratePaths[sim][totalYears - 1] + muBond);
    bondReturnPaths.push(returns);
  }

  // Get BASE earnings and expenses profiles (deterministic)
  const legacyParams = toLegacyParams(params, econParams);
  const { earnings: baseEarnings, expenses } = initializeEarningsExpenses(legacyParams, totalYears, workingYears);

  const rBar = econParams.rBar;
  const phi = econParams.phi;

  // Initialize output arrays for all simulations
  const financialWealthPaths: number[][] = [];
  const humanCapitalPaths: number[][] = [];
  const totalConsumptionPaths: number[][] = [];
  const subsistenceConsumptionPaths: number[][] = [];
  const variableConsumptionPaths: number[][] = [];
  const stockWeightPaths: number[][] = [];
  const bondWeightPaths: number[][] = [];
  const cashWeightPaths: number[][] = [];
  const actualEarningsPaths: number[][] = [];
  const defaultFlags: boolean[] = [];
  const defaultAges: (number | null)[] = [];

  // Simulate each path
  for (let sim = 0; sim < nSims; sim++) {
    let defaulted = false;
    let defaultAge: number | null = null;

    // Reset strategy state if it has a reset method (for RoT retirement values)
    if (strategy.reset) {
      strategy.reset();
    }

    const financialWealth: number[] = Array(totalYears).fill(0);
    const humanCapital: number[] = Array(totalYears).fill(0);
    const totalConsumption: number[] = Array(totalYears).fill(0);
    const subsistenceConsumption: number[] = Array(totalYears).fill(0);
    const variableConsumption: number[] = Array(totalYears).fill(0);
    const stockWeight: number[] = Array(totalYears).fill(0);
    const bondWeight: number[] = Array(totalYears).fill(0);
    const cashWeight: number[] = Array(totalYears).fill(0);
    const actualEarnings: number[] = Array(totalYears).fill(0);

    // Set initial wealth
    financialWealth[0] = params.initialWealth;

    for (let t = 0; t < totalYears; t++) {
      const fw = financialWealth[t];
      const isWorking = t < workingYears;
      const currentRate = ratePaths[sim][t];
      const age = params.startAge + t;

      // For now, no wage shocks (stockBetaHumanCapital handling can be added later)
      // This matches zero-shock scenario where earnings are deterministic
      const currentEarnings = isWorking ? baseEarnings[t] : 0.0;
      actualEarnings[t] = currentEarnings;

      // Compute PV values at current rate (dynamic revaluation)
      const remainingExpenses = expenses.slice(t);
      const pvExp = computePresentValue(remainingExpenses, currentRate, phi, rBar);
      const durationExp = computeDuration(remainingExpenses, currentRate, phi, rBar, econParams.maxDuration);

      let hc = 0.0;
      let durationHc = 0.0;
      if (isWorking) {
        const remainingEarnings = baseEarnings.slice(t, workingYears);
        hc = computePresentValue(remainingEarnings, currentRate, phi, rBar);
        durationHc = computeDuration(remainingEarnings, currentRate, phi, rBar, econParams.maxDuration);
      }

      humanCapital[t] = hc;

      // Compute HC decomposition at current rate
      let hcStock = 0.0;
      let hcBond = 0.0;
      let hcCash = 0.0;
      if (isWorking && hc > 0) {
        hcStock = hc * params.stockBetaHumanCapital;
        const nonStockHc = hc * (1.0 - params.stockBetaHumanCapital);
        if (econParams.bondDuration > 0) {
          const hcBondFrac = durationHc / econParams.bondDuration;
          hcBond = nonStockHc * hcBondFrac;
          hcCash = nonStockHc * (1.0 - hcBondFrac);
        } else {
          hcBond = 0.0;
          hcCash = nonStockHc;
        }
      }

      // Compute expense decomposition at current rate
      let expBond = 0.0;
      let expCash = 0.0;
      if (econParams.bondDuration > 0 && pvExp > 0) {
        const expBondFrac = durationExp / econParams.bondDuration;
        expBond = pvExp * expBondFrac;
        expCash = pvExp * (1.0 - expBondFrac);
      } else {
        expBond = 0.0;
        expCash = pvExp;
      }

      // Compute wealth measures
      const netWorth = hc + fw - pvExp;

      // Build state for strategy
      const state: SimulationState = {
        t: t,
        age: age,
        year: t,
        isWorking: isWorking,
        currentRate: currentRate,
        humanCapital: hc,
        pvExpenses: pvExp,
        durationHc: durationHc,
        durationExp: durationExp,
        financialWealth: fw,
        netWorth: netWorth,
        earnings: currentEarnings,
        expenses: expenses[t],
        hcStockComponent: hcStock,
        hcBondComponent: hcBond,
        hcCashComponent: hcCash,
        expBondComponent: expBond,
        expCashComponent: expCash,
        targetStock: targetStock,
        targetBond: targetBond,
        targetCash: targetCash,
        params: params,
        econParams: econParams,
      };

      // Get actions from strategy
      let actions: StrategyActions;
      if (defaulted) {
        actions = {
          consumption: 0.0,
          stockWeight: targetStock,
          bondWeight: targetBond,
          cashWeight: targetCash,
        };
      } else {
        actions = strategy(state);
      }

      // Check for default
      if (!isWorking && fw <= 0 && !defaulted) {
        defaulted = true;
        defaultAge = age;
      }

      // Store results
      // Split consumption into subsistence and variable
      const subsistence = Math.min(actions.consumption, expenses[t]);
      const variable = Math.max(0, actions.consumption - expenses[t]);

      totalConsumption[t] = actions.consumption;
      subsistenceConsumption[t] = subsistence;
      variableConsumption[t] = variable;

      stockWeight[t] = actions.stockWeight;
      bondWeight[t] = actions.bondWeight;
      cashWeight[t] = actions.cashWeight;

      // Evolve wealth to next period
      if (t < totalYears - 1 && !defaulted) {
        const savings = currentEarnings - actions.consumption;
        const investable = fw + savings;

        // Re-normalize weights to investable base for exact LDI hedge
        let wSAct = actions.stockWeight;
        let wBAct = actions.bondWeight;
        let wCAct = actions.cashWeight;
        if (actions.targetFinStock !== undefined && investable > 1e-6) {
          [wSAct, wBAct, wCAct] = normalizePortfolioWeights(
            actions.targetFinStock, actions.targetFinBond!, actions.targetFinCash!,
            investable, targetStock, targetBond, targetCash,
            params.maxLeverage
          );
          stockWeight[t] = wSAct;
          bondWeight[t] = wBAct;
          cashWeight[t] = wCAct;
        }

        const stockRet = stockReturnPaths[sim][t];
        const bondRet = bondReturnPaths[sim][t];
        const cashRet = ratePaths[sim][t];

        const portfolioReturn =
          wSAct * stockRet +
          wBAct * bondRet +
          wCAct * cashRet;

        financialWealth[t + 1] = investable * (1 + portfolioReturn);
      }
    }

    financialWealthPaths.push(financialWealth);
    humanCapitalPaths.push(humanCapital);
    totalConsumptionPaths.push(totalConsumption);
    subsistenceConsumptionPaths.push(subsistenceConsumption);
    variableConsumptionPaths.push(variableConsumption);
    stockWeightPaths.push(stockWeight);
    bondWeightPaths.push(bondWeight);
    cashWeightPaths.push(cashWeight);
    actualEarningsPaths.push(actualEarnings);
    defaultFlags.push(defaulted);
    defaultAges.push(defaultAge);
  }

  // Compute ages array
  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);

  // Compute final wealth (last period)
  const finalWealth = financialWealthPaths.map(fw => fw[totalYears - 1]);

  // Get strategy name
  const strategyName = strategy.name;

  // For single simulation, squeeze arrays to 1D for cleaner API
  if (nSims === 1) {
    return {
      strategyName: strategyName,
      ages: ages,
      financialWealth: financialWealthPaths[0],
      consumption: totalConsumptionPaths[0],
      subsistenceConsumption: subsistenceConsumptionPaths[0],
      variableConsumption: variableConsumptionPaths[0],
      stockWeight: stockWeightPaths[0],
      bondWeight: bondWeightPaths[0],
      cashWeight: cashWeightPaths[0],
      interestRates: ratePaths[0],
      stockReturns: stockReturnPaths[0],
      earnings: actualEarningsPaths[0],
      humanCapital: humanCapitalPaths[0],
      defaulted: defaultFlags[0],
      defaultAge: defaultAges[0],
      finalWealth: finalWealth[0],
      description: description,
    };
  }

  return {
    strategyName: strategyName,
    ages: ages,
    financialWealth: financialWealthPaths,
    consumption: totalConsumptionPaths,
    subsistenceConsumption: subsistenceConsumptionPaths,
    variableConsumption: variableConsumptionPaths,
    stockWeight: stockWeightPaths,
    bondWeight: bondWeightPaths,
    cashWeight: cashWeightPaths,
    interestRates: ratePaths,
    stockReturns: stockReturnPaths,
    earnings: actualEarningsPaths,
    humanCapital: humanCapitalPaths,
    defaulted: defaultFlags,
    defaultAge: defaultAges,
    finalWealth: finalWealth,
    description: description,
  };
}

// =============================================================================
// LDI Strategy Implementation
// =============================================================================

/**
 * Configuration options for LDI strategy.
 */
interface LDIStrategyOptions {
  /** Share of net worth consumed above subsistence (null = derive from expected return) */
  consumptionRate?: number | null;
  /** Maximum leverage ratio (1.0 = no borrowing, Infinity = unconstrained) */
  maxLeverage?: number;
}

/**
 * Create an LDI (Liability-Driven Investment) strategy.
 *
 * This implements the optimal lifecycle strategy from Python core/strategies.py:
 * - Consumption: subsistence + consumption_rate * max(0, net_worth)
 * - Allocation: LDI hedge (target - HC component + expense hedge)
 *
 * The allocation adjusts financial holdings to offset implicit positions
 * in human capital (an asset) and expense liabilities (a liability).
 *
 * @param options - Configuration options
 * @returns Strategy object implementing the Strategy interface
 *
 * @example
 * const ldi = createLDIStrategy({ maxLeverage: 1.0 });
 * const result = simulateWithStrategy(ldi, params, econ, rateShocks, stockShocks);
 */
function createLDIStrategy(options: LDIStrategyOptions = {}): Strategy {
  const { consumptionRate = null, maxLeverage = 1.0 } = options;

  const strategy = function ldiStrategy(state: SimulationState): StrategyActions {
    const fw = state.financialWealth;

    // Step 1: Compute LDI allocation (portfolio weights FIRST)
    const surplus = Math.max(0, state.netWorth);
    const targetFinStock = state.targetStock * surplus - state.hcStockComponent;
    const targetFinBond = state.targetBond * surplus - state.hcBondComponent + state.expBondComponent;
    const targetFinCash = state.targetCash * surplus - state.hcCashComponent + state.expCashComponent;

    // Step 2: Handle fw <= 0 edge case
    if (fw <= 0) {
      return {
        consumption: 0.0,
        stockWeight: state.targetStock,
        bondWeight: state.targetBond,
        cashWeight: state.targetCash,
      };
    }

    // Normalize to weights with leverage cap
    const [wS, wB, wC] = normalizePortfolioWeights(
      targetFinStock, targetFinBond, targetFinCash,
      fw,
      state.targetStock, state.targetBond, state.targetCash,
      maxLeverage
    );

    // Step 3: Compute consumption rate using REALIZED weights (after normalization)
    let effectiveConsumptionRate: number;
    if (consumptionRate === null) {
      const r = state.currentRate;
      const sigmaS = state.econParams.sigmaS;
      const sigmaR = state.econParams.sigmaR;
      const rho = state.econParams.rho;
      const D = state.econParams.bondDuration;
      const muBond = computeMuBondFromEcon(state.econParams);

      const expectedReturn =
        wS * (r + state.econParams.muExcess) +
        wB * (r + muBond) +
        wC * r;

      const portfolioVar = computePortfolioVariance(wS, wB, sigmaS, sigmaR, D, rho);
      effectiveConsumptionRate = expectedReturn - 0.5 * portfolioVar + state.params.consumptionBoost;
    } else {
      effectiveConsumptionRate = consumptionRate;
    }

    // Step 4: Compute consumption amounts
    let subsistence = state.expenses;
    let variable = Math.max(0, effectiveConsumptionRate * state.netWorth);
    let totalCons = subsistence + variable;

    // Step 5: Apply budget constraint: cap at fw + earnings
    const available = fw + state.earnings;
    if (totalCons > available) {
      totalCons = available;
      variable = Math.max(0, available - subsistence);
      if (variable < 0) {
        subsistence = available;
        variable = 0.0;
      }
    }

    return {
      consumption: totalCons,
      stockWeight: wS,
      bondWeight: wB,
      cashWeight: wC,
      targetFinStock,
      targetFinBond,
      targetFinCash,
    };
  } as Strategy;

  Object.defineProperty(strategy, 'name', { value: 'LDI', writable: true });

  return strategy;
}

// =============================================================================
// Rule-of-Thumb Strategy Implementation
// =============================================================================

/**
 * Configuration options for Rule-of-Thumb strategy.
 */
interface RuleOfThumbStrategyOptions {
  /** Fraction of income to save during working years (default 0.15) */
  savingsRate?: number;
  /** Fixed withdrawal rate in retirement (default 0.04) */
  withdrawalRate?: number;
  /** Target duration for fixed income allocation (default 6.0) */
  targetDuration?: number;
}

/**
 * Create a Rule-of-Thumb strategy.
 *
 * This implements the classic financial advisor heuristics from Python core/strategies.py:
 * - Working years: Save savingsRate of income, (100-age)% in stocks
 * - Retirement: Fixed 4% of initial retirement wealth, allocation frozen at retirement age
 *
 * This is NOT optimal but represents common retail investment advice.
 *
 * The strategy is stateful: it remembers the retirement wealth and allocation
 * once retirement begins, so these values remain fixed throughout retirement.
 *
 * @param options - Configuration options
 * @returns Strategy object implementing the Strategy interface
 *
 * @example
 * const rot = createRuleOfThumbStrategy({ savingsRate: 0.15, withdrawalRate: 0.04 });
 * const result = simulateWithStrategy(rot, params, econ, rateShocks, stockShocks);
 */
function createRuleOfThumbStrategy(options: RuleOfThumbStrategyOptions = {}): Strategy {
  const {
    savingsRate = 0.15,
    withdrawalRate = 0.04,
    targetDuration = 6.0,
  } = options;

  // Internal state for fixed retirement values (mutable, reset between simulations)
  let retirementConsumption: number | null = null;
  let retirementStockWeight: number | null = null;
  let retirementBondWeight: number | null = null;
  let retirementCashWeight: number | null = null;

  const strategy = function ruleOfThumbStrategy(state: SimulationState): StrategyActions {
    const fw = state.financialWealth;
    const age = state.age;

    // Compute allocation: (100 - age)% stocks
    const bondDuration = state.econParams.bondDuration;
    const bondWeightInFI = bondDuration > 0 ? Math.min(1.0, targetDuration / bondDuration) : 0.0;

    let stockPct: number;
    let bondPct: number;
    let cashPct: number;

    if (state.isWorking) {
      // Working years allocation: (100 - age)% stocks
      stockPct = Math.max(0.0, Math.min(1.0, (100 - age) / 100.0));
      const fixedIncomePct = 1.0 - stockPct;
      bondPct = fixedIncomePct * bondWeightInFI;
      cashPct = fixedIncomePct * (1.0 - bondWeightInFI);
    } else {
      // Retirement: freeze allocation at retirement age
      if (retirementStockWeight === null) {
        const retirementAge = state.params.retirementAge;
        retirementStockWeight = Math.max(0.0, Math.min(1.0, (100 - retirementAge) / 100.0));
        const retirementFI = 1.0 - retirementStockWeight;
        retirementBondWeight = retirementFI * bondWeightInFI;
        retirementCashWeight = retirementFI * (1.0 - bondWeightInFI);
      }

      stockPct = retirementStockWeight;
      bondPct = retirementBondWeight!;
      cashPct = retirementCashWeight!;
    }

    // Compute consumption
    let subsistence = state.expenses;
    let totalCons: number;
    let variable: number;

    if (state.isWorking) {
      // Working years: save savingsRate of earnings, consume the rest
      const baselineConsumption = state.earnings * (1.0 - savingsRate);

      if (baselineConsumption >= subsistence) {
        totalCons = baselineConsumption;
        variable = baselineConsumption - subsistence;
      } else {
        // Can't meet baseline, try to cover subsistence
        const available = state.earnings + fw;
        if (available >= subsistence) {
          totalCons = subsistence;
          variable = 0.0;
        } else {
          totalCons = Math.max(0, available);
          subsistence = totalCons;
          variable = 0.0;
        }
      }
    } else {
      // Retirement: 4% of initial retirement wealth (fixed dollar amount)
      if (retirementConsumption === null) {
        retirementConsumption = withdrawalRate * fw;
      }

      if (fw <= 0) {
        return {
          consumption: 0.0,
          stockWeight: stockPct,
          bondWeight: bondPct,
          cashWeight: cashPct,
        };
      }

      const targetConsumption = Math.max(retirementConsumption, subsistence);
      if (fw < targetConsumption) {
        totalCons = fw;
        subsistence = Math.min(fw, state.expenses);
        variable = Math.max(0, fw - state.expenses);
      } else {
        totalCons = targetConsumption;
        variable = targetConsumption - subsistence;
      }
    }

    return {
      consumption: totalCons,
      stockWeight: stockPct,
      bondWeight: bondPct,
      cashWeight: cashPct,
    };
  } as Strategy;

  Object.defineProperty(strategy, 'name', { value: 'RuleOfThumb', writable: true });

  // Reset method to clear state between simulations
  strategy.reset = function(): void {
    retirementConsumption = null;
    retirementStockWeight = null;
    retirementBondWeight = null;
    retirementCashWeight = null;
  };

  return strategy;
}

// =============================================================================
// Strategy Comparison Types and Functions
// =============================================================================

/**
 * Comparison of two strategies using identical market conditions.
 * Matches Python StrategyComparison from core/params.py.
 *
 * Simply holds two SimulationResult objects that used the same shocks.
 * Percentiles and summary statistics are computed on demand via methods.
 */
interface StrategyComparison {
  resultA: SimulationResult;        // First strategy result (typically LDI)
  resultB: SimulationResult;        // Second strategy result (typically RoT)
  strategyAParams: Record<string, unknown>;  // Strategy A parameters for display
  strategyBParams: Record<string, unknown>;  // Strategy B parameters for display
}

/**
 * Options for Rule-of-Thumb strategy in median path comparison.
 */
interface MedianPathComparisonOptions {
  rotSavingsRate?: number;          // Savings rate during working years (default: 0.15)
  rotTargetDuration?: number;       // Target FI duration (default: 6.0)
  rotWithdrawalRate?: number;       // Withdrawal rate in retirement (default: 0.04)
}

/**
 * Compare LDI strategy vs Rule-of-Thumb on deterministic median paths.
 * Matches Python compute_median_path_comparison from core/simulation.py.
 *
 * Both strategies use expected returns (zero shocks = deterministic path).
 * This enables side-by-side comparison of median paths to understand
 * structural differences between strategies without Monte Carlo noise.
 *
 * @param params - Lifecycle parameters (uses defaults if not provided)
 * @param econParams - Economic parameters (uses defaults if not provided)
 * @param options - Rule-of-thumb strategy options
 * @returns StrategyComparison with resultA = LDI, resultB = RuleOfThumb
 *
 * @example
 * const comparison = computeMedianPathComparison();
 * console.log(comparison.resultA.strategyName); // 'LDI'
 * console.log(comparison.resultB.strategyName); // 'RuleOfThumb'
 * console.log(comparison.resultA.finalWealth);  // LDI final wealth
 * console.log(comparison.resultB.finalWealth);  // RoT final wealth
 */
function computeMedianPathComparison(
  params: LifecycleParams = DEFAULT_LIFECYCLE_PARAMS,
  econParams: EconomicParams = DEFAULT_ECON_PARAMS,
  options: MedianPathComparisonOptions = {}
): StrategyComparison {
  const {
    rotSavingsRate = 0.15,
    rotTargetDuration = 6.0,
    rotWithdrawalRate = 0.04,
  } = options;

  // Zero shocks for deterministic median paths
  const nPeriods = params.endAge - params.startAge;
  const zeroRateShocks = [Array(nPeriods).fill(0)];    // [1][nPeriods]
  const zeroStockShocks = [Array(nPeriods).fill(0)];   // [1][nPeriods]

  // Strategy 1: LDI (Liability-Driven Investment)
  const ldiStrategy = createLDIStrategy({ maxLeverage: 1.0 });
  const ldiResult = simulateWithStrategy(
    ldiStrategy, params, econParams,
    zeroRateShocks, zeroStockShocks,
    null,
    "LDI (Liability-Driven Investment)"
  );

  // Strategy 2: Rule-of-Thumb (100-age rule)
  const rotStrategy = createRuleOfThumbStrategy({
    savingsRate: rotSavingsRate,
    withdrawalRate: rotWithdrawalRate,
    targetDuration: rotTargetDuration,
  });
  const rotResult = simulateWithStrategy(
    rotStrategy, params, econParams,
    zeroRateShocks, zeroStockShocks,
    null,
    "Rule-of-Thumb (100-age rule)"
  );

  return {
    resultA: ldiResult,
    resultB: rotResult,
    strategyAParams: { maxLeverage: 1.0 },
    strategyBParams: {
      savingsRate: rotSavingsRate,
      withdrawalRate: rotWithdrawalRate,
      targetDuration: rotTargetDuration,
    },
  };
}

// =============================================================================
// Monte Carlo Simulation with Percentile Statistics
// =============================================================================

/**
 * Percentile statistics for a single field across Monte Carlo simulations.
 * Contains values at the 5th, 25th, 50th (median), 75th, and 95th percentiles
 * for each time period.
 */
interface FieldPercentiles {
  p5: number[];    // 5th percentile (pessimistic)
  p25: number[];   // 25th percentile (below median)
  p50: number[];   // 50th percentile (median)
  p75: number[];   // 75th percentile (above median)
  p95: number[];   // 95th percentile (optimistic)
}

/**
 * Complete percentile statistics for Monte Carlo simulation results.
 * Provides percentiles for key output fields at each time period.
 */
interface PercentileStats {
  financialWealth: FieldPercentiles;
  consumption: FieldPercentiles;
  stockWeight: FieldPercentiles;
  bondWeight: FieldPercentiles;
  humanCapital: FieldPercentiles;
  interestRates: FieldPercentiles;
  stockReturns: FieldPercentiles;
  cumulativeStockReturns: FieldPercentiles;
  netFiPv: FieldPercentiles;
  dv01: FieldPercentiles;
  pvExpenses: FieldPercentiles;
  totalAssets: FieldPercentiles;   // HC + FW (renamed from totalWealth)
  netWorth: FieldPercentiles;
}

/**
 * Result from Monte Carlo simulation combining raw SimulationResult with
 * computed percentile statistics.
 *
 * The `result` field contains 2D arrays (nSims x nPeriods) for each trajectory.
 * The `percentiles` field contains computed percentile statistics across simulations.
 */
interface MonteCarloSimulationResult {
  /** Raw simulation result with 2D arrays for all paths */
  result: SimulationResult;
  /** Computed percentile statistics across simulations */
  percentiles: PercentileStats;
  /** Number of simulations */
  numSims: number;
  /** Random seed used */
  seed: number;
  /** Default rate across all simulations */
  defaultRate: number;
  /** Median final wealth */
  medianFinalWealth: number;
  /** Median present value of lifetime consumption (discounted at risk-free rate) */
  medianPvConsumption: number;
  /** PV consumption for each simulation (used for histogram) */
  pvConsumption: number[];
}

/**
 * Compute percentiles for a 2D array (nSims x nPeriods) at each time period.
 *
 * @param data - 2D array of values [nSims][nPeriods]
 * @param percentile - Percentile to compute (0-100)
 * @returns Array of percentile values at each time period
 */
function computePercentileArray(data: number[][], percentile: number): number[] {
  const nPeriods = data[0].length;
  const result: number[] = [];

  for (let t = 0; t < nPeriods; t++) {
    const values = data.map(row => row[t]);
    result.push(computePercentile(values, percentile));
  }

  return result;
}

/**
 * Compute all standard percentiles (5, 25, 50, 75, 95) for a 2D array.
 *
 * @param data - 2D array of values [nSims][nPeriods]
 * @returns FieldPercentiles containing all percentile arrays
 */
function computeFieldPercentiles(data: number[][]): FieldPercentiles {
  return {
    p5: computePercentileArray(data, 5),
    p25: computePercentileArray(data, 25),
    p50: computePercentileArray(data, 50),
    p75: computePercentileArray(data, 75),
    p95: computePercentileArray(data, 95),
  };
}

/**
 * Compute netFiPv and dv01 paths from simulation result.
 *
 * Net FI PV = Bond Holdings + HC Bond Component - Expense Bond Component
 * DV01 = (Asset Dollar Duration - Liability Dollar Duration) * 0.01
 *
 * @param result - Simulation result with 2D arrays
 * @param params - Lifecycle parameters
 * @param econParams - Economic parameters
 * @returns Object with netFiPv and dv01 2D arrays
 */
function computeNetFiPvAndDv01Paths(
  result: SimulationResult,
  params: LifecycleParams,
  econParams: EconomicParams
): { netFiPv: number[][]; dv01: number[][] } {
  const financialWealth = result.financialWealth as number[][];
  const bondWeight = result.bondWeight as number[][];
  const humanCapital = result.humanCapital as number[][];
  const interestRates = result.interestRates as number[][];

  const nSims = financialWealth.length;
  const nPeriods = financialWealth[0].length;
  const workingYears = params.retirementAge - params.startAge;
  const rBar = econParams.rBar;
  const phi = econParams.phi;
  const bondDuration = econParams.bondDuration;
  const stockBeta = params.stockBetaHumanCapital;

  // Get base earnings and expenses (deterministic)
  const legacyParams = toLegacyParams(params, econParams);
  const { earnings: baseEarnings, expenses } = initializeEarningsExpenses(legacyParams, nPeriods, workingYears);

  const netFiPvPaths: number[][] = [];
  const dv01Paths: number[][] = [];

  for (let sim = 0; sim < nSims; sim++) {
    const netFiPv: number[] = [];
    const dv01: number[] = [];

    for (let t = 0; t < nPeriods; t++) {
      const fw = financialWealth[sim][t];
      const wB = bondWeight[sim][t];
      const hc = humanCapital[sim][t];
      const currentRate = interestRates[sim][t];
      const isWorking = t < workingYears;

      // Compute PV and duration of remaining expenses at current rate
      const remainingExpenses = expenses.slice(t);
      const pvExp = computePresentValue(remainingExpenses, currentRate, phi, rBar);
      const durationExp = computeDuration(remainingExpenses, currentRate, phi, rBar, econParams.maxDuration);

      // Compute duration of HC at current rate
      let durationHc = 0;
      if (isWorking) {
        const remainingEarnings = baseEarnings.slice(t, workingYears);
        durationHc = computeDuration(remainingEarnings, currentRate, phi, rBar, econParams.maxDuration);
      }

      // Decompose HC into bond-like and other components
      let hcBond = 0;
      if (isWorking && hc > 0) {
        const nonStockHc = hc * (1 - stockBeta);
        if (bondDuration > 0) {
          const bondFrac = durationHc / bondDuration;
          hcBond = nonStockHc * bondFrac;
        }
      }

      // Decompose expenses into bond-like component
      let expBond = 0;
      if (bondDuration > 0 && pvExp > 0) {
        const bondFrac = durationExp / bondDuration;
        expBond = pvExp * bondFrac;
      }

      // Net FI PV = Bond Holdings + HC Bond - Expense Bond
      const bondHoldings = wB * fw;
      netFiPv.push(bondHoldings + hcBond - expBond);

      // DV01 = bondDuration * Net_FI_PV * 0.01
      // All components (hcBond, bondHoldings, expBond) are already in bond-equivalent dollars
      // so we use bondDuration uniformly (avoids double-counting duration)
      dv01.push(bondDuration * (hcBond + bondHoldings - expBond) * 0.01);
    }

    netFiPvPaths.push(netFiPv);
    dv01Paths.push(dv01);
  }

  return { netFiPv: netFiPvPaths, dv01: dv01Paths };
}

/**
 * Compute pvExpenses, totalAssets (HC+FW), and netWorth paths for all simulations.
 * These are derived from simulation results to support Net Wealth charts.
 *
 * @param result - SimulationResult from simulateWithStrategy (2D arrays for MC)
 * @param params - Lifecycle parameters
 * @param econParams - Economic parameters
 * @returns Object with pvExpenses, totalAssets, and netWorth 2D arrays
 */
function computeWealthDecompositionPaths(
  result: SimulationResult,
  params: LifecycleParams,
  econParams: EconomicParams
): { pvExpenses: number[][]; totalAssets: number[][]; netWorth: number[][] } {
  const financialWealth = result.financialWealth as number[][];
  const humanCapital = result.humanCapital as number[][];
  const interestRates = result.interestRates as number[][];

  const nSims = financialWealth.length;
  const nPeriods = financialWealth[0].length;
  const workingYears = params.retirementAge - params.startAge;
  const rBar = econParams.rBar;
  const phi = econParams.phi;

  // Get base expenses (deterministic)
  const legacyParams = toLegacyParams(params, econParams);
  const { expenses } = initializeEarningsExpenses(legacyParams, nPeriods, workingYears);

  const pvExpensesPaths: number[][] = [];
  const totalAssetsPaths: number[][] = [];
  const netWorthPaths: number[][] = [];

  for (let sim = 0; sim < nSims; sim++) {
    const pvExp: number[] = [];
    const ta: number[] = [];
    const nw: number[] = [];

    for (let t = 0; t < nPeriods; t++) {
      const fw = financialWealth[sim][t];
      const hc = humanCapital[sim][t];
      const currentRate = interestRates[sim][t];

      // Compute PV of remaining expenses at current rate
      const remainingExpenses = expenses.slice(t);
      const pvExpVal = computePresentValue(remainingExpenses, currentRate, phi, rBar);

      pvExp.push(pvExpVal);
      ta.push(hc + fw);
      nw.push(hc + fw - pvExpVal);
    }

    pvExpensesPaths.push(pvExp);
    totalAssetsPaths.push(ta);
    netWorthPaths.push(nw);
  }

  return { pvExpenses: pvExpensesPaths, totalAssets: totalAssetsPaths, netWorth: netWorthPaths };
}

/**
 * Run Monte Carlo simulation with a single strategy.
 * Matches Python run_lifecycle_monte_carlo from core/simulation.py.
 *
 * This function:
 * 1. Generates N sets of correlated shocks using mulberry32 and generateCorrelatedShocks
 * 2. Passes them to simulateWithStrategy
 * 3. Computes percentile statistics across all simulations
 *
 * The key insight is that simulateWithStrategy already handles 2D shock arrays,
 * so this function is primarily a convenience wrapper for shock generation
 * and percentile computation.
 *
 * @param strategy - Strategy implementing the Strategy interface
 * @param params - Lifecycle parameters
 * @param econParams - Economic parameters
 * @param numSims - Number of Monte Carlo simulations (default: 50)
 * @param seed - Random seed for reproducibility (default: 42)
 * @returns MonteCarloSimulationResult with raw result and percentile statistics
 *
 * @example
 * const ldi = createLDIStrategy({ maxLeverage: 1.0 });
 * const mcResult = runMonteCarloSimulation(ldi, params, econParams, 50, 42);
 *
 * // Access percentile statistics
 * console.log(mcResult.percentiles.financialWealth.p50); // Median wealth path
 * console.log(mcResult.percentiles.consumption.p5);      // 5th percentile consumption
 *
 * // Access raw simulation data
 * const allWealthPaths = mcResult.result.financialWealth as number[][];
 * console.log(allWealthPaths[0]); // First simulation path
 *
 * // Summary statistics
 * console.log(mcResult.defaultRate);        // Fraction that defaulted
 * console.log(mcResult.medianFinalWealth);  // Median terminal wealth
 */
function runMonteCarloSimulation(
  strategy: Strategy,
  params: LifecycleParams = DEFAULT_LIFECYCLE_PARAMS,
  econParams: EconomicParams = DEFAULT_ECON_PARAMS,
  numSims: number = 50,
  seed: number = 42
): MonteCarloSimulationResult {
  const nPeriods = params.endAge - params.startAge;
  const rho = econParams.rho;

  // Generate correlated shocks for all simulations
  // Each simulation gets its own PRNG seeded differently
  const rateShocks: number[][] = [];
  const stockShocks: number[][] = [];

  for (let sim = 0; sim < numSims; sim++) {
    // Each simulation uses a different seed for its shock sequence
    const rand = mulberry32(seed + sim * 1000);
    const simRateShocks: number[] = [];
    const simStockShocks: number[] = [];

    for (let t = 0; t < nPeriods; t++) {
      const [stockShock, rateShock] = generateCorrelatedShocks(rand, rho);
      simStockShocks.push(stockShock);
      simRateShocks.push(rateShock);
    }

    rateShocks.push(simRateShocks);
    stockShocks.push(simStockShocks);
  }

  // Run simulation with all shocks
  const result = simulateWithStrategy(
    strategy,
    params,
    econParams,
    rateShocks,
    stockShocks,
    null,
    `Monte Carlo (${numSims} sims, seed=${seed})`
  );

  // Extract 2D arrays from result (we know they're 2D for numSims > 1)
  const financialWealthPaths = result.financialWealth as number[][];
  const consumptionPaths = result.consumption as number[][];
  const stockWeightPaths = result.stockWeight as number[][];
  const bondWeightPaths = result.bondWeight as number[][];
  const humanCapitalPaths = result.humanCapital as number[][];
  const interestRatePaths = result.interestRates as number[][];
  const stockReturnPaths = result.stockReturns as number[][];
  const defaultedFlags = result.defaulted as boolean[];
  const finalWealthValues = result.finalWealth as number[];

  // Compute cumulative stock returns for each simulation
  // Matches Python: np.cumprod(1 + returns, axis=1)
  const cumulativeStockReturnPaths: number[][] = stockReturnPaths.map(returns => {
    const cumulative: number[] = [1.0];
    for (let t = 0; t < returns.length; t++) {
      cumulative.push(cumulative[t] * (1 + returns[t]));
    }
    return cumulative;
  });

  // Compute Net FI PV and DV01 paths
  const { netFiPv: netFiPvPaths, dv01: dv01Paths } = computeNetFiPvAndDv01Paths(result, params, econParams);

  // Compute wealth decomposition paths for Net Wealth charts
  const { pvExpenses: pvExpensesPaths, totalAssets: totalAssetsPaths, netWorth: netWorthPaths } = computeWealthDecompositionPaths(result, params, econParams);

  // Compute percentile statistics
  const percentiles: PercentileStats = {
    financialWealth: computeFieldPercentiles(financialWealthPaths),
    consumption: computeFieldPercentiles(consumptionPaths),
    stockWeight: computeFieldPercentiles(stockWeightPaths),
    bondWeight: computeFieldPercentiles(bondWeightPaths),
    humanCapital: computeFieldPercentiles(humanCapitalPaths),
    interestRates: computeFieldPercentiles(interestRatePaths),
    stockReturns: computeFieldPercentiles(stockReturnPaths),
    cumulativeStockReturns: computeFieldPercentiles(cumulativeStockReturnPaths),
    netFiPv: computeFieldPercentiles(netFiPvPaths),
    dv01: computeFieldPercentiles(dv01Paths),
    pvExpenses: computeFieldPercentiles(pvExpensesPaths),
    totalAssets: computeFieldPercentiles(totalAssetsPaths),
    netWorth: computeFieldPercentiles(netWorthPaths),
  };

  // Compute summary statistics
  const defaultCount = defaultedFlags.filter(d => d).length;
  const defaultRate = defaultCount / numSims;
  const medianFinalWealth = computePercentile(finalWealthValues, 50);

  // Compute PV consumption for each simulation using realized rate paths
  const pvConsumptionValues: number[] = [];
  for (let sim = 0; sim < numSims; sim++) {
    const pv = computePvConsumptionRealized(consumptionPaths[sim], interestRatePaths[sim]);
    pvConsumptionValues.push(pv);
  }
  const medianPvConsumption = computePercentile(pvConsumptionValues, 50);

  return {
    result,
    percentiles,
    numSims,
    seed,
    defaultRate,
    medianFinalWealth,
    medianPvConsumption,
    pvConsumption: pvConsumptionValues,
  };
}

/**
 * Options for Monte Carlo strategy comparison.
 */
interface MonteCarloComparisonOptions extends MedianPathComparisonOptions {
  /** Number of Monte Carlo simulations (default: 50) */
  numSims?: number;
  /** Random seed for reproducibility (default: 42) */
  seed?: number;
}

/**
 * Result from Monte Carlo strategy comparison.
 */
interface MonteCarloStrategyComparison {
  /** LDI strategy Monte Carlo result */
  resultA: MonteCarloSimulationResult;
  /** Rule-of-Thumb strategy Monte Carlo result */
  resultB: MonteCarloSimulationResult;
  /** Strategy A parameters for display */
  strategyAParams: Record<string, unknown>;
  /** Strategy B parameters for display */
  strategyBParams: Record<string, unknown>;
}

/**
 * Run Monte Carlo comparison between LDI and Rule-of-Thumb strategies.
 * Matches Python run_strategy_comparison from core/simulation.py.
 *
 * Both strategies are run with IDENTICAL random shocks (same market conditions)
 * for a fair comparison. This isolates the effect of strategy differences
 * from market luck.
 *
 * @param params - Lifecycle parameters
 * @param econParams - Economic parameters
 * @param options - Comparison options (numSims, seed, RoT parameters)
 * @returns MonteCarloStrategyComparison with both strategies' MC results
 *
 * @example
 * const comparison = runMonteCarloStrategyComparison();
 * console.log(comparison.resultA.defaultRate);  // LDI default rate
 * console.log(comparison.resultB.defaultRate);  // RoT default rate
 * console.log(comparison.resultA.percentiles.financialWealth.p50);  // LDI median wealth
 */
function runMonteCarloStrategyComparison(
  params: LifecycleParams = DEFAULT_LIFECYCLE_PARAMS,
  econParams: EconomicParams = DEFAULT_ECON_PARAMS,
  options: MonteCarloComparisonOptions = {}
): MonteCarloStrategyComparison {
  const {
    numSims = 50,
    seed = 42,
    rotSavingsRate = 0.15,
    rotTargetDuration = 6.0,
    rotWithdrawalRate = 0.04,
  } = options;

  const nPeriods = params.endAge - params.startAge;
  const rho = econParams.rho;

  // Generate correlated shocks ONCE - same for both strategies
  const rateShocks: number[][] = [];
  const stockShocks: number[][] = [];

  for (let sim = 0; sim < numSims; sim++) {
    const rand = mulberry32(seed + sim * 1000);
    const simRateShocks: number[] = [];
    const simStockShocks: number[] = [];

    for (let t = 0; t < nPeriods; t++) {
      const [stockShock, rateShock] = generateCorrelatedShocks(rand, rho);
      simStockShocks.push(stockShock);
      simRateShocks.push(rateShock);
    }

    rateShocks.push(simRateShocks);
    stockShocks.push(simStockShocks);
  }

  // Strategy 1: LDI
  const ldiStrategy = createLDIStrategy({ maxLeverage: 1.0 });
  const ldiResult = simulateWithStrategy(
    ldiStrategy, params, econParams,
    rateShocks, stockShocks,
    null, "LDI (Monte Carlo)"
  );

  // Strategy 2: Rule-of-Thumb
  const rotStrategy = createRuleOfThumbStrategy({
    savingsRate: rotSavingsRate,
    withdrawalRate: rotWithdrawalRate,
    targetDuration: rotTargetDuration,
  });
  const rotResult = simulateWithStrategy(
    rotStrategy, params, econParams,
    rateShocks, stockShocks,
    null, "RuleOfThumb (Monte Carlo)"
  );

  // Compute cumulative stock returns for percentiles
  const ldiStockReturns = ldiResult.stockReturns as number[][];
  const rotStockReturns = rotResult.stockReturns as number[][];
  const ldiCumulativeStockReturns = ldiStockReturns.map(returns => {
    const cumulative: number[] = [1.0];
    for (let t = 0; t < returns.length; t++) {
      cumulative.push(cumulative[t] * (1 + returns[t]));
    }
    return cumulative;
  });
  const rotCumulativeStockReturns = rotStockReturns.map(returns => {
    const cumulative: number[] = [1.0];
    for (let t = 0; t < returns.length; t++) {
      cumulative.push(cumulative[t] * (1 + returns[t]));
    }
    return cumulative;
  });

  // Compute Net FI PV and DV01 paths for both strategies
  const { netFiPv: ldiNetFiPvPaths, dv01: ldiDv01Paths } = computeNetFiPvAndDv01Paths(ldiResult, params, econParams);
  const { netFiPv: rotNetFiPvPaths, dv01: rotDv01Paths } = computeNetFiPvAndDv01Paths(rotResult, params, econParams);

  // Compute wealth decomposition paths for Net Wealth charts
  const { pvExpenses: ldiPvExpensesPaths, totalAssets: ldiTotalAssetsPaths, netWorth: ldiNetWorthPaths } = computeWealthDecompositionPaths(ldiResult, params, econParams);
  const { pvExpenses: rotPvExpensesPaths, totalAssets: rotTotalAssetsPaths, netWorth: rotNetWorthPaths } = computeWealthDecompositionPaths(rotResult, params, econParams);

  // Compute percentile statistics for both
  const ldiPercentiles: PercentileStats = {
    financialWealth: computeFieldPercentiles(ldiResult.financialWealth as number[][]),
    consumption: computeFieldPercentiles(ldiResult.consumption as number[][]),
    stockWeight: computeFieldPercentiles(ldiResult.stockWeight as number[][]),
    bondWeight: computeFieldPercentiles(ldiResult.bondWeight as number[][]),
    humanCapital: computeFieldPercentiles(ldiResult.humanCapital as number[][]),
    interestRates: computeFieldPercentiles(ldiResult.interestRates as number[][]),
    stockReturns: computeFieldPercentiles(ldiStockReturns),
    cumulativeStockReturns: computeFieldPercentiles(ldiCumulativeStockReturns),
    netFiPv: computeFieldPercentiles(ldiNetFiPvPaths),
    dv01: computeFieldPercentiles(ldiDv01Paths),
    pvExpenses: computeFieldPercentiles(ldiPvExpensesPaths),
    totalAssets: computeFieldPercentiles(ldiTotalAssetsPaths),
    netWorth: computeFieldPercentiles(ldiNetWorthPaths),
  };

  const rotPercentiles: PercentileStats = {
    financialWealth: computeFieldPercentiles(rotResult.financialWealth as number[][]),
    consumption: computeFieldPercentiles(rotResult.consumption as number[][]),
    stockWeight: computeFieldPercentiles(rotResult.stockWeight as number[][]),
    bondWeight: computeFieldPercentiles(rotResult.bondWeight as number[][]),
    humanCapital: computeFieldPercentiles(rotResult.humanCapital as number[][]),
    interestRates: computeFieldPercentiles(rotResult.interestRates as number[][]),
    stockReturns: computeFieldPercentiles(rotStockReturns),
    cumulativeStockReturns: computeFieldPercentiles(rotCumulativeStockReturns),
    netFiPv: computeFieldPercentiles(rotNetFiPvPaths),
    dv01: computeFieldPercentiles(rotDv01Paths),
    pvExpenses: computeFieldPercentiles(rotPvExpensesPaths),
    totalAssets: computeFieldPercentiles(rotTotalAssetsPaths),
    netWorth: computeFieldPercentiles(rotNetWorthPaths),
  };

  // Compute summary statistics
  const ldiDefaulted = ldiResult.defaulted as boolean[];
  const rotDefaulted = rotResult.defaulted as boolean[];
  const ldiFinalWealth = ldiResult.finalWealth as number[];
  const rotFinalWealth = rotResult.finalWealth as number[];
  const ldiConsumption = ldiResult.consumption as number[][];
  const rotConsumption = rotResult.consumption as number[][];
  const ldiInterestRates = ldiResult.interestRates as number[][];
  const rotInterestRates = rotResult.interestRates as number[][];

  // Compute PV consumption for each simulation using realized rate paths
  const ldiPvConsumptionValues: number[] = [];
  const rotPvConsumptionValues: number[] = [];
  for (let sim = 0; sim < numSims; sim++) {
    ldiPvConsumptionValues.push(computePvConsumptionRealized(ldiConsumption[sim], ldiInterestRates[sim]));
    rotPvConsumptionValues.push(computePvConsumptionRealized(rotConsumption[sim], rotInterestRates[sim]));
  }

  return {
    resultA: {
      result: ldiResult,
      percentiles: ldiPercentiles,
      numSims,
      seed,
      defaultRate: ldiDefaulted.filter(d => d).length / numSims,
      medianFinalWealth: computePercentile(ldiFinalWealth, 50),
      medianPvConsumption: computePercentile(ldiPvConsumptionValues, 50),
      pvConsumption: ldiPvConsumptionValues,
    },
    resultB: {
      result: rotResult,
      percentiles: rotPercentiles,
      numSims,
      seed,
      defaultRate: rotDefaulted.filter(d => d).length / numSims,
      medianFinalWealth: computePercentile(rotFinalWealth, 50),
      medianPvConsumption: computePercentile(rotPvConsumptionValues, 50),
      pvConsumption: rotPvConsumptionValues,
    },
    strategyAParams: { maxLeverage: 1.0 },
    strategyBParams: {
      savingsRate: rotSavingsRate,
      withdrawalRate: rotWithdrawalRate,
      targetDuration: rotTargetDuration,
    },
  };
}

// ==============================================================================
// Teaching Scenarios: Baseline, Sequence Risk, Rate Shock
// ==============================================================================

/**
 * Teaching scenario types for demonstrating different market conditions.
 *
 * - Baseline: Normal stochastic shocks (regular Monte Carlo)
 * - SequenceRisk: Forces bad stock returns in first 5 years of retirement
 * - RateShock: Applies a one-time interest rate shock at retirement
 */
type TeachingScenarioType = 'Baseline' | 'SequenceRisk' | 'RateShock';

/**
 * Options for generating teaching scenario shocks.
 */
interface TeachingScenarioOptions {
  /** Number of Monte Carlo simulations (default: 50) */
  numSims?: number;
  /** Random seed for reproducibility (default: 42) */
  seed?: number;
  /** Number of years with bad returns for Sequence Risk (default: 5) */
  sequenceRiskYears?: number;
  /** Stock shock magnitude during sequence risk (default: -1.0 = 1.0 std devs below mean, ~-12% returns) */
  sequenceRiskStockShock?: number;
  /** Rate shock magnitude (currently unused - Rate Shock uses -1.33 std devs for 5 years before retirement) */
  rateShockMagnitude?: number;
  /** Rule-of-Thumb savings rate (default: 0.15) */
  rotSavingsRate?: number;
  /** Rule-of-Thumb withdrawal rate (default: 0.04) */
  rotWithdrawalRate?: number;
  /** Rule-of-Thumb target duration (default: 6.0) */
  rotTargetDuration?: number;
}

/**
 * Generate shock arrays for a specific teaching scenario.
 *
 * Each scenario type produces different shock patterns:
 *
 * **Baseline**: Normal stochastic shocks using standard MC generation.
 * Represents typical market conditions where returns follow the assumed
 * distribution with correlation structure.
 *
 * **SequenceRisk**: Forces strongly negative stock returns in the first
 * 5 years of retirement. This demonstrates the vulnerability to poor
 * returns early in the decumulation phase when the portfolio is largest.
 * Rate shocks remain stochastic.
 *
 * **RateShock**: Applies a one-time large negative interest rate shock
 * at retirement (e.g., -2%). This shows how rate changes affect:
 * - Present value of future liabilities (PV increases when rates fall)
 * - Bond prices (prices rise when rates fall)
 * - Duration matching effectiveness
 * Stock shocks remain stochastic.
 *
 * @param scenarioType - Type of teaching scenario
 * @param nPeriods - Total number of periods in simulation
 * @param numSims - Number of MC simulations
 * @param rho - Correlation between stock and rate shocks
 * @param params - Lifecycle parameters (for retirement age)
 * @param seed - Random seed
 * @param options - Scenario-specific options
 * @returns Object with rateShocks and stockShocks arrays (numSims x nPeriods)
 */
function generateTeachingScenarioShocks(
  scenarioType: TeachingScenarioType,
  nPeriods: number,
  numSims: number,
  rho: number,
  params: LifecycleParams,
  seed: number,
  options: TeachingScenarioOptions = {}
): { rateShocks: number[][]; stockShocks: number[][] } {
  const {
    sequenceRiskYears = 5,
    sequenceRiskStockShock = -1.0,  // -1.0 std devs matches Python (~-12% returns)
    rateShockMagnitude = -0.02,
  } = options;

  const workingYears = params.retirementAge - params.startAge;
  const rateShocks: number[][] = [];
  const stockShocks: number[][] = [];

  for (let sim = 0; sim < numSims; sim++) {
    const rand = mulberry32(seed + sim * 1000);
    const simRateShocks: number[] = [];
    const simStockShocks: number[] = [];

    for (let t = 0; t < nPeriods; t++) {
      // Generate correlated shocks as baseline
      const [stockShock, rateShock] = generateCorrelatedShocks(rand, rho);

      if (scenarioType === 'SequenceRisk') {
        // Sequence Risk: Force bad stock returns in first N years of retirement
        // Bad returns mean negative shocks (below expected return)
        if (t >= workingYears && t < workingYears + sequenceRiskYears) {
          simStockShocks.push(sequenceRiskStockShock);
        } else {
          simStockShocks.push(stockShock);
        }
        simRateShocks.push(rateShock);
      } else if (scenarioType === 'RateShock') {
        // Rate Shock: Apply rate shock for 5 years BEFORE retirement (matches Python)
        // Rate drops (negative shock = falling rates = higher bond prices but higher PV liabilities)
        simStockShocks.push(stockShock);
        const shockStart = Math.max(0, workingYears - 5);
        if (t >= shockStart && t < workingYears) {
          // Use -1.33 std devs directly (matches Python's rate_shocks[:, shock_start:shock_end] = -1.33)
          simRateShocks.push(-1.33);
        } else {
          simRateShocks.push(rateShock);
        }
      } else {
        // Baseline: Normal stochastic shocks
        simStockShocks.push(stockShock);
        simRateShocks.push(rateShock);
      }
    }

    rateShocks.push(simRateShocks);
    stockShocks.push(simStockShocks);
  }

  return { rateShocks, stockShocks };
}

/**
 * Result for a single teaching scenario with both strategies.
 */
interface TeachingScenarioResult {
  /** Scenario type */
  scenarioType: TeachingScenarioType;
  /** Human-readable description */
  description: string;
  /** LDI strategy Monte Carlo result */
  ldi: MonteCarloSimulationResult;
  /** Rule-of-Thumb strategy Monte Carlo result */
  rot: MonteCarloSimulationResult;
}

/**
 * Result containing all three teaching scenarios.
 */
interface TeachingScenarioComparison {
  /** Baseline scenario: normal stochastic returns */
  baseline: TeachingScenarioResult;
  /** Sequence Risk scenario: bad returns early in retirement */
  sequenceRisk: TeachingScenarioResult;
  /** Rate Shock scenario: sudden rate drop at retirement */
  rateShock: TeachingScenarioResult;
}

/**
 * Run all three teaching scenarios for both LDI and Rule-of-Thumb strategies.
 *
 * The three teaching scenarios demonstrate key lifecycle investment concepts:
 *
 * **Baseline**: Represents typical market conditions with normal stochastic
 * returns. Shows how each strategy performs under expected market behavior.
 * Use this as the reference point for comparison.
 *
 * **Sequence Risk**: Demonstrates the vulnerability to poor returns early
 * in retirement. Even with the same average return, getting bad returns
 * when the portfolio is largest (just after retirement) causes permanent
 * wealth destruction. LDI's duration matching may mitigate some of this.
 *
 * **Rate Shock**: Shows impact of sudden interest rate changes on portfolios.
 * When rates fall:
 * - PV of future liabilities increases (bad for retirees)
 * - Bond prices rise (good for bondholders)
 * - Duration-matched portfolios should be hedged
 * LDI's explicit duration matching should outperform RoT here.
 *
 * @param params - Lifecycle parameters
 * @param econParams - Economic parameters
 * @param options - Configuration options for scenarios
 * @returns TeachingScenarioComparison with all three scenarios
 *
 * @example
 * const scenarios = runTeachingScenarios();
 *
 * // Compare default rates across scenarios
 * console.log('Baseline LDI default:', scenarios.baseline.ldi.defaultRate);
 * console.log('Sequence Risk LDI default:', scenarios.sequenceRisk.ldi.defaultRate);
 * console.log('Rate Shock LDI default:', scenarios.rateShock.ldi.defaultRate);
 *
 * // Show that sequence risk hits RoT harder than LDI
 * console.log('Baseline RoT default:', scenarios.baseline.rot.defaultRate);
 * console.log('Sequence Risk RoT default:', scenarios.sequenceRisk.rot.defaultRate);
 */
function runTeachingScenarios(
  params: LifecycleParams = DEFAULT_LIFECYCLE_PARAMS,
  econParams: EconomicParams = DEFAULT_ECON_PARAMS,
  options: TeachingScenarioOptions = {}
): TeachingScenarioComparison {
  const {
    numSims = 50,
    seed = 42,
    rotSavingsRate = 0.15,
    rotWithdrawalRate = 0.04,
    rotTargetDuration = 6.0,
  } = options;

  const nPeriods = params.endAge - params.startAge;
  const rho = econParams.rho;

  // Create strategies
  const ldiStrategy = createLDIStrategy({ maxLeverage: 1.0 });
  const rotStrategy = createRuleOfThumbStrategy({
    savingsRate: rotSavingsRate,
    withdrawalRate: rotWithdrawalRate,
    targetDuration: rotTargetDuration,
  });

  // Helper to run both strategies with given shocks
  const runBothStrategies = (
    rateShocks: number[][],
    stockShocks: number[][],
    scenarioName: string
  ): { ldi: MonteCarloSimulationResult; rot: MonteCarloSimulationResult } => {
    // LDI
    const ldiResult = simulateWithStrategy(
      ldiStrategy,
      params,
      econParams,
      rateShocks,
      stockShocks,
      null,
      `LDI (${scenarioName})`
    );

    // Reset RoT state for new simulation
    rotStrategy.reset?.();

    // RoT
    const rotResult = simulateWithStrategy(
      rotStrategy,
      params,
      econParams,
      rateShocks,
      stockShocks,
      null,
      `RuleOfThumb (${scenarioName})`
    );

    // Compute cumulative stock returns for percentiles
    const ldiStockReturns = ldiResult.stockReturns as number[][];
    const rotStockReturns = rotResult.stockReturns as number[][];
    const ldiCumulativeStockReturns = ldiStockReturns.map(returns => {
      const cumulative: number[] = [1.0];
      for (let t = 0; t < returns.length; t++) {
        cumulative.push(cumulative[t] * (1 + returns[t]));
      }
      return cumulative;
    });
    const rotCumulativeStockReturns = rotStockReturns.map(returns => {
      const cumulative: number[] = [1.0];
      for (let t = 0; t < returns.length; t++) {
        cumulative.push(cumulative[t] * (1 + returns[t]));
      }
      return cumulative;
    });

    // Compute Net FI PV and DV01 paths for both strategies
    const { netFiPv: ldiNetFiPvPaths, dv01: ldiDv01Paths } = computeNetFiPvAndDv01Paths(ldiResult, params, econParams);
    const { netFiPv: rotNetFiPvPaths, dv01: rotDv01Paths } = computeNetFiPvAndDv01Paths(rotResult, params, econParams);

    // Compute wealth decomposition paths for Net Wealth charts
    const { pvExpenses: ldiPvExpensesPaths, totalAssets: ldiTotalAssetsPaths, netWorth: ldiNetWorthPaths } = computeWealthDecompositionPaths(ldiResult, params, econParams);
    const { pvExpenses: rotPvExpensesPaths, totalAssets: rotTotalAssetsPaths, netWorth: rotNetWorthPaths } = computeWealthDecompositionPaths(rotResult, params, econParams);

    // Compute percentiles and summary stats for LDI
    const ldiPercentiles: PercentileStats = {
      financialWealth: computeFieldPercentiles(ldiResult.financialWealth as number[][]),
      consumption: computeFieldPercentiles(ldiResult.consumption as number[][]),
      stockWeight: computeFieldPercentiles(ldiResult.stockWeight as number[][]),
      bondWeight: computeFieldPercentiles(ldiResult.bondWeight as number[][]),
      humanCapital: computeFieldPercentiles(ldiResult.humanCapital as number[][]),
      interestRates: computeFieldPercentiles(ldiResult.interestRates as number[][]),
      stockReturns: computeFieldPercentiles(ldiStockReturns),
      cumulativeStockReturns: computeFieldPercentiles(ldiCumulativeStockReturns),
      netFiPv: computeFieldPercentiles(ldiNetFiPvPaths),
      dv01: computeFieldPercentiles(ldiDv01Paths),
      pvExpenses: computeFieldPercentiles(ldiPvExpensesPaths),
      totalAssets: computeFieldPercentiles(ldiTotalAssetsPaths),
      netWorth: computeFieldPercentiles(ldiNetWorthPaths),
    };
    const ldiDefaulted = ldiResult.defaulted as boolean[];
    const ldiFinalWealth = ldiResult.finalWealth as number[];

    // Compute percentiles and summary stats for RoT
    const rotPercentiles: PercentileStats = {
      financialWealth: computeFieldPercentiles(rotResult.financialWealth as number[][]),
      consumption: computeFieldPercentiles(rotResult.consumption as number[][]),
      stockWeight: computeFieldPercentiles(rotResult.stockWeight as number[][]),
      bondWeight: computeFieldPercentiles(rotResult.bondWeight as number[][]),
      humanCapital: computeFieldPercentiles(rotResult.humanCapital as number[][]),
      interestRates: computeFieldPercentiles(rotResult.interestRates as number[][]),
      stockReturns: computeFieldPercentiles(rotStockReturns),
      cumulativeStockReturns: computeFieldPercentiles(rotCumulativeStockReturns),
      netFiPv: computeFieldPercentiles(rotNetFiPvPaths),
      dv01: computeFieldPercentiles(rotDv01Paths),
      pvExpenses: computeFieldPercentiles(rotPvExpensesPaths),
      totalAssets: computeFieldPercentiles(rotTotalAssetsPaths),
      netWorth: computeFieldPercentiles(rotNetWorthPaths),
    };
    const rotDefaulted = rotResult.defaulted as boolean[];
    const rotFinalWealth = rotResult.finalWealth as number[];
    const ldiConsumption = ldiResult.consumption as number[][];
    const rotConsumption = rotResult.consumption as number[][];
    const ldiInterestRates = ldiResult.interestRates as number[][];
    const rotInterestRates = rotResult.interestRates as number[][];

    // Compute PV consumption for each simulation using realized rate paths
    const ldiPvConsumptionValues: number[] = [];
    const rotPvConsumptionValues: number[] = [];
    for (let sim = 0; sim < numSims; sim++) {
      ldiPvConsumptionValues.push(computePvConsumptionRealized(ldiConsumption[sim], ldiInterestRates[sim]));
      rotPvConsumptionValues.push(computePvConsumptionRealized(rotConsumption[sim], rotInterestRates[sim]));
    }

    return {
      ldi: {
        result: ldiResult,
        percentiles: ldiPercentiles,
        numSims,
        seed,
        defaultRate: ldiDefaulted.filter(d => d).length / numSims,
        medianFinalWealth: computePercentile(ldiFinalWealth, 50),
        medianPvConsumption: computePercentile(ldiPvConsumptionValues, 50),
        pvConsumption: ldiPvConsumptionValues,
      },
      rot: {
        result: rotResult,
        percentiles: rotPercentiles,
        numSims,
        seed,
        defaultRate: rotDefaulted.filter(d => d).length / numSims,
        medianFinalWealth: computePercentile(rotFinalWealth, 50),
        medianPvConsumption: computePercentile(rotPvConsumptionValues, 50),
        pvConsumption: rotPvConsumptionValues,
      },
    };
  };

  // Generate shocks and run scenarios

  // 1. Baseline: Normal stochastic shocks
  const baselineShocks = generateTeachingScenarioShocks(
    'Baseline', nPeriods, numSims, rho, params, seed, options
  );
  const baselineResults = runBothStrategies(
    baselineShocks.rateShocks,
    baselineShocks.stockShocks,
    'Baseline'
  );

  // 2. Sequence Risk: Bad returns early in retirement
  const sequenceRiskShocks = generateTeachingScenarioShocks(
    'SequenceRisk', nPeriods, numSims, rho, params, seed, options
  );
  const sequenceRiskResults = runBothStrategies(
    sequenceRiskShocks.rateShocks,
    sequenceRiskShocks.stockShocks,
    'Sequence Risk'
  );

  // 3. Rate Shock: Interest rate drop at retirement
  const rateShockShocks = generateTeachingScenarioShocks(
    'RateShock', nPeriods, numSims, rho, params, seed, options
  );
  const rateShockResults = runBothStrategies(
    rateShockShocks.rateShocks,
    rateShockShocks.stockShocks,
    'Rate Shock'
  );

  return {
    baseline: {
      scenarioType: 'Baseline',
      description: 'Normal stochastic market conditions - regular Monte Carlo simulation',
      ldi: baselineResults.ldi,
      rot: baselineResults.rot,
    },
    sequenceRisk: {
      scenarioType: 'SequenceRisk',
      description: 'Sequence risk - forced bad stock returns in first 5 years of retirement',
      ldi: sequenceRiskResults.ldi,
      rot: sequenceRiskResults.rot,
    },
    rateShock: {
      scenarioType: 'RateShock',
      description: 'Interest rate shock - sudden 2% rate drop at retirement',
      ldi: rateShockResults.ldi,
      rot: rateShockResults.rot,
    },
  };
}

function computeEarningsProfile(params: Params): number[] {
  const workingYears = params.retirementAge - params.startAge;
  const earnings: number[] = [];

  for (let i = 0; i < workingYears; i++) {
    const age = params.startAge + i;
    if (age <= params.earningsHumpAge) {
      // Growth phase
      earnings.push(params.initialEarnings * Math.pow(1 + params.earningsGrowth, i));
    } else {
      // Decline phase
      const yearsFromPeak = age - params.earningsHumpAge;
      const peakEarnings = params.initialEarnings *
        Math.pow(1 + params.earningsGrowth, params.earningsHumpAge - params.startAge);
      earnings.push(peakEarnings * Math.pow(1 - params.earningsDecline, yearsFromPeak));
    }
  }

  return earnings;
}

function computeExpenseProfile(params: Params): { working: number[]; retirement: number[] } {
  const workingYears = params.retirementAge - params.startAge;
  const retirementYears = params.endAge - params.retirementAge;

  const working: number[] = [];
  for (let i = 0; i < workingYears; i++) {
    working.push(params.baseExpenses * Math.pow(1 + params.expenseGrowth, i));
  }

  const retirement = Array(retirementYears).fill(params.retirementExpenses);

  return { working, retirement };
}

// Initialize earnings and expenses arrays from profile functions
function initializeEarningsExpenses(
  params: Params, totalYears: number, workingYears: number
): { earnings: number[]; expenses: number[] } {
  const earningsProfile = computeEarningsProfile(params);
  const expenseProfile = computeExpenseProfile(params);
  const earnings = Array(totalYears).fill(0);
  const expenses = Array(totalYears).fill(0);
  for (let i = 0; i < workingYears; i++) {
    earnings[i] = earningsProfile[i];
    expenses[i] = expenseProfile.working[i];
  }
  for (let i = workingYears; i < totalYears; i++) {
    expenses[i] = expenseProfile.retirement[i - workingYears];
  }
  return { earnings, expenses };
}

function computeLifecycleMedianPath(params: Params): LifecycleResult {
  const r = params.rBar;
  const phi = params.phi;
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  // Compute target allocations from MV optimization (unconstrained)
  const muBond = computeMuBond(params);
  const [targetStock, targetBond, targetCash] = computeFullMertonAllocation(
    params.muStock, muBond, params.sigmaS, params.sigmaR,
    params.rho, params.bondDuration, params.gamma
  );

  // Initialize arrays
  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const { earnings, expenses } = initializeEarningsExpenses(params, totalYears, workingYears);

  // Forward-looking present values
  const pvEarnings = Array(totalYears).fill(0);
  const pvExpenses = Array(totalYears).fill(0);
  const durationEarnings = Array(totalYears).fill(0);
  const durationExpenses = Array(totalYears).fill(0);

  for (let i = 0; i < totalYears; i++) {
    // Remaining earnings
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }

    // Remaining expenses
    const remainingExpenses = expenses.slice(i);

    // Use VCV term structure (phi, rBar) to match Python compute_static_pvs
    pvEarnings[i] = computePresentValue(remainingEarnings, r, phi, r);
    pvExpenses[i] = computePresentValue(remainingExpenses, r, phi, r);
    durationEarnings[i] = computeDuration(remainingEarnings, r, phi, r, params.maxDuration);
    durationExpenses[i] = computeDuration(remainingExpenses, r, phi, r, params.maxDuration);
  }

  // Human capital = PV of future earnings
  const humanCapital = [...pvEarnings];

  // Decompose human capital
  const hcStock = humanCapital.map(hc => hc * params.stockBetaHC);
  const nonStockHC = humanCapital.map(hc => hc * (1 - params.stockBetaHC));

  const hcBond = Array(totalYears).fill(0);
  const hcCash = Array(totalYears).fill(0);

  // Unconstrained calculation - bond fraction can exceed 1.0 when duration > bondDuration
  for (let i = 0; i < totalYears; i++) {
    if (params.bondDuration > 0 && nonStockHC[i] > 0) {
      const bondFraction = durationEarnings[i] / params.bondDuration;
      hcBond[i] = nonStockHC[i] * bondFraction;
      hcCash[i] = nonStockHC[i] * (1 - bondFraction);
    } else {
      hcCash[i] = nonStockHC[i];
    }
  }

  // Decompose expenses (liabilities) into bond-like and cash-like
  const expBond = Array(totalYears).fill(0);
  const expCash = Array(totalYears).fill(0);

  for (let i = 0; i < totalYears; i++) {
    if (params.bondDuration > 0 && pvExpenses[i] > 0) {
      const bondFraction = durationExpenses[i] / params.bondDuration;
      expBond[i] = pvExpenses[i] * bondFraction;
      expCash[i] = pvExpenses[i] * (1 - bondFraction);
    } else {
      expCash[i] = pvExpenses[i];
    }
  }

  // Financial wealth accumulation
  const financialWealth = Array(totalYears).fill(0);
  financialWealth[0] = params.initialWealth;

  const netWorth = Array(totalYears).fill(0);
  const subsistenceConsumption = [...expenses];
  const variableConsumption = Array(totalYears).fill(0);
  const totalConsumption = Array(totalYears).fill(0);

  // NOTE: consumptionRate is now computed INSIDE the loop using realized weights
  // (dynamic programming principle — optimal actions depend on current state)

  // For wealth evolution: use full portfolio return including mu_bond
  // Stock: r + mu_excess, Bond: r + mu_bond, Cash: r
  const stockReturn = r + params.muStock;
  const bondReturn = r + muBond;
  const cashReturn = r;

  // Initialize weight arrays (computed inside the loop)
  const stockWeight = Array(totalYears).fill(0);
  const bondWeight = Array(totalYears).fill(0);
  const cashWeight = Array(totalYears).fill(0);

  // Target financial positions ($k) - before normalization
  const targetFinStockArr = Array(totalYears).fill(0);
  const targetFinBondArr = Array(totalYears).fill(0);
  const targetFinCashArr = Array(totalYears).fill(0);

  // Simulate wealth accumulation with step-by-step portfolio weight calculation
  // This matches Python's simulate_paths which computes weights at each time step
  for (let i = 0; i < totalYears; i++) {
    // Current financial wealth and human capital
    const fw = financialWealth[i];
    const hc = humanCapital[i];
    netWorth[i] = hc + fw - pvExpenses[i];

    // Surplus optimization: target = target_pct * surplus - HC + expenses
    const surplus = Math.max(0, netWorth[i]);
    const targetFinStockI = targetStock * surplus - hcStock[i];
    const targetFinBondI = targetBond * surplus - hcBond[i] + expBond[i];
    const targetFinCashI = targetCash * surplus - hcCash[i] + expCash[i];

    targetFinStockArr[i] = targetFinStockI;
    targetFinBondArr[i] = targetFinBondI;
    targetFinCashArr[i] = targetFinCashI;

    // Preliminary weights (normalized to fw) for consumption rate calculation
    const [wSPrelim, wBPrelim, wCPrelim] = normalizePortfolioWeights(
      targetFinStockI, targetFinBondI, targetFinCashI,
      fw, targetStock, targetBond, targetCash,
      1.0  // no leverage for median path
    );

    // Dynamic consumption rate using preliminary weights and current rate (r_bar for median path)
    const expectedReturnPrelim = (
      wSPrelim * (r + params.muStock) +
      wBPrelim * (r + muBond) +
      wCPrelim * r
    );
    const portfolioVarPrelim = computePortfolioVariance(
      wSPrelim, wBPrelim, params.sigmaS, params.sigmaR, params.bondDuration, params.rho
    );
    const consumptionRate = expectedReturnPrelim - 0.5 * portfolioVarPrelim + params.consumptionBoost;

    // Compute consumption using dynamic rate
    variableConsumption[i] = Math.max(0, consumptionRate * netWorth[i]);
    totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

    // Cap consumption at available resources (fw + earnings)
    const availableI = fw + earnings[i];
    if (subsistenceConsumption[i] > availableI) {
      totalConsumption[i] = availableI;
      subsistenceConsumption[i] = availableI;
      variableConsumption[i] = 0;
    } else if (totalConsumption[i] > availableI) {
      totalConsumption[i] = availableI;
      variableConsumption[i] = availableI - subsistenceConsumption[i];
    }

    const savings = earnings[i] - totalConsumption[i];
    const investable = fw + savings;

    // Re-normalize weights to investable base for exact LDI hedge
    const [wS, wB, wC] = investable > 1e-6
      ? normalizePortfolioWeights(
          targetFinStockI, targetFinBondI, targetFinCashI,
          investable, targetStock, targetBond, targetCash,
          1.0
        )
      : [wSPrelim, wBPrelim, wCPrelim] as [number, number, number];

    stockWeight[i] = wS;
    bondWeight[i] = wB;
    cashWeight[i] = wC;

    // Use geometric (median) return for wealth evolution: E[R_p] - 0.5*Var(R_p)
    const expectedReturnI = wS * (r + params.muStock) + wB * (r + muBond) + wC * r;
    const portfolioVarI = computePortfolioVariance(
      wS, wB, params.sigmaS, params.sigmaR, params.bondDuration, params.rho
    );
    const portfolioReturn = expectedReturnI - 0.5 * portfolioVarI;

    if (i < totalYears - 1) {
      financialWealth[i + 1] = Math.max(0, investable * (1 + portfolioReturn));
    }
  }

  // For median path, interest rate is constant at rBar
  // and cumulative stock return grows at median (geometric) rate
  const interestRate = Array(totalYears).fill(r);
  // Jensen's correction: geometric mean = arithmetic mean - 0.5 * variance
  // TOTAL return = risk-free rate + excess return - Jensen's adjustment
  const medianStockTotalReturn = r + params.muStock - 0.5 * params.sigmaS * params.sigmaS;
  const cumulativeStockReturn = Array.from({ length: totalYears }, (_, i) =>
    Math.pow(1 + medianStockTotalReturn, i)
  );

  // Fixed Income Hedging Metrics
  // Net FI PV = Bond Holdings + HC Bond Component - Expense Bond Component
  const netFiPv = financialWealth.map((fw, t) =>
    bondWeight[t] * fw + hcBond[t] - expBond[t]
  );

  // DV01 = bondDuration * (hcBond + bondHoldings - expBond) * 0.01
  // hcBond and expBond are already bond-equivalent amounts (scaled by dur_X / bondDuration),
  // so we use bondDuration uniformly (avoids double-counting duration)
  const dv01 = financialWealth.map((fw, t) => {
    const bondHoldings = bondWeight[t] * fw;
    return params.bondDuration * (hcBond[t] + bondHoldings - expBond[t]) * 0.01;
  });

  return {
    ages,
    earnings,
    expenses,
    pvEarnings,
    pvExpenses,
    durationEarnings,
    durationExpenses,
    humanCapital,
    hcStock,
    hcBond,
    hcCash,
    expBond,
    expCash,
    financialWealth,
    stockWeight,
    bondWeight,
    cashWeight,
    subsistenceConsumption,
    variableConsumption,
    totalConsumption,
    netWorth,
    targetStock,
    targetBond,
    targetCash,
    targetFinStock: targetFinStockArr,
    targetFinBond: targetFinBondArr,
    targetFinCash: targetFinCashArr,
    cumulativeStockReturn,
    interestRate,
    netFiPv,
    dv01,
  };
}

// =============================================================================
// Verification Export Function
// =============================================================================

/**
 * Generate verification data for comparison with Python implementation.
 * This exports all key values to JSON for systematic verification.
 *
 * Usage:
 *   1. Click "Debug" button in the UI
 *   2. Copy the JSON output
 *   3. Save to output/typescript_verification.json
 *   4. Run: python compare_implementations.py --compare output/typescript_verification.json
 */
function generateVerificationData(params: Params): Record<string, unknown> {
  const r = params.rBar;
  const phi = params.phi;
  const muBond = computeMuBond(params);

  // Economic function tests
  const economicFunctions = {
    effective_duration_20_085: effectiveDuration(20, 0.85),
    effective_duration_20_100: effectiveDuration(20, 1.0),
    effective_duration_10_085: effectiveDuration(10, 0.85),
    effective_duration_5_085: effectiveDuration(5, 0.85),
    zero_coupon_price_002_20: zeroCouponPrice(0.02, 20, 0.02, 1.0),
    zero_coupon_price_003_20: zeroCouponPrice(0.03, 20, 0.02, 1.0),
    zero_coupon_price_002_10_085: zeroCouponPrice(0.02, 10, 0.02, 0.85),
    mu_bond: muBond,
    mu_bond_formula: params.bondSharpe * params.bondDuration * params.sigmaR,
  };

  // PV function tests
  const earnings40 = Array(40).fill(100.0);
  const expenses30 = Array(30).fill(100.0);
  const pvFunctions = {
    pv_earnings_40yr_flat: computePresentValue(earnings40, r),
    pv_expenses_30yr_flat: computePresentValue(expenses30, r),
    pv_earnings_40yr_vcv: computePresentValue(earnings40, r, phi, r),
    pv_expenses_30yr_vcv: computePresentValue(expenses30, r, phi, r),
    duration_earnings_40yr_flat: computeDuration(earnings40, r),
    duration_expenses_30yr_flat: computeDuration(expenses30, r),
    duration_earnings_40yr_vcv: computeDuration(earnings40, r, phi, r),
    duration_expenses_30yr_vcv: computeDuration(expenses30, r, phi, r),
  };

  // MV optimization tests
  const [stockU, bondU, cashU] = computeFullMertonAllocation(
    params.muStock, muBond, params.sigmaS, params.sigmaR,
    params.rho, params.bondDuration, params.gamma
  );
  const [stockC, bondC, cashC] = computeFullMertonAllocationConstrained(
    params.muStock, muBond, params.sigmaS, params.sigmaR,
    params.rho, params.bondDuration, params.gamma
  );
  const mvOptimization = {
    target_stock_unconstrained: stockU,
    target_bond_unconstrained: bondU,
    target_cash_unconstrained: cashU,
    target_stock: stockC,
    target_bond: bondC,
    target_cash: cashC,
    gamma: params.gamma,
  };

  // Lifecycle arrays
  const result = computeLifecycleMedianPath(params);
  const lifecycleArrays = {
    ages: result.ages,
    earnings: result.earnings,
    expenses: result.expenses,
    pv_earnings: result.pvEarnings,
    pv_expenses: result.pvExpenses,
    duration_earnings: result.durationEarnings,
    duration_expenses: result.durationExpenses,
    human_capital: result.humanCapital,
    hc_stock: result.hcStock,
    hc_bond: result.hcBond,
    hc_cash: result.hcCash,
    exp_bond: result.expBond,
    exp_cash: result.expCash,
    // Sample values at key ages
    pv_earnings_age_25: result.pvEarnings[0],
    pv_earnings_age_45: result.pvEarnings[20],
    pv_earnings_age_64: result.pvEarnings[39],
    pv_expenses_age_25: result.pvExpenses[0],
    pv_expenses_age_65: result.pvExpenses[40],
    pv_expenses_age_85: result.pvExpenses[60],
    duration_earnings_age_25: result.durationEarnings[0],
    duration_earnings_age_45: result.durationEarnings[20],
    duration_expenses_age_65: result.durationExpenses[40],
  };

  // Median path
  const medianPath = {
    ages: result.ages,
    financial_wealth: result.financialWealth,
    net_worth: result.netWorth,
    stock_weight: result.stockWeight,
    bond_weight: result.bondWeight,
    cash_weight: result.cashWeight,
    total_consumption: result.totalConsumption,
    subsistence_consumption: result.subsistenceConsumption,
    variable_consumption: result.variableConsumption,
    // Sample values at key ages
    fw_age_25: result.financialWealth[0],
    fw_age_45: result.financialWealth[20],
    fw_age_65: result.financialWealth[40],
    fw_age_85: result.financialWealth[60],
    stock_weight_age_25: result.stockWeight[0],
    stock_weight_age_45: result.stockWeight[20],
    stock_weight_age_65: result.stockWeight[40],
    stock_weight_age_85: result.stockWeight[60],
    consumption_age_65: result.totalConsumption[40],
    consumption_age_85: result.totalConsumption[60],
  };

  // Simulation test with zero shocks
  const totalYears = params.endAge - params.startAge;
  const zeroShocks = [Array(totalYears).fill(0)];

  // Convert Params to LifecycleParams and EconomicParams for simulateWithStrategy
  const lp: LifecycleParams = {
    startAge: params.startAge,
    retirementAge: params.retirementAge,
    endAge: params.endAge,
    initialEarnings: params.initialEarnings,
    earningsGrowth: params.earningsGrowth,
    earningsHumpAge: params.earningsHumpAge,
    earningsDecline: params.earningsDecline,
    baseExpenses: params.baseExpenses,
    expenseGrowth: params.expenseGrowth,
    retirementExpenses: params.retirementExpenses,
    consumptionShare: 0.05,
    consumptionBoost: 0.0,
    stockBetaHumanCapital: params.stockBetaHC,
    gamma: params.gamma,
    targetStockAllocation: 0.6,
    targetBondAllocation: 0.3,
    maxLeverage: 1.0,
    riskFreeRate: params.rBar,
    equityPremium: params.muStock,
    initialWealth: params.initialWealth,
  };
  const ep: EconomicParams = {
    rBar: params.rBar,
    phi: params.phi,
    sigmaR: params.sigmaR,
    muExcess: params.muStock,
    bondSharpe: params.bondSharpe,
    sigmaS: params.sigmaS,
    rho: params.rho,
    bondDuration: params.bondDuration,
    maxDuration: params.maxDuration,
  };

  const ldiStrategy = createLDIStrategy({ maxLeverage: 1.0 });
  const simResult = simulateWithStrategy(ldiStrategy, lp, ep, zeroShocks, zeroShocks);

  const simulationTest = {
    zero_shock_fw: simResult.financialWealth as number[],
    zero_shock_hc: simResult.humanCapital as number[],
    zero_shock_stock_weight: simResult.stockWeight as number[],
    zero_shock_consumption: simResult.consumption as number[],
    target_stock: stockC,
    target_bond: bondC,
    target_cash: cashC,
  };

  return {
    metadata: {
      source: "TypeScript (web visualizer)",
      version: "1.0",
      parameters: {
        r_bar: params.rBar,
        phi: params.phi,
        sigma_r: params.sigmaR,
        mu_excess: params.muStock,
        bond_sharpe: params.bondSharpe,
        sigma_s: params.sigmaS,
        rho: params.rho,
        bond_duration: params.bondDuration,
        start_age: params.startAge,
        retirement_age: params.retirementAge,
        end_age: params.endAge,
        initial_earnings: params.initialEarnings,
        base_expenses: params.baseExpenses,
        gamma: params.gamma,
        initial_wealth: params.initialWealth,
        stock_beta_human_capital: params.stockBetaHC,
      },
    },
    economic_functions: economicFunctions,
    pv_functions: pvFunctions,
    mv_optimization: mvOptimization,
    lifecycle_arrays: lifecycleArrays,
    median_path: medianPath,
    simulation_test: simulationTest,
  };
}

/**
 * Export verification data to console and clipboard.
 * Call this function from the Debug button.
 */
function exportVerificationData(params: Params): void {
  const data = generateVerificationData(params);
  const json = JSON.stringify(data, null, 2);
  console.log("=== TYPESCRIPT VERIFICATION DATA ===");
  console.log(json);
  console.log("=== END VERIFICATION DATA ===");

  // Copy to clipboard if available
  if (navigator.clipboard) {
    navigator.clipboard.writeText(json).then(() => {
      console.log("Verification data copied to clipboard!");
    }).catch(err => {
      console.error("Failed to copy to clipboard:", err);
    });
  }

  // Also create a downloadable file
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'typescript_verification.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// =============================================================================
// Monte Carlo Simulation
// =============================================================================

function computeStochasticPath(params: Params, rand: () => number): LifecycleResult {
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  // Initialize arrays
  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const { earnings, expenses } = initializeEarningsExpenses(params, totalYears, workingYears);

  // State variables that evolve with shocks
  let latentRate = params.rBar;  // Latent rate follows pure random walk
  let currentRate = params.rBar; // Observed rate = capped latent rate
  const phi = params.phi;
  const r = params.rBar;
  const muBondVal = computeMuBond(params);

  // Compute target allocations ONCE (constant, matching Python — unconstrained)
  const [targetStock, targetBond, targetCash] = computeFullMertonAllocation(
    params.muStock, muBondVal, params.sigmaS, params.sigmaR,
    params.rho, params.bondDuration, params.gamma
  );

  // NOTE: consumptionRate is now computed INSIDE the loop using current_rate
  // and realized weights (dynamic programming principle)

  // Track arrays for each year
  const {
    pvEarnings, pvExpenses, durationEarnings, durationExpenses,
    humanCapital, hcStock, hcBond, hcCash, expBond, expCash,
    financialWealth, stockWeight, bondWeight, cashWeight,
    subsistenceConsumption, variableConsumption, totalConsumption, netWorth,
  } = initializeLifecycleArrays(totalYears, expenses);
  // Market conditions tracking
  const interestRateArr = Array(totalYears).fill(params.rBar);
  const cumulativeStockReturnArr = Array(totalYears).fill(1);
  let cumStockReturn = 1;

  financialWealth[0] = params.initialWealth;

  // Simulate year by year
  for (let i = 0; i < totalYears; i++) {
    // Generate correlated shocks for this year
    const [stockShock, rateShock] = generateCorrelatedShocks(rand, params.rho);

    // Save pre-shock rate for PV computation and bond returns (matches Python timing)
    const preShockRate = currentRate;

    // Update interest rate using random walk (phi=1.0)
    [latentRate, currentRate] = updateInterestRate(latentRate, params.sigmaR, rateShock);

    // Track interest rate (post-shock, for display)
    interestRateArr[i] = currentRate;

    // Compute PVs and durations with PRE-SHOCK rate (matches Python: rate_paths[sim, t])
    // Python pre-generates the full rate path, then at time t uses rate_paths[sim, t]
    // which is the rate BEFORE this period's shock is applied to wealth evolution
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }
    const remainingExpenses = expenses.slice(i);

    pvEarnings[i] = computePresentValue(remainingEarnings, preShockRate, phi, params.rBar);
    pvExpenses[i] = computePresentValue(remainingExpenses, preShockRate, phi, params.rBar);
    durationEarnings[i] = computeDuration(remainingEarnings, preShockRate, phi, params.rBar, params.maxDuration);
    durationExpenses[i] = computeDuration(remainingExpenses, preShockRate, phi, params.rBar, params.maxDuration);

    humanCapital[i] = pvEarnings[i];

    // Decompose human capital (no cap on bond fraction — matches Python and median path)
    hcStock[i] = humanCapital[i] * params.stockBetaHC;
    const nonStockHC = humanCapital[i] * (1 - params.stockBetaHC);

    if (params.bondDuration > 0 && nonStockHC > 0) {
      const bondFraction = durationEarnings[i] / params.bondDuration;
      hcBond[i] = nonStockHC * bondFraction;
      hcCash[i] = nonStockHC * (1 - bondFraction);
    } else {
      hcCash[i] = nonStockHC;
    }

    // Decompose expenses
    if (params.bondDuration > 0 && pvExpenses[i] > 0) {
      const bondFraction = durationExpenses[i] / params.bondDuration;
      expBond[i] = pvExpenses[i] * bondFraction;
      expCash[i] = pvExpenses[i] * (1 - bondFraction);
    } else {
      expCash[i] = pvExpenses[i];
    }

    // Surplus optimization: target = target_pct * surplus - HC + expenses
    netWorth[i] = humanCapital[i] + financialWealth[i] - pvExpenses[i];
    const surplus = Math.max(0, netWorth[i]);
    const targetFinStocks = targetStock * surplus - hcStock[i];
    const targetFinBonds = targetBond * surplus - hcBond[i] + expBond[i];
    const targetFinCash = targetCash * surplus - hcCash[i] + expCash[i];

    // Preliminary weights (normalized to fw) for consumption rate calculation
    const [wSPrelim, wBPrelim, wCPrelim] = normalizePortfolioWeights(
      targetFinStocks, targetFinBonds, targetFinCash,
      financialWealth[i], targetStock, targetBond, targetCash,
      1.0  // no leverage
    );

    // Dynamic consumption rate: use preShockRate (current rate) and preliminary weights
    const expectedReturnPrelimI = (
      wSPrelim * (preShockRate + params.muStock) +
      wBPrelim * (preShockRate + muBondVal) +
      wCPrelim * preShockRate
    );
    const portfolioVarPrelimI = computePortfolioVariance(
      wSPrelim, wBPrelim, params.sigmaS, params.sigmaR, params.bondDuration, params.rho
    );
    const consumptionRateI = expectedReturnPrelimI - 0.5 * portfolioVarPrelimI + params.consumptionBoost;

    variableConsumption[i] = Math.max(0, consumptionRateI * netWorth[i]);
    totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

    // Cap consumption at available resources (fw + earnings)
    {
      const fw = financialWealth[i];
      const availableI = fw + earnings[i];
      if (subsistenceConsumption[i] > availableI) {
        totalConsumption[i] = availableI;
        subsistenceConsumption[i] = availableI;
        variableConsumption[i] = 0;
      } else if (totalConsumption[i] > availableI) {
        totalConsumption[i] = availableI;
        variableConsumption[i] = availableI - subsistenceConsumption[i];
      }
    }

    // Re-normalize weights to investable base for exact LDI hedge
    {
      const savings = earnings[i] - totalConsumption[i];
      const investable = financialWealth[i] + savings;
      if (investable > 1e-6) {
        const [wS, wB, wC] = normalizePortfolioWeights(
          targetFinStocks, targetFinBonds, targetFinCash,
          investable, targetStock, targetBond, targetCash,
          1.0
        );
        stockWeight[i] = wS;
        bondWeight[i] = wB;
        cashWeight[i] = wC;
      } else {
        stockWeight[i] = wSPrelim;
        bondWeight[i] = wBPrelim;
        cashWeight[i] = wCPrelim;
      }
    }

    // Wealth accumulation with stochastic returns
    // Track cumulative stock return at start of this year
    cumulativeStockReturnArr[i] = cumStockReturn;

    if (i < totalYears - 1) {
      const savings = earnings[i] - totalConsumption[i];
      const investable = financialWealth[i] + savings;

      // Realized returns with shocks (matching Python duration approximation)
      // Stock return uses pre-shock rate as yield component (matches Python: rate_paths[sim, t])
      const stockReturn = preShockRate + params.muStock
        + params.sigmaS * stockShock;
      // Bond return: r_t - duration * delta_r + mu_bond (matches Python compute_duration_approx_returns)
      // delta_r = currentRate - preShockRate = sigma_r * rateShock (when phi=1)
      const deltaR = currentRate - preShockRate;
      const bondReturn = preShockRate - params.bondDuration * deltaR + muBondVal;
      const cashReturn = preShockRate;

      // Update cumulative stock return
      cumStockReturn *= (1 + stockReturn);

      const portfolioReturn = stockWeight[i] * stockReturn +
                             bondWeight[i] * bondReturn +
                             cashWeight[i] * cashReturn;

      financialWealth[i + 1] = Math.max(0, investable * (1 + portfolioReturn));
    }
  }

  return {
    ages,
    earnings,
    expenses,
    pvEarnings,
    pvExpenses,
    durationEarnings,
    durationExpenses,
    humanCapital,
    hcStock,
    hcBond,
    hcCash,
    expBond,
    expCash,
    financialWealth,
    stockWeight,
    bondWeight,
    cashWeight,
    subsistenceConsumption,
    variableConsumption,
    totalConsumption,
    netWorth,
    targetStock,
    targetBond,
    targetCash,
    targetFinStock: Array(totalYears).fill(0),
    targetFinBond: Array(totalYears).fill(0),
    targetFinCash: Array(totalYears).fill(0),
    cumulativeStockReturn: cumulativeStockReturnArr,
    interestRate: interestRateArr,
    // These are computed separately by computeNetFiPvAndDv01Paths, not in single-path simulation
    netFiPv: Array(totalYears).fill(0),
    dv01: Array(totalYears).fill(0),
  };
}

function computePercentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

// Create histogram bins from two datasets using geometrically spaced edges
function createGeomHistogramBins(
  dataA: number[], dataB: number[],
  edges: number[],
  labelFn: (lo: number) => string,
  nameA = 'LDI', nameB = 'RoT'
): { bin: string; [key: string]: number | string }[] {
  const result: { bin: string; [key: string]: number | string }[] = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const lo = edges[i];
    const hi = edges[i + 1];
    result.push({
      bin: labelFn(lo),
      [nameA]: dataA.filter(v => v >= lo && v < hi).length,
      [nameB]: dataB.filter(v => v >= lo && v < hi).length,
    });
  }
  return result;
}

// Create geometrically spaced bin edges
function geomSpacedEdges(min: number, max: number, numBins: number): number[] {
  const edges: number[] = [];
  for (let i = 0; i <= numBins; i++) {
    edges.push(min * Math.pow(max / min, i / numBins));
  }
  return edges;
}

// Initialize standard lifecycle tracking arrays (all zero-filled except subsistenceConsumption)
function initializeLifecycleArrays(totalYears: number, expenses: number[]) {
  return {
    pvEarnings: Array(totalYears).fill(0) as number[],
    pvExpenses: Array(totalYears).fill(0) as number[],
    durationEarnings: Array(totalYears).fill(0) as number[],
    durationExpenses: Array(totalYears).fill(0) as number[],
    humanCapital: Array(totalYears).fill(0) as number[],
    hcStock: Array(totalYears).fill(0) as number[],
    hcBond: Array(totalYears).fill(0) as number[],
    hcCash: Array(totalYears).fill(0) as number[],
    expBond: Array(totalYears).fill(0) as number[],
    expCash: Array(totalYears).fill(0) as number[],
    financialWealth: Array(totalYears).fill(0) as number[],
    stockWeight: Array(totalYears).fill(0) as number[],
    bondWeight: Array(totalYears).fill(0) as number[],
    cashWeight: Array(totalYears).fill(0) as number[],
    subsistenceConsumption: [...expenses] as number[],
    variableConsumption: Array(totalYears).fill(0) as number[],
    totalConsumption: Array(totalYears).fill(0) as number[],
    netWorth: Array(totalYears).fill(0) as number[],
  };
}

// =============================================================================
// Scenario Simulation (for teaching concepts)
// =============================================================================

interface ScenarioResult {
  ages: number[];
  financialWealth: number[];
  totalConsumption: number[];
  subsistenceConsumption: number[];
  variableConsumption: number[];
  rate: number[];
  defaulted: boolean;        // Did they fail to meet subsistence?
  defaultAge: number | null; // Age when default occurred
  terminalWealth: number;
  // For scenario visualization
  cumulativeStockReturn: number[];  // Cumulative stock return (1 = starting value)
  interestRate: number[];           // Interest rate path
}

function computeScenarioPath(
  params: Params,
  scenario: ScenarioParams,
  rand: () => number,
  preRetirementConsumption?: number[]  // Optional: use this consumption during working years
): ScenarioResult {
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const { earnings, expenses } = initializeEarningsExpenses(params, totalYears, workingYears);

  // State tracking
  let latentRate = params.rBar;  // Latent rate follows pure random walk
  let currentRate = params.rBar; // Observed rate = capped latent rate
  const phi = params.phi;
  const r = params.rBar;
  const muBondVal = computeMuBond(params);

  // Compute target allocations ONCE (constant, matching Python — unconstrained)
  const [targetStockAlloc, targetBondAlloc, targetCashAlloc] = computeFullMertonAllocation(
    params.muStock, muBondVal, params.sigmaS, params.sigmaR,
    params.rho, params.bondDuration, params.gamma
  );

  // NOTE: consumptionRate is now computed INSIDE the loop using current_rate
  // and realized weights (dynamic programming principle)

  const financialWealth = Array(totalYears).fill(0);
  const subsistenceConsumption = [...expenses];
  const variableConsumption = Array(totalYears).fill(0);
  const totalConsumption = Array(totalYears).fill(0);
  const rateHistory = Array(totalYears).fill(params.rBar);
  const cumulativeStockReturn = Array(totalYears).fill(1);
  let cumStockReturn = 1;

  financialWealth[0] = params.initialWealth;

  // For 4% rule: calculate initial withdrawal amount at retirement
  // Based on FINANCIAL wealth at retirement (not human capital)
  let fourPercentWithdrawal = 0;
  let fourPercentCalculated = false;

  let defaulted = false;
  let defaultAge: number | null = null;

  // Simulate year by year
  for (let i = 0; i < totalYears; i++) {
    const age = params.startAge + i;

    // Generate shocks
    let [stockShock, rateShock] = generateCorrelatedShocks(rand, params.rho);

    // Apply "bad returns early" scenario: force negative stock returns in first 10 years of retirement
    if (scenario.badReturnsEarly && i >= workingYears && i < workingYears + 10) {
      stockShock = -Math.abs(stockShock) * 0.5 - 0.3; // Force moderately negative (~-5% avg/year)
    }

    // Apply rate shock to latent rate at specified age
    if (age === scenario.rateShockAge) {
      latentRate += scenario.rateShockMagnitude;
    }

    // Save pre-shock rate for PV computation and bond returns (matches Python timing)
    const preShockRate = currentRate;

    // Update interest rate using random walk (phi=1.0)
    [latentRate, currentRate] = updateInterestRate(latentRate, params.sigmaR, rateShock);
    rateHistory[i] = currentRate;

    // Compute PVs with PRE-SHOCK rate (matches Python: rate_paths[sim, t])
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }
    const remainingExpenses = expenses.slice(i);

    const pvEarnings = computePresentValue(remainingEarnings, preShockRate, phi, params.rBar);
    const pvExpenses = computePresentValue(remainingExpenses, preShockRate, phi, params.rBar);
    const durationEarnings = computeDuration(remainingEarnings, preShockRate, phi, params.rBar, params.maxDuration);

    const humanCapital = pvEarnings;

    // HC decomposition for portfolio (no cap on bond fraction — matches Python and median path)
    const hcStock = humanCapital * params.stockBetaHC;
    const nonStockHC = humanCapital * (1 - params.stockBetaHC);
    let hcBond = 0;
    let hcCash = 0;
    if (params.bondDuration > 0 && nonStockHC > 0) {
      const bondFraction = durationEarnings / params.bondDuration;
      hcBond = nonStockHC * bondFraction;
      hcCash = nonStockHC * (1 - bondFraction);
    } else {
      hcCash = nonStockHC;
    }

    // Decompose expenses
    let expBond = 0;
    let expCash = 0;
    if (params.bondDuration > 0 && pvExpenses > 0) {
      const durationExp = computeDuration(remainingExpenses, preShockRate, phi, params.rBar, params.maxDuration);
      const bondFraction = durationExp / params.bondDuration;
      expBond = pvExpenses * bondFraction;
      expCash = pvExpenses * (1 - bondFraction);
    } else {
      expCash = pvExpenses;
    }

    // Surplus optimization: target = target_pct * surplus - HC + expenses
    const netWorth = humanCapital + financialWealth[i] - pvExpenses;
    const surplus = Math.max(0, netWorth);
    const targetFinStocks = targetStockAlloc * surplus - hcStock;
    const targetFinBonds = targetBondAlloc * surplus - hcBond + expBond;
    const targetFinCash = targetCashAlloc * surplus - hcCash + expCash;

    // Preliminary weights (normalized to fw) for consumption rate calculation
    const [stockWeightPrelim, bondWeightPrelim, cashWeightPrelim] = normalizePortfolioWeights(
      targetFinStocks, targetFinBonds, targetFinCash,
      financialWealth[i], targetStockAlloc, targetBondAlloc, targetCashAlloc,
      1.0  // no leverage
    );

    // Consumption decision based on rule
    if (scenario.consumptionRule === 'adaptive') {
      // Adaptive: dynamic consumption rate using preShockRate and preliminary weights
      const expectedReturnI = (
        stockWeightPrelim * (preShockRate + params.muStock) +
        bondWeightPrelim * (preShockRate + muBondVal) +
        cashWeightPrelim * preShockRate
      );
      const portfolioVarI = computePortfolioVariance(
        stockWeightPrelim, bondWeightPrelim, params.sigmaS, params.sigmaR, params.bondDuration, params.rho
      );
      const consumptionRateI = expectedReturnI - 0.5 * portfolioVarI + params.consumptionBoost;
      variableConsumption[i] = Math.max(0, consumptionRateI * netWorth);
      totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

      // Cap consumption at available resources (fw + earnings)
      {
        const fw = financialWealth[i];
        const availableI = fw + earnings[i];
        if (subsistenceConsumption[i] > availableI) {
          if (!defaulted) {
            defaulted = true;
            defaultAge = params.startAge + i;
          }
          totalConsumption[i] = availableI;
          subsistenceConsumption[i] = availableI;
          variableConsumption[i] = 0;
        } else if (totalConsumption[i] > availableI) {
          totalConsumption[i] = availableI;
          variableConsumption[i] = availableI - subsistenceConsumption[i];
        }
      }
    } else {
      // 4% Rule: fixed withdrawal from financial wealth at retirement
      if (i < workingYears) {
        // Before retirement: use the same consumption as the adaptive case (if provided)
        if (preRetirementConsumption && preRetirementConsumption[i] !== undefined) {
          totalConsumption[i] = preRetirementConsumption[i];
          variableConsumption[i] = Math.max(0, totalConsumption[i] - subsistenceConsumption[i]);
        } else {
          // Fallback: consume from earnings, capped at financial wealth
          variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
          totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];
          const fwCap = financialWealth[i];
          if (totalConsumption[i] > fwCap) {
            totalConsumption[i] = fwCap;
            variableConsumption[i] = Math.max(0, fwCap - subsistenceConsumption[i]);
          }
        }
      } else {
        // At retirement, calculate the 4% withdrawal based on FINANCIAL wealth only
        if (!fourPercentCalculated) {
          fourPercentWithdrawal = 0.04 * financialWealth[i];
          fourPercentCalculated = true;
        }

        // After retirement: fixed 4% withdrawal (same dollar amount each year)
        totalConsumption[i] = fourPercentWithdrawal;
        variableConsumption[i] = Math.max(0, fourPercentWithdrawal - subsistenceConsumption[i]);

        // Check if we can meet the withdrawal (or at least subsistence)
        if (financialWealth[i] < fourPercentWithdrawal) {
          // Can only withdraw what we have
          totalConsumption[i] = Math.max(0, financialWealth[i]);
          variableConsumption[i] = Math.max(0, totalConsumption[i] - subsistenceConsumption[i]);

          // Check if we can't even meet subsistence
          if (financialWealth[i] < subsistenceConsumption[i]) {
            if (!defaulted) {
              defaulted = true;
              defaultAge = age;
            }
            totalConsumption[i] = Math.max(0, financialWealth[i]);
            variableConsumption[i] = 0;
          }
        }
      }
    }

    // Wealth accumulation
    const savings = earnings[i] - totalConsumption[i];
    const investable = financialWealth[i] + savings;

    // Re-normalize weights to investable base for exact LDI hedge
    let stockWeight: number, bondWeight: number, cashWeight: number;
    if (investable > 1e-6) {
      [stockWeight, bondWeight, cashWeight] = normalizePortfolioWeights(
        targetFinStocks, targetFinBonds, targetFinCash,
        investable, targetStockAlloc, targetBondAlloc, targetCashAlloc,
        1.0
      );
    } else {
      stockWeight = stockWeightPrelim;
      bondWeight = bondWeightPrelim;
      cashWeight = cashWeightPrelim;
    }

    // Realized returns with shocks (matching Python duration approximation)
    // Stock return uses pre-shock rate as yield component (matches Python: rate_paths[sim, t])
    const stockReturn = preShockRate + params.muStock
      + params.sigmaS * stockShock;
    cumStockReturn *= (1 + stockReturn);
    cumulativeStockReturn[i] = cumStockReturn;

    if (i < totalYears - 1) {
      // Bond return: r_t - duration * delta_r + mu_bond (matches Python compute_duration_approx_returns)
      const deltaR = currentRate - preShockRate;
      const bondReturn = preShockRate - params.bondDuration * deltaR + muBondVal;
      const cashReturn = preShockRate;

      const portfolioReturn = stockWeight * stockReturn +
                             bondWeight * bondReturn +
                             cashWeight * cashReturn;

      financialWealth[i + 1] = Math.max(0, investable * (1 + portfolioReturn));
    }
  }

  return {
    ages,
    financialWealth,
    totalConsumption,
    subsistenceConsumption,
    variableConsumption,
    rate: rateHistory,
    defaulted,
    defaultAge,
    terminalWealth: financialWealth[financialWealth.length - 1],
    cumulativeStockReturn,
    interestRate: rateHistory,
  };
}

function runScenarioComparison(
  params: Params,
  numRuns: number = 50,
  baseSeed: number = 42,
  badReturnsEarly: boolean = false,
  rateShockAge: number = 0,
  rateShockMagnitude: number = 0
): { adaptive: ScenarioResult[]; fourPercent: ScenarioResult[] } {
  const adaptiveRuns: ScenarioResult[] = [];
  const fourPercentRuns: ScenarioResult[] = [];

  for (let run = 0; run < numRuns; run++) {
    const rand = mulberry32(baseSeed + run * 1000);

    // Run adaptive scenario first
    const adaptiveScenario: ScenarioParams = {
      consumptionRule: 'adaptive',
      rateShockAge,
      rateShockMagnitude,
      badReturnsEarly,
    };
    const adaptiveResult = computeScenarioPath(params, adaptiveScenario, rand);
    adaptiveRuns.push(adaptiveResult);

    // Run 4% rule with same random seed AND same pre-retirement consumption
    // This ensures both arrive at retirement with identical wealth
    const rand2 = mulberry32(baseSeed + run * 1000);
    const fourPercentScenario: ScenarioParams = {
      consumptionRule: 'fourPercent',
      rateShockAge,
      rateShockMagnitude,
      badReturnsEarly,
    };
    // Pass the adaptive consumption for working years so 4% rule has same starting point
    fourPercentRuns.push(computeScenarioPath(params, fourPercentScenario, rand2, adaptiveResult.totalConsumption));
  }

  return { adaptive: adaptiveRuns, fourPercent: fourPercentRuns };
}

// =============================================================================
// UI Components
// =============================================================================

interface StepperInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  decimals?: number;
}

function StepperInput({ label, value, onChange, min, max, step, suffix = '', decimals = 0 }: StepperInputProps) {
  const [inputValue, setInputValue] = useState(value.toFixed(decimals));

  const handleBlur = () => {
    const parsed = parseFloat(inputValue);
    if (!isNaN(parsed)) {
      const clamped = Math.max(min, Math.min(max, parsed));
      onChange(clamped);
      setInputValue(clamped.toFixed(decimals));
    } else {
      setInputValue(value.toFixed(decimals));
    }
  };

  const handleStep = (delta: number) => {
    const newValue = Math.max(min, Math.min(max, value + delta));
    onChange(newValue);
    setInputValue(newValue.toFixed(decimals));
  };

  React.useEffect(() => {
    setInputValue(value.toFixed(decimals));
  }, [value, decimals]);

  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{ fontSize: '11px', color: '#666', marginBottom: '2px' }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <button
          onClick={() => handleStep(-step)}
          style={{
            width: '24px',
            height: '24px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            background: '#f5f5f5',
            cursor: 'pointer',
            fontSize: '14px',
          }}
        >
          -
        </button>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={(e) => e.key === 'Enter' && handleBlur()}
          style={{
            width: '50px',
            height: '24px',
            textAlign: 'center',
            border: '1px solid #ccc',
            borderRadius: '4px',
            fontSize: '12px',
          }}
        />
        <span style={{ fontSize: '11px', color: '#666', minWidth: '20px' }}>{suffix}</span>
        <button
          onClick={() => handleStep(step)}
          style={{
            width: '24px',
            height: '24px',
            border: '1px solid #ccc',
            borderRadius: '4px',
            background: '#f5f5f5',
            cursor: 'pointer',
            fontSize: '14px',
          }}
        >
          +
        </button>
      </div>
    </div>
  );
}

function ParamGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{
        fontSize: '12px',
        fontWeight: 'bold',
        color: '#333',
        borderBottom: '1px solid #ddd',
        paddingBottom: '4px',
        marginBottom: '8px',
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

// =============================================================================
// Chart Components
// =============================================================================

const COLORS = {
  stock: '#e74c3c',
  bond: '#3498db',
  cash: '#2ecc71',
  earnings: '#9b59b6',
  expenses: '#e67e22',
  hc: '#1abc9c',
  fw: '#34495e',
  subsistence: '#7f8c8d',  // Darker gray for better contrast
  variable: '#f39c12',
};

// Number formatters
const formatDollarM = (value: number) => `$${(value / 1000).toFixed(2)}M`;
const formatDollarK = (value: number) => `$${Math.round(value)}k`;
const formatDollar = (value: number) => Math.round(value).toLocaleString();
const formatPercent = (value: number) => `${Math.round(value)}`;
const formatYears = (value: number) => value.toFixed(1);

const tooltipFmt = (fn: (v: number) => string) =>
  (v: number | undefined) => v !== undefined ? fn(v) : '';

const dollarMTooltipFormatter = tooltipFmt(formatDollarM);
const dollarKTooltipFormatter = tooltipFmt(formatDollarK);
const dollarTooltipFormatter = tooltipFmt((v) => `$${formatDollar(v)}k`);
const percentTooltipFormatter = tooltipFmt((v) => `${Math.round(v)}%`);
const yearsTooltipFormatter = tooltipFmt((v) => `${formatYears(v)} yrs`);

// Symmetric log scale — matches Python's symlog(linthresh=50, linscale=0.5)
// Linear within ±linthresh, log outside. Used for wealth charts that cross zero.
const SYMLOG_THRESH = 50;
const SYMLOG_LINSCALE = 0.5;
const SYMLOG_LOG_THRESH = Math.log10(SYMLOG_THRESH);

function symlog(v: number): number {
  if (Math.abs(v) <= SYMLOG_THRESH) return v * SYMLOG_LINSCALE / SYMLOG_THRESH;
  const sign = v > 0 ? 1 : -1;
  return sign * (SYMLOG_LINSCALE + Math.log10(Math.abs(v)) - SYMLOG_LOG_THRESH);
}

function symlogInv(t: number): number {
  if (Math.abs(t) <= SYMLOG_LINSCALE) return t * SYMLOG_THRESH / SYMLOG_LINSCALE;
  const sign = t > 0 ? 1 : -1;
  return sign * Math.pow(10, Math.abs(t) - SYMLOG_LINSCALE + SYMLOG_LOG_THRESH);
}

const symlogTickFormatter = (t: number) => {
  const v = symlogInv(t);
  const absV = Math.abs(v);
  if (absV < 1) return '$0k';
  if (absV < 1000) return `$${Math.round(v)}k`;
  return `$${(v / 1000).toFixed(1)}M`;
};

const EarningsConsumptionTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;
  const earnings = data.earnings;
  const consumption = data.totalConsumption;
  const savings = earnings - consumption;
  return (
    <div style={{ background: 'white', padding: '8px 12px', border: '1px solid #ccc', borderRadius: 4, fontSize: '12px', lineHeight: '1.6' }}>
      <div style={{ fontWeight: 'bold', marginBottom: 2 }}>{label}</div>
      <div style={{ color: COLORS.earnings }}>Earnings: ${formatDollar(earnings)}k</div>
      <div style={{ color: COLORS.expenses }}>Total Consumption: ${formatDollar(consumption)}k</div>
      <div style={{ color: savings >= 0 ? COLORS.cash : COLORS.expenses }}>
        {savings >= 0 ? `Savings: $${formatDollar(savings)}k` : `Drawdown: -$${formatDollar(Math.abs(savings))}k`}
      </div>
    </div>
  );
};

const symlogTooltipFormatter = tooltipFmt((t) => {
  const v = symlogInv(t);
  return `$${Math.round(v)}k`;
});

function ChartSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '24px' }}>
      <h3 style={{ fontSize: '15px', fontWeight: 'bold', marginBottom: '12px', color: '#2c3e50' }}>
        {title}
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
        {children}
      </div>
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: '#fff',
      border: '1px solid #e0e0e0',
      borderRadius: '8px',
      padding: '12px',
      minHeight: '320px',
    }}>
      <div style={{ fontSize: '13px', fontWeight: '500', marginBottom: '8px', color: '#444' }}>
        {title}
      </div>
      {children}
    </div>
  );
}

// =============================================================================
// Main App Component
// =============================================================================

export default function LifecycleVisualizer() {
  // Page navigation
  const [currentPage, setCurrentPage] = useState<PageType>('base');

  // Default parameters - synced with Python core/params.py defaults
  const [params, setParams] = useState<Params>({
    startAge: 25,
    retirementAge: 65,
    endAge: 95,
    initialEarnings: 200,
    earningsGrowth: 0.0,
    earningsHumpAge: 65,
    earningsDecline: 0.0,
    baseExpenses: 100,
    expenseGrowth: 0.0,
    retirementExpenses: 100,
    stockBetaHC: 0.0,
    gamma: 2,
    initialWealth: 100,
    rBar: 0.02,
    muStock: 0.045,
    bondSharpe: 0.0,
    sigmaS: 0.18,
    sigmaR: 0.003,
    rho: 0.0,
    bondDuration: 20,
    phi: 1.0,
    consumptionBoost: 0.0,
  });

  const updateParam = (key: keyof Params, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Compute lifecycle results
  const result = useMemo(() => computeLifecycleMedianPath(params), [params]);

  // Scenario state
  const [scenarioType, setScenarioType] = useState<'summary' | 'baseline' | 'sequenceRisk' | 'rateShock'>('summary');
  const [rateShockAge, setRateShockAge] = useState(50);
  const [rateShockMagnitude, setRateShockMagnitude] = useState(-0.02);
  const [scenarioRetirementAge, setScenarioRetirementAge] = useState(params.retirementAge);
  const [scenarioEndAge, setScenarioEndAge] = useState(params.endAge);

  // Sync scenario ages when main params change
  useEffect(() => { setScenarioRetirementAge(params.retirementAge); }, [params.retirementAge]);
  useEffect(() => { setScenarioEndAge(params.endAge); }, [params.endAge]);

  // scenarioBadReturns is handled by the scenarioType - sequenceRisk forces bad returns
  // Simulation control state for deferred computation
  // simulationVersion increments each time "Run Simulation" is clicked
  const [simulationVersion, setSimulationVersion] = useState(0);
  const [scenarioComputing, setScenarioComputing] = useState(false);
  // Cache teaching scenarios results - kept even when params change
  const [cachedTeachingScenarios, setCachedTeachingScenarios] = useState<ReturnType<typeof runTeachingScenarios> | null>(null);
  // Track the params that were used for the last simulation (to detect staleness)
  const [lastSimulationParams, setLastSimulationParams] = useState<{
    lifecycleParams: LifecycleParams;
    econParams: EconomicParams;
    rateShockMagnitude: number;
  } | null>(null);

  // One Draw state
  const [oneDrawSeedInput, setOneDrawSeedInput] = useState('');
  const [oneDrawVersion, setOneDrawVersion] = useState(0);
  const [oneDrawComputing, setOneDrawComputing] = useState(false);
  const [cachedOneDraw, setCachedOneDraw] = useState<{
    ldiResult: SimulationResult;
    seed: number;
  } | null>(null);

  // Create LifecycleParams for runTeachingScenarios
  const lifecycleParams = useMemo((): LifecycleParams => ({
    startAge: params.startAge,
    retirementAge: scenarioRetirementAge,
    endAge: scenarioEndAge,
    initialEarnings: params.initialEarnings,
    earningsGrowth: params.earningsGrowth,
    earningsHumpAge: params.earningsHumpAge,
    earningsDecline: params.earningsDecline,
    baseExpenses: params.baseExpenses,
    expenseGrowth: params.expenseGrowth,
    retirementExpenses: params.retirementExpenses,
    consumptionShare: 0.05,
    consumptionBoost: params.consumptionBoost,
    gamma: params.gamma,
    stockBetaHumanCapital: params.stockBetaHC,  // Use selected beta from params
    targetStockAllocation: 0.6,
    targetBondAllocation: 0.3,
    maxLeverage: 1.0,
    riskFreeRate: params.rBar,
    equityPremium: params.muStock,
    initialWealth: params.initialWealth,
  }), [params, scenarioRetirementAge, scenarioEndAge]);

  // Create EconomicParams for runTeachingScenarios
  const econParams = useMemo((): EconomicParams => ({
    rBar: params.rBar,
    phi: params.phi,
    sigmaR: params.sigmaR,
    muExcess: params.muStock,
    bondSharpe: params.bondSharpe,
    sigmaS: params.sigmaS,
    rho: params.rho,
    bondDuration: params.bondDuration,
    maxDuration: params.maxDuration,
  }), [params]);

  // ==========================================================================
  // SINGLE SOURCE OF TRUTH: Teaching Scenarios Computation
  // ==========================================================================
  // This useEffect computes cachedTeachingScenarios which is used by BOTH:
  //   1. Summary tab (bar charts and table)
  //   2. Individual scenario tabs (Baseline, Sequence Risk, Rate Shock)
  //
  // Both tabs access the SAME teachingScenarios object, so they MUST show
  // identical numbers. If you see different numbers between tabs:
  //   - Check browser cache (hard refresh with Cmd+Shift+R)
  //   - Verify the page has been rebuilt after code changes
  //   - Check console for any errors during simulation
  //
  // Deferred computation: only runs after "Run Simulation" button is clicked
  // Keeps old results when params change (don't auto-clear on param changes)
  // Uses useEffect to trigger computation only when simulationVersion changes
  // Store params to use when simulation runs (captured at button click time)
  const [pendingSimulationParams, setPendingSimulationParams] = useState<{
    lifecycleParams: LifecycleParams;
    econParams: EconomicParams;
    rateShockMagnitude: number;
  } | null>(null);

  // ONLY trigger on simulationVersion changes - NOT on param changes
  // This prevents param changes from automatically re-running simulation
  useEffect(() => {
    if (simulationVersion === 0) return; // Don't run until button clicked
    if (currentPage !== 'scenarios') return;
    if (!pendingSimulationParams) return; // No params captured

    // Compute and cache the results using the captured params
    const results = runTeachingScenarios(
      pendingSimulationParams.lifecycleParams,
      pendingSimulationParams.econParams,
      {
        numSims: 500,  // 500 simulations for statistical significance
        seed: 42,
        rotSavingsRate: 0.15,
        rotWithdrawalRate: 0.04,
        rotTargetDuration: 6.0,
        rateShockMagnitude: pendingSimulationParams.rateShockMagnitude,
      }
    );
    setCachedTeachingScenarios(results);
    // Save the params used for this simulation (to detect staleness later)
    setLastSimulationParams(pendingSimulationParams);
    setScenarioComputing(false);
  }, [simulationVersion, currentPage, pendingSimulationParams]);

  // teachingScenarios is the SINGLE SOURCE OF TRUTH for both Summary and individual tabs
  const teachingScenarios = cachedTeachingScenarios;

  // ==========================================================================
  // One Draw Computation
  // ==========================================================================
  const [pendingOneDrawParams, setPendingOneDrawParams] = useState<{
    lifecycleParams: LifecycleParams;
    econParams: EconomicParams;
    seed: number;
  } | null>(null);

  useEffect(() => {
    if (oneDrawVersion === 0) return;
    if (!pendingOneDrawParams) return;

    const { lifecycleParams: lp, econParams: ep, seed } = pendingOneDrawParams;
    const totalYears = lp.endAge - lp.startAge;

    // Generate shocks for one simulation
    const rand = mulberry32(seed);
    const rateShocks: number[] = [];
    const stockShocks: number[] = [];
    for (let t = 0; t < totalYears; t++) {
      const [sShock, rShock] = generateCorrelatedShocks(rand, ep.rho);
      stockShocks.push(sShock);
      rateShocks.push(rShock);
    }

    // Run LDI strategy with these shocks
    const ldiStrategy = createLDIStrategy({ maxLeverage: 1.0 });
    const ldiResult = simulateWithStrategy(
      ldiStrategy, lp, ep,
      [rateShocks], [stockShocks],
      null, `One Draw (seed=${seed})`
    );

    setCachedOneDraw({ ldiResult, seed });
    setOneDrawComputing(false);
  }, [oneDrawVersion, pendingOneDrawParams]);

  // Transform cached one-draw result into chart data
  const oneDrawChartData = useMemo(() => {
    if (!cachedOneDraw) return null;
    const { ldiResult } = cachedOneDraw;
    const ages = ldiResult.ages;
    const fw = ldiResult.financialWealth as number[];
    const rates = ldiResult.interestRates as number[];
    const stockReturns = ldiResult.stockReturns as number[];
    const hc = ldiResult.humanCapital as number[];
    const sW = ldiResult.stockWeight as number[];
    const bW = ldiResult.bondWeight as number[];
    const cW = ldiResult.cashWeight as number[];
    const subsCons = ldiResult.subsistenceConsumption as number[];
    const varCons = ldiResult.variableConsumption as number[];
    const earnings = ldiResult.earnings as number[];

    // Compute cumulative stock return (growth of $1)
    const cumStockReturn: number[] = [1];
    for (let i = 0; i < stockReturns.length; i++) {
      cumStockReturn.push(cumStockReturn[i] * (1 + stockReturns[i]));
    }

    // Compute bond returns: r_t + mu_bond - D * delta_r
    const muBond = computeMuBondFromEcon(econParams);
    const D = econParams.bondDuration;
    const bondReturns: number[] = [];
    for (let t = 0; t < ages.length - 1; t++) {
      const deltaR = rates[t + 1] - rates[t];
      bondReturns.push(rates[t] + muBond - D * deltaR);
    }
    bondReturns.push(rates[ages.length - 1] + muBond); // last period: no rate change

    // Compute cumulative bond return (growth of $1)
    const cumBondReturn: number[] = [1];
    for (let i = 0; i < bondReturns.length; i++) {
      cumBondReturn.push(cumBondReturn[i] * (1 + bondReturns[i]));
    }

    // Compute rebalancing: purchase = fw[t+1]*w[t+1] - fw[t]*w[t]*(1+asset_ret[t])
    const stockPurchasePct: (number | null)[] = [null]; // no trade at t=0
    const bondPurchasePct: (number | null)[] = [null];
    for (let t = 0; t < ages.length - 1; t++) {
      const stockAfter = fw[t] * sW[t] * (1 + stockReturns[t]);
      const bondAfter = fw[t] * bW[t] * (1 + bondReturns[t]);
      const sPurch = fw[t + 1] * sW[t + 1] - stockAfter;
      const bPurch = fw[t + 1] * bW[t + 1] - bondAfter;
      stockPurchasePct.push(fw[t + 1] > 0 ? (sPurch / fw[t + 1]) * 100 : 0);
      bondPurchasePct.push(fw[t + 1] > 0 ? (bPurch / fw[t + 1]) * 100 : 0);
    }

    // Compute PV expenses at each time step using the realized rate path
    const nPeriods = ages.length;
    const workingYears = params.retirementAge - params.startAge;
    const legacyParams = toLegacyParams(lifecycleParams, econParams);
    const { expenses } = initializeEarningsExpenses(legacyParams, nPeriods, workingYears);
    const pvExpenses: number[] = [];
    for (let t = 0; t < nPeriods; t++) {
      const remainingExpenses = expenses.slice(t);
      const pvExp = computePresentValue(remainingExpenses, rates[t], econParams.phi, econParams.rBar);
      pvExpenses.push(pvExp);
    }

    return ages.map((age, i) => ({
      age,
      // Market
      interestRate: rates[i] * 100,
      cumStockReturn: cumStockReturn[i],
      cumBondReturn: cumBondReturn[i],
      // Wealth
      humanCapital: hc[i],
      financialWealth: fw[i],
      totalAssets: hc[i] + fw[i],
      // PV Expenses and Net Worth
      pvExpenses: pvExpenses[i],
      netWorth: hc[i] + fw[i] - pvExpenses[i],
      // Allocation
      stockWeight: sW[i] * 100,
      bondWeight: bW[i] * 100,
      cashWeight: cW[i] * 100,
      // Consumption
      subsistence: subsCons[i],
      variable: varCons[i],
      totalConsumption: subsCons[i] + varCons[i],
      // Base case median reference
      baseMedianFW: result.financialWealth[i],
      baseMedianConsumption: result.totalConsumption[i],
      // Cash flow
      earnings: earnings[i],
      cashFlow: earnings[i] - (subsCons[i] + varCons[i]),
      // Rebalancing
      stockPurchasePct: stockPurchasePct[i],
      bondPurchasePct: bondPurchasePct[i],
    }));
  }, [cachedOneDraw, result, econParams, params, lifecycleParams]);

  // Check if simulation results are stale (params have changed since last run)
  const simulationResultsStale = useMemo(() => {
    if (!lastSimulationParams || !teachingScenarios) return false;
    // Compare current params to the params used for the last simulation
    const lp = lastSimulationParams.lifecycleParams;
    const ep = lastSimulationParams.econParams;
    return (
      lp.startAge !== lifecycleParams.startAge ||
      lp.retirementAge !== lifecycleParams.retirementAge ||
      lp.endAge !== lifecycleParams.endAge ||
      lp.initialEarnings !== lifecycleParams.initialEarnings ||
      lp.earningsGrowth !== lifecycleParams.earningsGrowth ||
      lp.baseExpenses !== lifecycleParams.baseExpenses ||
      lp.retirementExpenses !== lifecycleParams.retirementExpenses ||
      lp.gamma !== lifecycleParams.gamma ||
      lp.stockBetaHumanCapital !== lifecycleParams.stockBetaHumanCapital ||
      lp.initialWealth !== lifecycleParams.initialWealth ||
      ep.rBar !== econParams.rBar ||
      ep.phi !== econParams.phi ||
      ep.sigmaR !== econParams.sigmaR ||
      ep.muExcess !== econParams.muExcess ||
      ep.bondSharpe !== econParams.bondSharpe ||
      ep.sigmaS !== econParams.sigmaS ||
      ep.rho !== econParams.rho ||
      ep.bondDuration !== econParams.bondDuration ||
      lastSimulationParams.rateShockMagnitude !== rateShockMagnitude
    );
  }, [lastSimulationParams, teachingScenarios, lifecycleParams, econParams, rateShockMagnitude]);

  // Prepare chart data
  const chartData = useMemo(() => {
    return result.ages.map((age, i) => ({
      age,
      earnings: result.earnings[i],
      expenses: result.expenses[i],
      pvEarnings: result.pvEarnings[i],
      pvExpenses: result.pvExpenses[i],
      durationEarnings: result.durationEarnings[i],
      durationExpenses: result.durationExpenses[i],  // Positive for easier comparison with HC duration
      humanCapital: result.humanCapital[i],
      financialWealth: result.financialWealth[i],
      hcStock: result.hcStock[i],
      hcBond: result.hcBond[i],
      hcCash: result.hcCash[i],
      expBond: result.expBond[i],
      expCash: result.expCash[i],
      netStock: result.hcStock[i],
      netBond: result.hcBond[i] - result.expBond[i],
      netCash: result.hcCash[i] - result.expCash[i],
      netTotal: result.humanCapital[i] - result.pvExpenses[i],
      stockWeight: result.stockWeight[i] * 100,
      bondWeight: result.bondWeight[i] * 100,
      cashWeight: result.cashWeight[i] * 100,
      subsistence: result.subsistenceConsumption[i],
      variable: result.variableConsumption[i],
      totalConsumption: result.totalConsumption[i],
      // Market assumptions
      cumulativeStockReturn: result.cumulativeStockReturn[i],
      cumulativeStockReturnArithmetic: Math.pow(1 + params.rBar + params.muStock, i),
      interestRate: result.interestRate[i] * 100,  // Convert to percentage
      // Total Assets (HC + FW)
      totalAssets: result.humanCapital[i] + result.financialWealth[i],
      // Net Worth (HC + FW - PV Expenses)
      netWorth: result.netWorth[i],
      // Risk Management
      netFiPv: result.netFiPv[i],
      dv01: result.dv01[i],
      // Target Financial Positions ($k)
      targetFinStock: result.targetFinStock[i],
      targetFinBond: result.targetFinBond[i],
      targetFinCash: result.targetFinCash[i],
      // Savings = earnings - consumption
      savings: result.earnings[i] - result.totalConsumption[i],
      savingsPositive: Math.max(0, result.earnings[i] - result.totalConsumption[i]),
      savingsNegative: Math.min(0, result.earnings[i] - result.totalConsumption[i]),
    }));
  }, [result, params.rBar, params.muStock]);

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      {/* Left Sidebar - Parameters */}
      <div style={{
        width: '220px',
        borderRight: '1px solid #e0e0e0',
        padding: '12px',
        overflowY: 'auto',
        background: '#fafafa',
        flexShrink: 0,
      }}>
        <h2 style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '16px', color: '#2c3e50' }}>
          Parameters
        </h2>

        <ParamGroup title="Retirement Planning">
          <StepperInput
            label="Current Age"
            value={params.startAge}
            onChange={(v) => updateParam('startAge', v)}
            min={20} max={60} step={1} suffix="" decimals={0}
          />
          <StepperInput
            label="Retirement Age"
            value={params.retirementAge}
            onChange={(v) => updateParam('retirementAge', v)}
            min={params.startAge + 5} max={80} step={1} suffix="" decimals={0}
          />
          <StepperInput
            label="End Age"
            value={params.endAge}
            onChange={(v) => updateParam('endAge', v)}
            min={params.retirementAge + 5} max={100} step={5} suffix="" decimals={0}
          />
          <StepperInput
            label="Current Wealth"
            value={params.initialWealth}
            onChange={(v) => updateParam('initialWealth', v)}
            min={0} max={5000} step={50} suffix="$k" decimals={0}
          />
          {/* Retirement Summary Box */}
          <div style={{
            background: '#e8f5e9',
            padding: '8px',
            borderRadius: '4px',
            fontSize: '11px',
            marginTop: '8px',
          }}>
            <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#2e7d32' }}>Timeline Summary</div>
            <div>Years Until Retirement: <strong>{params.retirementAge - params.startAge}</strong> years</div>
            <div>Retirement Duration: <strong>{params.endAge - params.retirementAge}</strong> years</div>
          </div>
        </ParamGroup>

        <ParamGroup title="Returns & Market">
          <StepperInput
            label="Risk-free rate"
            value={params.rBar * 100}
            onChange={(v) => updateParam('rBar', v / 100)}
            min={0} max={5} step={0.25} suffix="%" decimals={2}
          />
          <StepperInput
            label="Stock excess return (μ)"
            value={params.muStock * 100}
            onChange={(v) => updateParam('muStock', v / 100)}
            min={0} max={8} step={0.5} suffix="%" decimals={1}
          />
          <StepperInput
            label="Bond Sharpe ratio"
            value={params.bondSharpe}
            onChange={(v) => updateParam('bondSharpe', v)}
            min={0} max={0.5} step={0.01} suffix="" decimals={2}
          />
          <StepperInput
            label="Stock volatility (σ)"
            value={params.sigmaS * 100}
            onChange={(v) => updateParam('sigmaS', v / 100)}
            min={10} max={30} step={2} suffix="%" decimals={0}
          />
          <StepperInput
            label="Rate shock vol (σᵣ)"
            value={params.sigmaR * 100}
            onChange={(v) => updateParam('sigmaR', v / 100)}
            min={0.1} max={3} step={0.1} suffix="%" decimals={1}
          />
          <StepperInput
            label="Rate/stock corr (ρ)"
            value={params.rho}
            onChange={(v) => updateParam('rho', v)}
            min={-0.5} max={0.5} step={0.1} suffix="" decimals={1}
          />
          <StepperInput
            label="Bond duration"
            value={params.bondDuration}
            onChange={(v) => updateParam('bondDuration', v)}
            min={1} max={30} step={1} suffix="yrs" decimals={0}
          />
          <StepperInput
            label="Max duration cap"
            value={params.maxDuration ?? 0}
            onChange={(v) => setParams(prev => ({ ...prev, maxDuration: v === 0 ? undefined : v }))}
            min={0} max={50} step={5} suffix={params.maxDuration ? 'yrs' : ''} decimals={0}
          />
        </ParamGroup>

        <ParamGroup title="Income">
          <StepperInput
            label="Initial earnings"
            value={params.initialEarnings}
            onChange={(v) => updateParam('initialEarnings', v)}
            min={50} max={200} step={25} suffix="$k" decimals={0}
          />
          <StepperInput
            label="Earnings growth"
            value={params.earningsGrowth * 100}
            onChange={(v) => updateParam('earningsGrowth', v / 100)}
            min={0} max={4} step={0.5} suffix="%" decimals={1}
          />
          <StepperInput
            label="Peak earnings age"
            value={params.earningsHumpAge}
            onChange={(v) => updateParam('earningsHumpAge', v)}
            min={40} max={60} step={5} suffix="" decimals={0}
          />
        </ParamGroup>

        <ParamGroup title="Expenses">
          <StepperInput
            label="Working expenses"
            value={params.baseExpenses}
            onChange={(v) => updateParam('baseExpenses', v)}
            min={30} max={100} step={10} suffix="$k" decimals={0}
          />
          <StepperInput
            label="Expense growth"
            value={params.expenseGrowth * 100}
            onChange={(v) => updateParam('expenseGrowth', v / 100)}
            min={0} max={3} step={0.5} suffix="%" decimals={1}
          />
          <StepperInput
            label="Retirement expenses"
            value={params.retirementExpenses}
            onChange={(v) => updateParam('retirementExpenses', v)}
            min={40} max={120} step={10} suffix="$k" decimals={0}
          />
        </ParamGroup>

        <ParamGroup title="Risk Preferences">
          <StepperInput
            label="Risk aversion (γ)"
            value={params.gamma}
            onChange={(v) => updateParam('gamma', v)}
            min={1} max={10} step={1} suffix="" decimals={0}
          />
          <StepperInput
            label="HC stock beta"
            value={params.stockBetaHC}
            onChange={(v) => updateParam('stockBetaHC', v)}
            min={0} max={0.5} step={0.1} suffix="" decimals={1}
          />
          <StepperInput
            label="Consumption boost"
            value={params.consumptionBoost * 100}
            onChange={(v) => updateParam('consumptionBoost', v / 100)}
            min={0} max={5} step={0.5} suffix="%" decimals={1}
          />
        </ParamGroup>

        {/* Target allocation summary */}
        <div style={{
          background: '#e8f4f8',
          padding: '8px',
          borderRadius: '4px',
          fontSize: '11px',
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>MV Target Allocation:</div>
          <div>Stocks: {Math.round(result.targetStock * 100)}%</div>
          <div>Bonds: {Math.round(result.targetBond * 100)}%</div>
          <div>Cash: {Math.round(result.targetCash * 100)}%</div>
        </div>

        {/* Debug button for verification */}
        <button
          onClick={() => exportVerificationData(params)}
          style={{
            marginTop: '16px',
            padding: '8px 12px',
            background: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            fontSize: '11px',
            cursor: 'pointer',
            width: '100%',
          }}
        >
          Export Verification Data
        </button>
      </div>

      {/* Main Content - Charts */}
      <div style={{
        flex: 1,
        padding: '24px',
        overflowY: 'auto',
        background: '#f5f5f5',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '16px' }}>
          <h1 style={{ fontSize: '18px', fontWeight: 'bold', color: '#2c3e50', margin: 0 }}>
            Lifecycle Path Visualizer
          </h1>

          {/* Tab Navigation */}
          <div style={{ display: 'flex', gap: '4px', background: '#e0e0e0', padding: '4px', borderRadius: '8px' }}>
            <button
              onClick={() => setCurrentPage('base')}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: currentPage === 'base' ? 'bold' : 'normal',
                background: currentPage === 'base' ? '#fff' : 'transparent',
                color: currentPage === 'base' ? '#2c3e50' : '#666',
                cursor: 'pointer',
                boxShadow: currentPage === 'base' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              }}
            >
              Base Case
            </button>
            <button
              onClick={() => setCurrentPage('oneDraw')}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: currentPage === 'oneDraw' ? 'bold' : 'normal',
                background: currentPage === 'oneDraw' ? '#fff' : 'transparent',
                color: currentPage === 'oneDraw' ? '#2c3e50' : '#666',
                cursor: 'pointer',
                boxShadow: currentPage === 'oneDraw' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              }}
            >
              One Draw
            </button>
            <button
              onClick={() => setCurrentPage('scenarios')}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: currentPage === 'scenarios' ? 'bold' : 'normal',
                background: currentPage === 'scenarios' ? '#fff' : 'transparent',
                color: currentPage === 'scenarios' ? '#2c3e50' : '#666',
                cursor: 'pointer',
                boxShadow: currentPage === 'scenarios' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              }}
            >
              Teaching Scenarios
            </button>
          </div>
        </div>

        {currentPage === 'base' && (
          <>
        {/* Section 1: Assumptions */}
        <ChartSection title="Section 1: Assumptions">
          <ChartCard title="Earnings & Expenses Profile ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="earnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="Earnings" />
                <Line type="monotone" dataKey="expenses" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="Expenses" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Cumulative Stock Returns (Arithmetic vs Geometric)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} scale="log" domain={['auto', 'auto']} tickFormatter={(v: number) => `${v.toFixed(1)}x`} />
                <Tooltip formatter={(value: number | undefined) => value !== undefined ? `${value.toFixed(2)}x` : ''} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={1} stroke="#999" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="cumulativeStockReturnArithmetic" stroke={COLORS.stock} strokeWidth={2} strokeDasharray="5 5" dot={false}
                  name={`Arithmetic: r+μ = ${((params.rBar + params.muStock) * 100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="cumulativeStockReturn" stroke={COLORS.stock} strokeWidth={2} dot={false}
                  name={`Geometric: r+μ−½σ² = ${((params.rBar + params.muStock - 0.5 * params.sigmaS * params.sigmaS) * 100).toFixed(1)}%`} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px', color: '#666' }}>
              Jensen's correction: −½σ² = {(-0.5 * params.sigmaS * params.sigmaS * 100).toFixed(2)}% per year (log scale).
            </div>
          </ChartCard>

          <ChartCard title="Interest Rate (%)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={(v: number) => `${v.toFixed(1)}%`} />
                <Tooltip formatter={percentTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="interestRate" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Interest Rate" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </ChartSection>

        {/* Section 2: Forward-Looking Values */}
        <ChartSection title="Section 2: Forward-Looking Values">
          <ChartCard title="Present Values ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="pvEarnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="PV Earnings" />
                <Line type="monotone" dataKey="pvExpenses" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="PV Expenses" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Durations (years)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatYears} />
                <Tooltip formatter={yearsTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="durationEarnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="Duration (Earnings)" />
                <Line type="monotone" dataKey="durationExpenses" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="Duration (Expenses)" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </ChartSection>

        {/* Section 3: Wealth */}
        <ChartSection title="Section 3: Wealth">
          <ChartCard title="Human Capital vs Financial Wealth ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area type="monotone" dataKey="financialWealth" stackId="1" stroke={COLORS.fw} fill={COLORS.fw} name="Financial Wealth" />
                <Area type="monotone" dataKey="humanCapital" stackId="1" stroke={COLORS.hc} fill={COLORS.hc} name="Human Capital" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Human Capital Decomposition ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="hcCash" stroke={COLORS.cash} strokeWidth={2} dot={false} name="HC Cash" />
                <Line type="monotone" dataKey="hcBond" stroke={COLORS.bond} strokeWidth={2} dot={false} name="HC Bond" />
                <Line type="monotone" dataKey="hcStock" stroke={COLORS.stock} strokeWidth={2} dot={false} name="HC Stock" />
                <Line type="monotone" dataKey="humanCapital" stroke="#333" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Total HC" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Expense Liability Decomposition ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="expCash" stroke={COLORS.cash} strokeWidth={2} dot={false} name="Expense Cash" />
                <Line type="monotone" dataKey="expBond" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Expense Bond" />
                <Line type="monotone" dataKey="pvExpenses" stroke="#333" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Total Expenses" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Net HC minus Expenses ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line type="monotone" dataKey="netCash" stroke={COLORS.cash} strokeWidth={2} dot={false} name="Net Cash" />
                <Line type="monotone" dataKey="netBond" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Net Bond" />
                <Line type="monotone" dataKey="netStock" stroke={COLORS.stock} strokeWidth={2} dot={false} name="Net Stock" />
                <Line type="monotone" dataKey="netTotal" stroke="#333" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Net Total" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Net Wealth: HC + FW − Expenses ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={0} stroke="#333" strokeWidth={1.5} />
                <ReferenceLine x={params.retirementAge} stroke="#999" strokeDasharray="3 3" />
                <Area type="monotone" dataKey="netWorth" stroke={COLORS.fw} fill={COLORS.fw} fillOpacity={0.4} name="Net Worth" />
                <Line type="monotone" dataKey="totalAssets" stroke="#333" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Total Assets (HC+FW)" />
                <Line type="monotone" dataKey="pvExpenses" stroke={COLORS.expenses} strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="PV Expenses" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </ChartSection>

        {/* Section 4: Choices */}
        <ChartSection title="Section 4: Choices (Optimal Decisions)">
          <ChartCard title="Consumption Path ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area type="monotone" dataKey="subsistence" stackId="1" stroke={COLORS.subsistence} fill={COLORS.subsistence} name="Subsistence" />
                <Area type="monotone" dataKey="variable" stackId="1" stroke={COLORS.variable} fill={COLORS.variable} name="Variable" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Portfolio Allocation (%)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} domain={[0, 100]} tickFormatter={formatPercent} />
                <Tooltip formatter={percentTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Area type="monotone" dataKey="cashWeight" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Cash" />
                <Area type="monotone" dataKey="bondWeight" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Bonds" />
                <Area type="monotone" dataKey="stockWeight" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="Stocks" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Earnings vs Consumption ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip content={<EarningsConsumptionTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={0} stroke="#333" strokeWidth={1} />
                <ReferenceLine x={params.retirementAge} stroke="#999" strokeDasharray="3 3" />
                <Area type="monotone" dataKey="savingsPositive" stroke="transparent" fill={COLORS.cash} fillOpacity={0.4} name="Savings" />
                <Area type="monotone" dataKey="savingsNegative" stroke="transparent" fill={COLORS.expenses} fillOpacity={0.4} name="Drawdown" />
                <Line type="monotone" dataKey="earnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="Earnings" />
                <Line type="monotone" dataKey="totalConsumption" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="Total Consumption" />
              </AreaChart>
            </ResponsiveContainer>
            <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px', color: '#666' }}>
              Green fill = savings (earnings {'>'} consumption). Orange fill = drawdown (consumption {'>'} earnings).
            </div>
          </ChartCard>
        </ChartSection>

        {/* Section 5: Risk Management */}
        <ChartSection title="Section 5: Risk Management">
          <ChartCard title="Net Fixed Income PV ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={0} stroke="#333" strokeWidth={1.5} />
                <ReferenceLine x={params.retirementAge} stroke="#999" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="netFiPv" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Net FI PV" />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px', color: '#666' }}>
              Net FI PV = Bond Holdings + HC Bond - Expense Bond. Zero = perfectly hedged.
            </div>
          </ChartCard>

          <ChartCard title="Interest Rate Sensitivity (DV01)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={(v: number) => `$${Math.round(v)}`} />
                <Tooltip formatter={(v: number | undefined) => v !== undefined ? `$${Math.round(v)}k` : ''} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={0} stroke="#333" strokeWidth={1.5} />
                <ReferenceLine x={params.retirementAge} stroke="#999" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="dv01" stroke={COLORS.bond} strokeWidth={2} dot={false} name="DV01" />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px', color: '#666' }}>
              Dollar value change per 1pp rate move. Zero = duration matched.
            </div>
          </ChartCard>

          <ChartCard title="Target Financial Portfolio ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <ReferenceLine y={0} stroke="#333" strokeWidth={1} />
                <ReferenceLine x={params.retirementAge} stroke="#999" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="targetFinStock" stroke={COLORS.stock} strokeWidth={2} dot={false} name="Target Stocks ($)" />
                <Line type="monotone" dataKey="targetFinBond" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Target Bonds ($)" />
                <Line type="monotone" dataKey="targetFinCash" stroke={COLORS.cash} strokeWidth={2} dot={false} name="Target Cash ($)" />
                <Line type="monotone" dataKey="financialWealth" stroke="#333" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Financial Wealth" />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px', color: '#666' }}>
              Target = MV weight * surplus - HC component + expense component. Sum = FW (no leverage).
            </div>
          </ChartCard>
        </ChartSection>
          </>
        )}

        {currentPage === 'oneDraw' && (
          <>
            {/* Control Bar */}
            <div style={{
              background: '#fff',
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              padding: '16px',
              marginBottom: '24px',
            }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px', color: '#2c3e50' }}>
                One Random Market Scenario
              </div>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '12px' }}>
                Draw one random future and see how the LDI strategy responds. Each draw is a complete 70-year market history.
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                <button
                  onClick={() => {
                    const seed = oneDrawSeedInput.trim() !== ''
                      ? parseInt(oneDrawSeedInput.trim(), 10)
                      : Math.floor(Math.random() * 100000);
                    if (isNaN(seed)) return;
                    setOneDrawComputing(true);
                    setPendingOneDrawParams({ lifecycleParams, econParams, seed });
                    setTimeout(() => {
                      setOneDrawVersion(v => v + 1);
                    }, 50);
                  }}
                  disabled={oneDrawComputing}
                  style={{
                    padding: '10px 24px',
                    background: oneDrawComputing ? '#ccc' : '#27ae60',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: oneDrawComputing ? 'wait' : 'pointer',
                    fontWeight: 'bold',
                    fontSize: '14px',
                  }}
                >
                  {oneDrawComputing ? 'Drawing...' : 'New Draw'}
                </button>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <label style={{ fontSize: '12px', color: '#666' }}>Seed (optional):</label>
                  <input
                    type="text"
                    value={oneDrawSeedInput}
                    onChange={(e) => setOneDrawSeedInput(e.target.value)}
                    placeholder="random"
                    style={{
                      width: '80px',
                      padding: '4px 8px',
                      border: '1px solid #ccc',
                      borderRadius: '4px',
                      fontSize: '12px',
                    }}
                  />
                </div>
                {cachedOneDraw && (
                  <div style={{ fontSize: '12px', color: '#888' }}>
                    Current seed: <strong>{cachedOneDraw.seed}</strong>
                  </div>
                )}
              </div>
            </div>

            {/* Empty state */}
            {!cachedOneDraw && !oneDrawComputing && (
              <div style={{ textAlign: 'center', padding: '60px', color: '#666' }}>
                <div style={{ fontSize: '16px', marginBottom: '8px' }}>
                  Click "New Draw" to generate a random market scenario
                </div>
                <div style={{ fontSize: '13px' }}>
                  Each draw simulates one complete market history with random interest rate and stock return shocks
                </div>
              </div>
            )}
            {oneDrawComputing && !cachedOneDraw && (
              <div style={{ textAlign: 'center', padding: '60px', color: '#666' }}>
                Computing...
              </div>
            )}

            {/* Charts */}
            {oneDrawChartData && cachedOneDraw && (() => {
              const retAge = params.retirementAge;
              const ldiRes = cachedOneDraw.ldiResult;
              const fwArr = ldiRes.financialWealth as number[];
              const peakWealth = Math.max(...fwArr);
              const peakAge = ldiRes.ages[fwArr.indexOf(peakWealth)];
              const terminalWealth = fwArr[fwArr.length - 1];
              const defaulted = ldiRes.defaulted as boolean;
              const defaultAge = ldiRes.defaultAge as number | null;

              return (
                <>
              {/* Section 1: The Market You Drew */}
              <ChartSection title="Section 1: The Market You Drew">
                <ChartCard title="Interest Rate Path (%)">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={(v: number) => `${v.toFixed(1)}%`} />
                      <Tooltip formatter={percentTooltipFormatter} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" label={{ value: 'Retire', fontSize: 10 }} />
                      <ReferenceLine y={params.rBar * 100} stroke="#999" strokeDasharray="5 5" label={{ value: `r̄=${(params.rBar * 100).toFixed(1)}%`, fontSize: 10, position: 'right' }} />
                      <Line type="monotone" dataKey="interestRate" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Interest Rate" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Cumulative Stock Return (Growth of $1, Log Scale)">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} scale="log" domain={['auto', 'auto']} allowDataOverflow tickFormatter={(v: number) => `${v.toFixed(1)}x`} />
                      <Tooltip formatter={(value: number | undefined) => value !== undefined ? `${value.toFixed(2)}x` : ''} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <ReferenceLine y={1} stroke="#999" strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="cumStockReturn" stroke={COLORS.stock} strokeWidth={2} dot={false} name="Stocks" />
                      <Line type="monotone" dataKey="cumBondReturn" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Bonds" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              </ChartSection>

              {/* Section 2: LDI Advice */}
              <ChartSection title="Section 2: LDI Advice for This Draw">
                <ChartCard title="Human Capital + Financial Wealth ($k)">
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={formatDollar} />
                      <Tooltip formatter={dollarTooltipFormatter} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <Area type="monotone" dataKey="financialWealth" stackId="1" stroke={COLORS.fw} fill={COLORS.fw} fillOpacity={0.6} name="Financial Wealth" />
                      <Area type="monotone" dataKey="humanCapital" stackId="1" stroke={COLORS.hc} fill={COLORS.hc} fillOpacity={0.6} name="Human Capital" />
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Portfolio Allocation (%)">
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} domain={[0, 100]} tickFormatter={formatPercent} />
                      <Tooltip formatter={percentTooltipFormatter} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <Area type="monotone" dataKey="cashWeight" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Cash" />
                      <Area type="monotone" dataKey="bondWeight" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Bonds" />
                      <Area type="monotone" dataKey="stockWeight" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="Stocks" />
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Consumption Path ($k)">
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={formatDollar} />
                      <Tooltip formatter={dollarTooltipFormatter} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <Area type="monotone" dataKey="subsistence" stackId="1" stroke={COLORS.subsistence} fill={COLORS.subsistence} fillOpacity={0.6} name="Subsistence" />
                      <Area type="monotone" dataKey="variable" stackId="1" stroke={COLORS.variable} fill={COLORS.variable} fillOpacity={0.6} name="Variable" />
                      <Line type="monotone" dataKey="baseMedianConsumption" stroke="#999" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Base Case Median" />
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Financial Wealth ($k)">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={formatDollar} />
                      <Tooltip formatter={dollarTooltipFormatter} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="financialWealth" stroke={COLORS.fw} strokeWidth={2} dot={false} name="Realized FW" />
                      <Line type="monotone" dataKey="baseMedianFW" stroke="#999" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Base Case Median" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Net Wealth: HC + FW − Expenses ($k)">
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={formatDollar} />
                      <Tooltip formatter={dollarTooltipFormatter} />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <ReferenceLine y={0} stroke="#333" strokeWidth={1.5} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <Area type="monotone" dataKey="netWorth" stroke={COLORS.fw} fill={COLORS.fw} fillOpacity={0.4} name="Net Worth" />
                      <Line type="monotone" dataKey="totalAssets" stroke="#333" strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="Total Assets (HC+FW)" />
                      <Line type="monotone" dataKey="pvExpenses" stroke={COLORS.expenses} strokeWidth={1.5} strokeDasharray="5 5" dot={false} name="PV Expenses" />
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartCard>
              </ChartSection>

              {/* Section 3: Rebalancing & Outcomes */}
              <ChartSection title="Section 3: Rebalancing & Outcomes">
                <ChartCard title="Stock Rebalancing (% of portfolio)">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={oneDrawChartData.slice(1)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
                      <Tooltip formatter={(value: number | undefined) => value !== undefined ? `${value.toFixed(1)}%` : ''} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <ReferenceLine y={0} stroke="#666" strokeWidth={1} />
                      <Bar dataKey="stockPurchasePct" name="Stock Purchase">
                        {oneDrawChartData.slice(1).map((entry, index) => (
                          <Cell key={index} fill={(entry.stockPurchasePct ?? 0) >= 0 ? '#1abc9c' : COLORS.expenses} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                    Teal = buying | Orange = selling
                  </div>
                </ChartCard>

                <ChartCard title="Bond Rebalancing (% of portfolio)">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={oneDrawChartData.slice(1)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
                      <Tooltip formatter={(value: number | undefined) => value !== undefined ? `${value.toFixed(1)}%` : ''} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <ReferenceLine y={0} stroke="#666" strokeWidth={1} />
                      <Bar dataKey="bondPurchasePct" name="Bond Purchase">
                        {oneDrawChartData.slice(1).map((entry, index) => (
                          <Cell key={index} fill={(entry.bondPurchasePct ?? 0) >= 0 ? '#1abc9c' : COLORS.expenses} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                    Teal = buying | Orange = selling
                  </div>
                </ChartCard>

                <ChartCard title="Cash Flow: Earnings minus Consumption ($k)">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={oneDrawChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="age" fontSize={11} />
                      <YAxis fontSize={11} tickFormatter={formatDollar} />
                      <Tooltip formatter={dollarTooltipFormatter} />
                      <ReferenceLine x={retAge} stroke="#999" strokeDasharray="3 3" />
                      <ReferenceLine y={0} stroke="#666" strokeWidth={1} />
                      <Bar dataKey="cashFlow" name="Cash Flow">
                        {oneDrawChartData.map((entry, index) => (
                          <Cell key={index} fill={entry.cashFlow >= 0 ? COLORS.cash : COLORS.stock} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Summary Statistics">
                  <div style={{ padding: '16px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: '#888' }}>Terminal Wealth</div>
                        <div style={{ fontSize: '22px', fontWeight: 'bold', color: terminalWealth > 0 ? COLORS.cash : COLORS.stock }}>
                          ${Math.round(terminalWealth)}k
                        </div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: '#888' }}>Peak Wealth</div>
                        <div style={{ fontSize: '22px', fontWeight: 'bold', color: COLORS.fw }}>
                          ${Math.round(peakWealth)}k
                        </div>
                        <div style={{ fontSize: '11px', color: '#888' }}>at age {peakAge}</div>
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: '#888' }}>Default?</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: defaulted ? COLORS.stock : COLORS.cash }}>
                          {defaulted ? `Yes (age ${defaultAge})` : 'No'}
                        </div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '11px', color: '#888' }}>Seed</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#666' }}>
                          {cachedOneDraw.seed}
                        </div>
                      </div>
                    </div>
                    <div style={{ borderTop: '1px solid #eee', paddingTop: '12px' }}>
                      <div style={{ fontSize: '11px', color: '#888', marginBottom: '8px' }}>MV Optimal Targets</div>
                      <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', marginBottom: '8px' }}>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '11px', color: COLORS.stock }}>Stocks</div>
                          <div style={{ fontWeight: 'bold' }}>{(result.targetStock * 100).toFixed(0)}%</div>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '11px', color: COLORS.bond }}>Bonds</div>
                          <div style={{ fontWeight: 'bold' }}>{(result.targetBond * 100).toFixed(0)}%</div>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '11px', color: COLORS.cash }}>Cash</div>
                          <div style={{ fontWeight: 'bold' }}>{(result.targetCash * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                      {(() => {
                        // Compute median portfolio return (same formula as LDI strategy)
                        const r = econParams.rBar;
                        const wS = result.targetStock;
                        const wB = result.targetBond;
                        const wC = result.targetCash;
                        const muBondVal = computeMuBondFromEcon(econParams);
                        const expectedReturn = wS * (r + econParams.muExcess) + wB * (r + muBondVal) + wC * r;
                        const portfolioVar = computePortfolioVariance(
                          wS, wB, econParams.sigmaS, econParams.sigmaR, econParams.bondDuration, econParams.rho
                        );
                        const medianReturn = expectedReturn - 0.5 * portfolioVar;
                        return (
                          <div style={{ textAlign: 'center', fontSize: '12px', color: '#555' }}>
                            Median portfolio return: <strong>{(medianReturn * 100).toFixed(2)}%</strong>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                </ChartCard>
              </ChartSection>
                </>
              );
            })()}
          </>
        )}

        {currentPage === 'scenarios' && (
          <>
            {/* Scenario Selection */}
            <div style={{
              background: '#fff',
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              padding: '16px',
              marginBottom: '24px',
            }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px', color: '#2c3e50' }}>
                LDI vs Rule-of-Thumb: Strategy Comparison
              </div>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '12px' }}>
                Comparing Liability-Driven Investment (LDI) strategy vs Rule-of-Thumb (15% savings, 100-age stocks, 4% withdrawal)
              </div>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
                <button
                  onClick={() => setScenarioType('summary')}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderBottom: scenarioType === 'summary' ? '3px solid #27ae60' : '3px solid transparent',
                    background: scenarioType === 'summary' ? '#e8f8f5' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: scenarioType === 'summary' ? 'bold' : 'normal',
                    color: scenarioType === 'summary' ? '#27ae60' : '#666',
                  }}
                >
                  3-Scenario Summary
                </button>
                <button
                  onClick={() => setScenarioType('baseline')}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderBottom: scenarioType === 'baseline' ? '3px solid #3498db' : '3px solid transparent',
                    background: scenarioType === 'baseline' ? '#e8f4f8' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: scenarioType === 'baseline' ? 'bold' : 'normal',
                    color: scenarioType === 'baseline' ? '#2c3e50' : '#666',
                  }}
                >
                  Baseline (Normal)
                </button>
                <button
                  onClick={() => setScenarioType('sequenceRisk')}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderBottom: scenarioType === 'sequenceRisk' ? '3px solid #e74c3c' : '3px solid transparent',
                    background: scenarioType === 'sequenceRisk' ? '#ffebee' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: scenarioType === 'sequenceRisk' ? 'bold' : 'normal',
                    color: scenarioType === 'sequenceRisk' ? '#c0392b' : '#666',
                  }}
                >
                  Sequence Risk
                </button>
                <button
                  onClick={() => setScenarioType('rateShock')}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderBottom: scenarioType === 'rateShock' ? '3px solid #f39c12' : '3px solid transparent',
                    background: scenarioType === 'rateShock' ? '#fff8e1' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: scenarioType === 'rateShock' ? 'bold' : 'normal',
                    color: scenarioType === 'rateShock' ? '#d68910' : '#666',
                  }}
                >
                  Rate Shock
                </button>
              </div>

              {/* Age Controls */}
              <div style={{ display: 'flex', gap: '24px', padding: '12px', background: '#f5f5f5', borderRadius: '6px', marginBottom: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>Retirement Age:</div>
                  <input
                    type="number"
                    value={scenarioRetirementAge}
                    onChange={(e) => setScenarioRetirementAge(parseInt(e.target.value) || 65)}
                    min={params.startAge + 10}
                    max={scenarioEndAge - 5}
                    style={{ width: '60px', padding: '4px 8px', border: '1px solid #ccc', borderRadius: '4px' }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>End Age (Death):</div>
                  <input
                    type="number"
                    value={scenarioEndAge}
                    onChange={(e) => setScenarioEndAge(parseInt(e.target.value) || 85)}
                    min={scenarioRetirementAge + 5}
                    max={100}
                    style={{ width: '60px', padding: '4px 8px', border: '1px solid #ccc', borderRadius: '4px' }}
                  />
                </div>
                {scenarioType === 'sequenceRisk' && (
                  <div style={{ fontSize: '12px', color: '#c0392b', fontStyle: 'italic' }}>
                    Stocks return ~-5% avg/year for first 10 years of retirement.
                  </div>
                )}
              </div>

              {scenarioType === 'rateShock' && (
                <div style={{ display: 'flex', gap: '24px', padding: '12px', background: '#fff3cd', borderRadius: '6px' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>Rate shock at age:</div>
                    <input
                      type="number"
                      value={rateShockAge}
                      onChange={(e) => setRateShockAge(parseInt(e.target.value) || 50)}
                      min={params.startAge}
                      max={scenarioEndAge - 1}
                      style={{ width: '60px', padding: '4px 8px', border: '1px solid #ccc', borderRadius: '4px' }}
                    />
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>Rate change (%):</div>
                    <input
                      type="number"
                      value={rateShockMagnitude * 100}
                      onChange={(e) => setRateShockMagnitude((parseFloat(e.target.value) || 0) / 100)}
                      step={0.5}
                      style={{ width: '60px', padding: '4px 8px', border: '1px solid #ccc', borderRadius: '4px' }}
                    />
                  </div>
                </div>
              )}

              {/* Run Simulation Button */}
              <div style={{ marginTop: '16px' }}>
                <button
                  onClick={() => {
                    setScenarioComputing(true);
                    // Capture current params at click time (before async computation)
                    setPendingSimulationParams({
                      lifecycleParams,
                      econParams,
                      rateShockMagnitude,
                    });
                    // Use setTimeout to allow UI to update before computation
                    // Increment simulationVersion to trigger useEffect computation
                    setTimeout(() => {
                      setSimulationVersion(v => v + 1);
                      // Note: setScenarioComputing(false) is called in the useEffect after computation
                    }, 50);
                  }}
                  disabled={scenarioComputing}
                  style={{
                    padding: '10px 24px',
                    background: scenarioComputing ? '#ccc' : '#27ae60',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: scenarioComputing ? 'wait' : 'pointer',
                    fontWeight: 'bold',
                    fontSize: '14px',
                  }}
                >
                  {scenarioComputing ? 'Running...' : 'Run Simulation'}
                </button>
              </div>
            </div>

            {/* Concept Explanation */}
            {scenarioType === 'baseline' && (
              <div style={{
                background: '#e3f2fd',
                border: '1px solid #2196f3',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '24px',
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#1565c0' }}>
                  Baseline Scenario: Normal Monte Carlo
                </div>
                <div style={{ fontSize: '13px', color: '#1565c0', lineHeight: 1.5 }}>
                  <p style={{ margin: '0 0 8px 0' }}>
                    <strong>Market Conditions:</strong> Standard stochastic returns with normal random shocks. This represents typical market behavior without forced scenarios.
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div>
                      <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>LDI Strategy:</p>
                      <ul style={{ margin: 0, paddingLeft: '16px' }}>
                        <li>Adaptive consumption based on net worth</li>
                        <li>Portfolio accounts for human capital</li>
                        <li>Duration matching for liabilities</li>
                        <li>Never defaults on subsistence</li>
                      </ul>
                    </div>
                    <div>
                      <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>Rule of Thumb:</p>
                      <ul style={{ margin: 0, paddingLeft: '16px' }}>
                        <li>Save 15% of income</li>
                        <li>Fixed target duration = 6 years</li>
                        <li>4% withdrawal rule in retirement</li>
                        <li>No adaptive consumption</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {scenarioType === 'sequenceRisk' && (
              <div style={{
                background: '#fff3cd',
                border: '1px solid #ffc107',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '24px',
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#856404' }}>
                  Concept: Sequence of Returns Risk & the 4% Rule
                </div>
                <div style={{ fontSize: '13px', color: '#856404', lineHeight: 1.5 }}>
                  <p style={{ margin: '0 0 8px 0' }}>
                    <strong>Setup:</strong> Both strategies have identical behavior during working years (same consumption, same savings, same wealth at retirement).
                    The only difference is what happens after retirement.
                  </p>
                  <p style={{ margin: '0 0 8px 0' }}>
                    <strong>The 4% Rule:</strong> At retirement, calculate 4% of your financial portfolio. Withdraw that fixed dollar amount each year, regardless of how the market performs.
                  </p>
                  <p style={{ margin: '0 0 8px 0' }}>
                    <strong>The Problem:</strong> If markets perform poorly in the first years of retirement, you're withdrawing fixed amounts from a shrinking portfolio.
                    This "sequence of returns risk" means even if returns recover later, you may have already depleted too much capital.
                  </p>
                  <p style={{ margin: 0 }}>
                    <strong>Adaptive Consumption:</strong> By adjusting consumption based on current net worth, you protect subsistence spending.
                    When wealth drops, variable consumption drops too—but you never default on essentials.
                  </p>
                </div>
              </div>
            )}

            {scenarioType === 'rateShock' && (
              <div style={{
                background: '#d4edda',
                border: '1px solid #28a745',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '24px',
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#155724' }}>
                  Concept: Duration Matching & Interest Rate Risk
                </div>
                <div style={{ fontSize: '13px', color: '#155724', lineHeight: 1.5 }}>
                  <p style={{ margin: '0 0 8px 0' }}>
                    <strong>When rates fall:</strong> The present value of your future expenses increases (you need more money now to fund them).
                  </p>
                  <p style={{ margin: '0 0 8px 0' }}>
                    <strong>Without duration matching:</strong> If you hold short-duration assets (cash), they don't appreciate enough to offset the increased cost of future liabilities.
                  </p>
                  <p style={{ margin: 0 }}>
                    <strong>With duration matching:</strong> Long-duration bonds rise in value when rates fall, offsetting the increased liability.
                    Your net position is hedged against rate moves.
                  </p>
                </div>
              </div>
            )}

            {/* Loading/placeholder state for Summary tab */}
            {scenarioType === 'summary' && !teachingScenarios && (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                {scenarioComputing ? (
                  'Computing 500 simulation runs across 3 scenarios...'
                ) : (
                  <>
                    <div style={{ fontSize: '16px', marginBottom: '8px' }}>
                      Click "Run Simulation" to compute teaching scenarios
                    </div>
                    <div style={{ fontSize: '13px' }}>
                      This will run 500 Monte Carlo simulations for LDI vs Rule-of-Thumb comparison
                    </div>
                  </>
                )}
              </div>
            )}
            {/* Loading/placeholder for individual scenario tabs */}
            {(scenarioType === 'baseline' || scenarioType === 'sequenceRisk' || scenarioType === 'rateShock') && !teachingScenarios && (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                {scenarioComputing ? (
                  'Computing 500 simulation runs across 3 scenarios...'
                ) : (
                  <>
                    <div style={{ fontSize: '16px', marginBottom: '8px' }}>
                      Click "Run Simulation" to compute teaching scenarios
                    </div>
                    <div style={{ fontSize: '13px' }}>
                      This will run 500 Monte Carlo simulations for LDI vs Rule-of-Thumb comparison
                    </div>
                  </>
                )}
              </div>
            )}

            {/* 3-Scenario Summary - Matching teaching_scenarios.pdf */}
            {/* CRITICAL: This summary uses teachingScenarios.baseline/sequenceRisk/rateShock
                which is the SAME object used by the individual scenario tabs.
                Single source of truth: cachedTeachingScenarios computed in useEffect above. */}
            {scenarioType === 'summary' && teachingScenarios && (
              <div style={{
                opacity: simulationResultsStale ? 0.5 : 1,
                transition: 'opacity 0.3s ease',
                position: 'relative',
              }}>
                {/* Stale results warning banner */}
                {simulationResultsStale && (
                  <div style={{
                    background: '#fff3cd',
                    border: '1px solid #ffc107',
                    borderRadius: '8px',
                    padding: '12px 16px',
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}>
                    <span style={{ fontSize: '18px' }}>⚠️</span>
                    <div>
                      <div style={{ fontWeight: 'bold', color: '#856404', fontSize: '13px' }}>
                        Results are stale
                      </div>
                      <div style={{ color: '#856404', fontSize: '12px' }}>
                        Parameters have changed since the last simulation. Click "Run Simulation" to update.
                      </div>
                    </div>
                  </div>
                )}
                {/* Description */}
                <div style={{
                  background: '#e8f8f5',
                  border: '1px solid #27ae60',
                  borderRadius: '8px',
                  padding: '16px',
                  marginBottom: '24px',
                }}>
                  <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#27ae60' }}>
                    Teaching Scenarios: LDI vs Rule-of-Thumb
                  </div>
                  <div style={{ fontSize: '13px', color: '#1e8449', lineHeight: 1.5 }}>
                    <p style={{ margin: '0 0 8px 0' }}>
                      <strong>Baseline:</strong> Normal stochastic market conditions - shows typical strategy performance
                    </p>
                    <p style={{ margin: '0 0 8px 0' }}>
                      <strong>Sequence Risk:</strong> Forces bad stock returns (~-12%/yr) in first 5 years of retirement - tests vulnerability to early losses
                    </p>
                    <p style={{ margin: 0 }}>
                      <strong>Rate Shock:</strong> Interest rate drop (~4% cumulative) in 5 years before retirement - tests duration matching effectiveness
                    </p>
                  </div>
                </div>

                {/* Three Metrics Bar Charts */}
                <ChartSection title="Strategy Performance Across Scenarios">
                  {/* Default Rates Bar Chart */}
                  <ChartCard title="Default Rates (%)">
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={[
                          {
                            scenario: 'Baseline',
                            LDI: teachingScenarios.baseline.ldi.defaultRate * 100,
                            RoT: teachingScenarios.baseline.rot.defaultRate * 100,
                          },
                          {
                            scenario: 'Sequence\nRisk',
                            LDI: teachingScenarios.sequenceRisk.ldi.defaultRate * 100,
                            RoT: teachingScenarios.sequenceRisk.rot.defaultRate * 100,
                          },
                          {
                            scenario: 'Rate\nShock',
                            LDI: teachingScenarios.rateShock.ldi.defaultRate * 100,
                            RoT: teachingScenarios.rateShock.rot.defaultRate * 100,
                          },
                        ]}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="scenario" fontSize={11} />
                        <YAxis fontSize={11} domain={[0, 60]} tickFormatter={(v) => `${v}%`} />
                        <Tooltip formatter={(v: number | undefined) => v !== undefined ? [`${v.toFixed(1)}%`, ''] : ''} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Bar dataKey="LDI" fill="#2980b9" name="LDI" />
                        <Bar dataKey="RoT" fill="#d4a84c" name="RoT" />
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                      Lower is better. LDI consistently outperforms RoT across all scenarios.
                    </div>
                  </ChartCard>

                  {/* PV Consumption Bar Chart */}
                  <ChartCard title="Median PV Lifetime Consumption ($k)">
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={[
                          {
                            scenario: 'Baseline',
                            LDI: teachingScenarios.baseline.ldi.medianPvConsumption,
                            RoT: teachingScenarios.baseline.rot.medianPvConsumption,
                          },
                          {
                            scenario: 'Sequence\nRisk',
                            LDI: teachingScenarios.sequenceRisk.ldi.medianPvConsumption,
                            RoT: teachingScenarios.sequenceRisk.rot.medianPvConsumption,
                          },
                          {
                            scenario: 'Rate\nShock',
                            LDI: teachingScenarios.rateShock.ldi.medianPvConsumption,
                            RoT: teachingScenarios.rateShock.rot.medianPvConsumption,
                          },
                        ]}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="scenario" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={(v) => `$${Math.round(v)}k`} />
                        <Tooltip formatter={(v: number | undefined) => v !== undefined ? [`$${Math.round(v)}k`, ''] : ''} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Bar dataKey="LDI" fill="#2980b9" name="LDI" />
                        <Bar dataKey="RoT" fill="#d4a84c" name="RoT" />
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                      Higher is better. Present value of lifetime consumption discounted at risk-free rate.
                    </div>
                  </ChartCard>

                  {/* Terminal Wealth Bar Chart */}
                  <ChartCard title={`Median Terminal Wealth at Age ${scenarioEndAge} ($k)`}>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={[
                          {
                            scenario: 'Baseline',
                            LDI: Math.max(0, teachingScenarios.baseline.ldi.medianFinalWealth),
                            RoT: Math.max(0, teachingScenarios.baseline.rot.medianFinalWealth),
                          },
                          {
                            scenario: 'Sequence\nRisk',
                            LDI: Math.max(0, teachingScenarios.sequenceRisk.ldi.medianFinalWealth),
                            RoT: Math.max(0, teachingScenarios.sequenceRisk.rot.medianFinalWealth),
                          },
                          {
                            scenario: 'Rate\nShock',
                            LDI: Math.max(0, teachingScenarios.rateShock.ldi.medianFinalWealth),
                            RoT: Math.max(0, teachingScenarios.rateShock.rot.medianFinalWealth),
                          },
                        ]}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="scenario" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={(v) => `$${Math.round(v)}k`} />
                        <Tooltip formatter={(v: number | undefined) => v !== undefined ? [`$${Math.round(v)}k`, ''] : ''} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Bar dataKey="LDI" fill="#2980b9" name="LDI" />
                        <Bar dataKey="RoT" fill="#d4a84c" name="RoT" />
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                      Higher is better. RoT shows $0 in stress scenarios due to high default rates.
                    </div>
                  </ChartCard>
                </ChartSection>

                {/* Detailed Metrics Table */}
                <ChartSection title="Detailed Metrics Comparison">
                  <ChartCard title="All Metrics by Scenario and Strategy">
                    <div style={{ padding: '16px', overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                        <thead>
                          <tr style={{ background: '#f5f5f5' }}>
                            <th style={{ padding: '8px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Scenario</th>
                            <th style={{ padding: '8px', textAlign: 'center', borderBottom: '2px solid #ddd' }}>Strategy</th>
                            <th style={{ padding: '8px', textAlign: 'right', borderBottom: '2px solid #ddd' }}>Default Rate</th>
                            <th style={{ padding: '8px', textAlign: 'right', borderBottom: '2px solid #ddd' }}>PV Consumption</th>
                            <th style={{ padding: '8px', textAlign: 'right', borderBottom: '2px solid #ddd' }}>Terminal Wealth</th>
                          </tr>
                        </thead>
                        <tbody>
                          {/* Baseline */}
                          <tr>
                            <td rowSpan={2} style={{ padding: '8px', borderBottom: '1px solid #eee', verticalAlign: 'middle' }}>
                              <strong>Baseline</strong>
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'center', color: '#2980b9' }}>LDI</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              {(teachingScenarios.baseline.ldi.defaultRate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.baseline.ldi.medianPvConsumption)}k
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.baseline.ldi.medianFinalWealth)}k
                            </td>
                          </tr>
                          <tr>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'center', color: '#d4a84c' }}>RoT</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right' }}>
                              {(teachingScenarios.baseline.rot.defaultRate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.baseline.rot.medianPvConsumption)}k
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.baseline.rot.medianFinalWealth)}k
                            </td>
                          </tr>
                          {/* Sequence Risk */}
                          <tr>
                            <td rowSpan={2} style={{ padding: '8px', borderBottom: '1px solid #eee', verticalAlign: 'middle' }}>
                              <strong>Sequence Risk</strong>
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'center', color: '#2980b9' }}>LDI</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              {(teachingScenarios.sequenceRisk.ldi.defaultRate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.sequenceRisk.ldi.medianPvConsumption)}k
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.sequenceRisk.ldi.medianFinalWealth)}k
                            </td>
                          </tr>
                          <tr>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'center', color: '#d4a84c' }}>RoT</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right', color: teachingScenarios.sequenceRisk.rot.defaultRate > 0.3 ? '#e74c3c' : 'inherit' }}>
                              {(teachingScenarios.sequenceRisk.rot.defaultRate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.sequenceRisk.rot.medianPvConsumption)}k
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right', color: teachingScenarios.sequenceRisk.rot.medianFinalWealth < 1 ? '#e74c3c' : 'inherit' }}>
                              ${Math.round(teachingScenarios.sequenceRisk.rot.medianFinalWealth)}k
                            </td>
                          </tr>
                          {/* Rate Shock */}
                          <tr>
                            <td rowSpan={2} style={{ padding: '8px', borderBottom: '1px solid #eee', verticalAlign: 'middle' }}>
                              <strong>Rate Shock</strong>
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'center', color: '#2980b9' }}>LDI</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              {(teachingScenarios.rateShock.ldi.defaultRate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.rateShock.ldi.medianPvConsumption)}k
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.rateShock.ldi.medianFinalWealth)}k
                            </td>
                          </tr>
                          <tr>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'center', color: '#d4a84c' }}>RoT</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right', color: teachingScenarios.rateShock.rot.defaultRate > 0.3 ? '#e74c3c' : 'inherit' }}>
                              {(teachingScenarios.rateShock.rot.defaultRate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right' }}>
                              ${Math.round(teachingScenarios.rateShock.rot.medianPvConsumption)}k
                            </td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #ddd', textAlign: 'right', color: teachingScenarios.rateShock.rot.medianFinalWealth < 1 ? '#e74c3c' : 'inherit' }}>
                              ${Math.round(teachingScenarios.rateShock.rot.medianFinalWealth)}k
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </ChartCard>
                </ChartSection>

                {/* Key Takeaways */}
                <div style={{
                  background: '#2c3e50',
                  color: '#fff',
                  borderRadius: '8px',
                  padding: '20px',
                  marginTop: '24px',
                }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '12px' }}>
                    Key Takeaways
                  </div>
                  <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                    <li><strong>LDI has lower default rates</strong> across all scenarios - adaptive consumption protects subsistence</li>
                    <li><strong>Sequence Risk is most damaging to RoT</strong> - fixed 4% withdrawal from falling portfolio leads to ruin</li>
                    <li><strong>Rate Shock shows duration matching benefit</strong> - LDI hedges against rate changes</li>
                    <li><strong>PV Consumption is similar</strong> - LDI doesn't sacrifice much consumption for safety</li>
                    <li><strong>Terminal wealth varies</strong> - RoT can leave more in good scenarios but nothing in bad ones</li>
                  </ul>
                </div>
              </div>
            )}

            {/* 8-Panel Individual Scenario View - Matching teaching_scenarios.pdf */}
            {/* CRITICAL: This view uses the SAME teachingScenarios data as the Summary tab above.
                Both tabs access teachingScenarios.baseline (or sequenceRisk/rateShock).
                If you see different numbers between Summary and individual tabs, check:
                1. Browser cache - hard refresh (Cmd+Shift+R)
                2. The teachingScenarios object is the single source of truth
                3. Both tabs multiply defaultRate by 100 for percentage display */}
            {(scenarioType === 'baseline' || scenarioType === 'sequenceRisk' || scenarioType === 'rateShock') && teachingScenarios && (() => {
              // Get the scenario data based on scenarioType
              // This accesses the SAME object as Summary tab: teachingScenarios[scenarioKey]
              const scenarioKey = scenarioType as 'baseline' | 'sequenceRisk' | 'rateShock';
              const scenario = teachingScenarios[scenarioKey];
              const retirementIdx = scenarioRetirementAge - params.startAge;

              // Strategy colors - matching PDF
              const COLOR_LDI = '#1A759F';   // Deep blue
              const COLOR_ROT = '#E9C46A';   // Amber/gold
              const COLOR_RATES = '#3498db'; // Blue
              const COLOR_STOCKS = '#9b59b6'; // Purple

              // Prepare chart data from percentiles
              const ages = scenario.ldi.result.ages as number[];
              const nPeriods = ages.length;

              // Panel 1 & 2 data: Market conditions
              const marketData = ages.map((age, i) => ({
                age,
                // Cumulative stock returns (log scale for chart)
                sr_p5: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p5[i]),
                sr_p25: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p25[i]),
                sr_p50: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p50[i]),
                sr_p75: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p75[i]),
                sr_p95: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p95[i]),
                sr_band_5_25: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p25[i]) - Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p5[i]),
                sr_band_25_75: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p75[i]) - Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p25[i]),
                sr_band_75_95: Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p95[i]) - Math.log(scenario.ldi.percentiles.cumulativeStockReturns.p75[i]),
                // Interest rates (as percentage)
                rate_p5: scenario.ldi.percentiles.interestRates.p5[i] * 100,
                rate_p25: scenario.ldi.percentiles.interestRates.p25[i] * 100,
                rate_p50: scenario.ldi.percentiles.interestRates.p50[i] * 100,
                rate_p75: scenario.ldi.percentiles.interestRates.p75[i] * 100,
                rate_p95: scenario.ldi.percentiles.interestRates.p95[i] * 100,
                rate_band_5_25: (scenario.ldi.percentiles.interestRates.p25[i] - scenario.ldi.percentiles.interestRates.p5[i]) * 100,
                rate_band_25_75: (scenario.ldi.percentiles.interestRates.p75[i] - scenario.ldi.percentiles.interestRates.p25[i]) * 100,
                rate_band_75_95: (scenario.ldi.percentiles.interestRates.p95[i] - scenario.ldi.percentiles.interestRates.p75[i]) * 100,
              }));

              // Panel 3, 5, 6 data: Overlaid LDI vs RoT
              const wealthAllocationData = ages.map((age, i) => ({
                age,
                // Financial wealth - LDI (symlog scale)
                ldi_fw_p5: symlog(scenario.ldi.percentiles.financialWealth.p5[i]),
                ldi_fw_p25: symlog(scenario.ldi.percentiles.financialWealth.p25[i]),
                ldi_fw_p50: symlog(scenario.ldi.percentiles.financialWealth.p50[i]),
                ldi_fw_p75: symlog(scenario.ldi.percentiles.financialWealth.p75[i]),
                ldi_fw_p95: symlog(scenario.ldi.percentiles.financialWealth.p95[i]),
                // Financial wealth - RoT (symlog scale)
                rot_fw_p5: symlog(scenario.rot.percentiles.financialWealth.p5[i]),
                rot_fw_p25: symlog(scenario.rot.percentiles.financialWealth.p25[i]),
                rot_fw_p50: symlog(scenario.rot.percentiles.financialWealth.p50[i]),
                rot_fw_p75: symlog(scenario.rot.percentiles.financialWealth.p75[i]),
                rot_fw_p95: symlog(scenario.rot.percentiles.financialWealth.p95[i]),
                // Stock allocation - LDI (as %)
                ldi_stock_p5: scenario.ldi.percentiles.stockWeight.p5[i] * 100,
                ldi_stock_p25: scenario.ldi.percentiles.stockWeight.p25[i] * 100,
                ldi_stock_p50: scenario.ldi.percentiles.stockWeight.p50[i] * 100,
                ldi_stock_p75: scenario.ldi.percentiles.stockWeight.p75[i] * 100,
                ldi_stock_p95: scenario.ldi.percentiles.stockWeight.p95[i] * 100,
                // Stock allocation - RoT (as %)
                rot_stock_p5: scenario.rot.percentiles.stockWeight.p5[i] * 100,
                rot_stock_p25: scenario.rot.percentiles.stockWeight.p25[i] * 100,
                rot_stock_p50: scenario.rot.percentiles.stockWeight.p50[i] * 100,
                rot_stock_p75: scenario.rot.percentiles.stockWeight.p75[i] * 100,
                rot_stock_p95: scenario.rot.percentiles.stockWeight.p95[i] * 100,
                // Bond allocation - LDI (as %)
                ldi_bond_p5: scenario.ldi.percentiles.bondWeight.p5[i] * 100,
                ldi_bond_p25: scenario.ldi.percentiles.bondWeight.p25[i] * 100,
                ldi_bond_p50: scenario.ldi.percentiles.bondWeight.p50[i] * 100,
                ldi_bond_p75: scenario.ldi.percentiles.bondWeight.p75[i] * 100,
                ldi_bond_p95: scenario.ldi.percentiles.bondWeight.p95[i] * 100,
                // Bond allocation - RoT (as %)
                rot_bond_p5: scenario.rot.percentiles.bondWeight.p5[i] * 100,
                rot_bond_p25: scenario.rot.percentiles.bondWeight.p25[i] * 100,
                rot_bond_p50: scenario.rot.percentiles.bondWeight.p50[i] * 100,
                rot_bond_p75: scenario.rot.percentiles.bondWeight.p75[i] * 100,
                rot_bond_p95: scenario.rot.percentiles.bondWeight.p95[i] * 100,
              }));

              // Panel 4 data: Default timing histogram
              const ldiDefaultAges = (scenario.ldi.result.defaultAge as (number | null)[]).filter((a): a is number => a !== null);
              const rotDefaultAges = (scenario.rot.result.defaultAge as (number | null)[]).filter((a): a is number => a !== null);
              const binSize = 2;
              const minAge = scenarioRetirementAge;
              const maxAge = scenarioEndAge;
              const bins: { age: string; LDI: number; RoT: number }[] = [];
              for (let binStart = minAge; binStart < maxAge; binStart += binSize) {
                const binEnd = binStart + binSize;
                bins.push({
                  age: `${binStart}-${binEnd}`,
                  LDI: ldiDefaultAges.filter(a => a >= binStart && a < binEnd).length,
                  RoT: rotDefaultAges.filter(a => a >= binStart && a < binEnd).length,
                });
              }

              // Panel 7 data: Terminal wealth histogram
              const ldiTerminalWealth = scenario.ldi.result.finalWealth as number[];
              const rotTerminalWealth = scenario.rot.result.finalWealth as number[];
              const wealthFloor = 10;
              const ldiWealthFloored = ldiTerminalWealth.map(w => Math.max(w, wealthFloor));
              const rotWealthFloored = rotTerminalWealth.map(w => Math.max(w, wealthFloor));
              const wealthMax = Math.max(
                computePercentile(ldiWealthFloored, 99),
                computePercentile(rotWealthFloored, 99)
              );
              const wealthBins = geomSpacedEdges(wealthFloor, wealthMax, 15);
              const wealthHistData = createGeomHistogramBins(
                ldiWealthFloored, rotWealthFloored, wealthBins,
                (lo) => `$${Math.round(lo)}k`
              );

              // Panel 8 data: PV consumption histogram
              const ldiPvConsumption = scenario.ldi.pvConsumption;
              const rotPvConsumption = scenario.rot.pvConsumption;
              const pvFloor = 100;
              const ldiPvFloored = ldiPvConsumption.map(pv => Math.max(pv, pvFloor));
              const rotPvFloored = rotPvConsumption.map(pv => Math.max(pv, pvFloor));
              const pvMin = Math.min(
                Math.min(...ldiPvFloored),
                Math.min(...rotPvFloored)
              );
              const pvMax = Math.max(
                computePercentile(ldiPvFloored, 99),
                computePercentile(rotPvFloored, 99)
              );
              const pvBins = geomSpacedEdges(pvMin, pvMax, 15);
              const pvHistData = createGeomHistogramBins(
                ldiPvFloored, rotPvFloored, pvBins,
                (lo) => `$${Math.round(lo / 1000)}M`
              );

              // Panel 9 data: Net FI PV (median path comparison)
              const netFiPvData = ages.map((age, i) => ({
                age,
                ldi_p50: scenario.ldi.percentiles.netFiPv.p50[i],
                rot_p50: scenario.rot.percentiles.netFiPv.p50[i],
                ldi_p5: scenario.ldi.percentiles.netFiPv.p5[i],
                ldi_p95: scenario.ldi.percentiles.netFiPv.p95[i],
                rot_p5: scenario.rot.percentiles.netFiPv.p5[i],
                rot_p95: scenario.rot.percentiles.netFiPv.p95[i],
              }));

              // Panel 10 data: DV01 (interest rate sensitivity)
              const dv01Data = ages.map((age, i) => ({
                age,
                ldi_p50: scenario.ldi.percentiles.dv01.p50[i],
                rot_p50: scenario.rot.percentiles.dv01.p50[i],
                ldi_p5: scenario.ldi.percentiles.dv01.p5[i],
                ldi_p95: scenario.ldi.percentiles.dv01.p95[i],
                rot_p5: scenario.rot.percentiles.dv01.p5[i],
                rot_p95: scenario.rot.percentiles.dv01.p95[i],
              }));

              // Panel 11 data: Net Wealth (HC + FW - PV Expenses)
              const netWealthData = ages.map((age, i) => ({
                age,
                ldi_nw_p5: symlog(scenario.ldi.percentiles.netWorth.p5[i]),
                ldi_nw_p50: symlog(scenario.ldi.percentiles.netWorth.p50[i]),
                ldi_nw_p95: symlog(scenario.ldi.percentiles.netWorth.p95[i]),
                rot_nw_p5: symlog(scenario.rot.percentiles.netWorth.p5[i]),
                rot_nw_p50: symlog(scenario.rot.percentiles.netWorth.p50[i]),
                rot_nw_p95: symlog(scenario.rot.percentiles.netWorth.p95[i]),
                // Reference lines: total wealth and PV expenses (median only)
                ldi_tw_p50: symlog(scenario.ldi.percentiles.totalAssets.p50[i]),
                rot_tw_p50: symlog(scenario.rot.percentiles.totalAssets.p50[i]),
                ldi_pvexp_p50: symlog(scenario.ldi.percentiles.pvExpenses.p50[i]),
                rot_pvexp_p50: symlog(scenario.rot.percentiles.pvExpenses.p50[i]),
              }));

              // Panel 12 data: Annual Consumption (LDI vs RoT)
              const consumptionData = ages.map((age, i) => ({
                age,
                ldi_cons_p5: symlog(scenario.ldi.percentiles.consumption.p5[i]),
                ldi_cons_p50: symlog(scenario.ldi.percentiles.consumption.p50[i]),
                ldi_cons_p95: symlog(scenario.ldi.percentiles.consumption.p95[i]),
                rot_cons_p5: symlog(scenario.rot.percentiles.consumption.p5[i]),
                rot_cons_p50: symlog(scenario.rot.percentiles.consumption.p50[i]),
                rot_cons_p95: symlog(scenario.rot.percentiles.consumption.p95[i]),
              }));

              return (
              <div style={{
                opacity: simulationResultsStale ? 0.5 : 1,
                transition: 'opacity 0.3s ease',
                position: 'relative',
              }}>
                {/* Stale results warning banner */}
                {simulationResultsStale && (
                  <div style={{
                    background: '#fff3cd',
                    border: '1px solid #ffc107',
                    borderRadius: '8px',
                    padding: '12px 16px',
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}>
                    <span style={{ fontSize: '18px' }}>⚠️</span>
                    <div>
                      <div style={{ fontWeight: 'bold', color: '#856404', fontSize: '13px' }}>
                        Results are stale
                      </div>
                      <div style={{ color: '#856404', fontSize: '12px' }}>
                        Parameters have changed since the last simulation. Click "Run Simulation" to update.
                      </div>
                    </div>
                  </div>
                )}
                {/* 2-column grid layout matching PDF 4x2 structure */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>

                  {/* Panel 1: Cumulative Stock Market Returns (fan chart) */}
                  <ChartCard title="Panel 1: Cumulative Stock Market Returns">
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={marketData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={(v) => `${Math.exp(v).toFixed(0)}x`} domain={['auto', 'auto']} />
                        <Tooltip formatter={(v) => v !== undefined ? [`${Math.exp(v as number).toFixed(1)}x`, 'Cumulative'] : ['', '']} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#666" strokeDasharray="5 5" />
                        <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                        <Area type="monotone" dataKey="sr_p5" stackId="sr" fill="transparent" stroke="transparent" />
                        <Area type="monotone" dataKey="sr_band_5_25" stackId="sr" fill={COLOR_STOCKS} fillOpacity={0.15} stroke="transparent" />
                        <Area type="monotone" dataKey="sr_band_25_75" stackId="sr" fill={COLOR_STOCKS} fillOpacity={0.3} stroke="transparent" />
                        <Area type="monotone" dataKey="sr_band_75_95" stackId="sr" fill={COLOR_STOCKS} fillOpacity={0.15} stroke="transparent" />
                        <Line type="monotone" dataKey="sr_p50" stroke={COLOR_STOCKS} strokeWidth={2} dot={false} name="Median" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                      Log scale (1x = starting value). Retirement age marked.
                    </div>
                  </ChartCard>

                  {/* Panel 2: Interest Rate Paths (fan chart) */}
                  <ChartCard title="Panel 2: Interest Rate Paths">
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={marketData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={(v) => `${v.toFixed(1)}%`} domain={['auto', 'auto']} />
                        <Tooltip formatter={(v) => v !== undefined ? [`${(v as number).toFixed(2)}%`, 'Rate'] : ['', '']} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#666" strokeDasharray="5 5" />
                        <Area type="monotone" dataKey="rate_p5" stackId="rate" fill="transparent" stroke="transparent" />
                        <Area type="monotone" dataKey="rate_band_5_25" stackId="rate" fill={COLOR_RATES} fillOpacity={0.15} stroke="transparent" />
                        <Area type="monotone" dataKey="rate_band_25_75" stackId="rate" fill={COLOR_RATES} fillOpacity={0.3} stroke="transparent" />
                        <Area type="monotone" dataKey="rate_band_75_95" stackId="rate" fill={COLOR_RATES} fillOpacity={0.15} stroke="transparent" />
                        <Line type="monotone" dataKey="rate_p50" stroke={COLOR_RATES} strokeWidth={2} dot={false} name="Median" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '4px' }}>
                      Percentile bands (5th-95th). Retirement age marked.
                    </div>
                  </ChartCard>

                  {/* Panel 3: Financial Wealth - LDI vs RoT (overlaid fan chart) */}
                  <ChartCard title="Panel 3: Financial Wealth (LDI vs RoT)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={wealthAllocationData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={symlogTickFormatter} domain={['auto', 'auto']} />
                        <Tooltip formatter={symlogTooltipFormatter} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#666" strokeDasharray="5 5" />
                        <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                        {/* LDI lines */}
                        <Line type="monotone" dataKey="ldi_fw_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_fw_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_fw_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT lines */}
                        <Line type="monotone" dataKey="rot_fw_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_fw_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_fw_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      <span style={{ color: COLOR_LDI }}>LDI (blue)</span> vs <span style={{ color: COLOR_ROT }}>RoT (gold)</span>. Shows 5th, 50th, 95th percentiles.
                    </div>
                  </ChartCard>

                  {/* Panel 4: Default Timing (histogram) */}
                  <ChartCard title="Panel 4: Default Timing">
                    <div style={{ padding: '8px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '12px', textAlign: 'center' }}>
                        <div style={{ background: '#e8f5e9', padding: '12px', borderRadius: '6px' }}>
                          <div style={{ fontSize: '11px', color: '#2e7d32' }}>LDI Default Rate</div>
                          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1b5e20' }}>{(scenario.ldi.defaultRate * 100).toFixed(1)}%</div>
                        </div>
                        <div style={{ background: scenario.rot.defaultRate > 0.1 ? '#ffebee' : '#e8f5e9', padding: '12px', borderRadius: '6px' }}>
                          <div style={{ fontSize: '11px', color: scenario.rot.defaultRate > 0.1 ? '#c62828' : '#2e7d32' }}>RoT Default Rate</div>
                          <div style={{ fontSize: '24px', fontWeight: 'bold', color: scenario.rot.defaultRate > 0.1 ? '#b71c1c' : '#1b5e20' }}>{(scenario.rot.defaultRate * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                      {(ldiDefaultAges.length > 0 || rotDefaultAges.length > 0) ? (
                        <ResponsiveContainer width="100%" height={180}>
                          <BarChart data={bins}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="age" fontSize={10} />
                            <YAxis fontSize={11} />
                            <Tooltip />
                            <Bar dataKey="LDI" fill={COLOR_LDI} name="LDI" />
                            <Bar dataKey="RoT" fill={COLOR_ROT} name="RoT" />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>No defaults in either strategy</div>
                      )}
                    </div>
                  </ChartCard>

                  {/* Panel 5: Stock Allocation - LDI vs RoT (overlaid fan chart) */}
                  <ChartCard title="Panel 5: Stock Allocation (LDI vs RoT)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={wealthAllocationData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} domain={[0, 100]} tickFormatter={formatPercent} />
                        <Tooltip formatter={percentTooltipFormatter} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#666" strokeDasharray="5 5" />
                        {/* LDI lines */}
                        <Line type="monotone" dataKey="ldi_stock_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_stock_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_stock_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT lines */}
                        <Line type="monotone" dataKey="rot_stock_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_stock_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_stock_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      LDI dynamic allocation vs RoT static. Y-axis: 0-100%.
                    </div>
                  </ChartCard>

                  {/* Panel 6: Bond Allocation - LDI vs RoT (overlaid fan chart) */}
                  <ChartCard title="Panel 6: Bond Allocation (LDI vs RoT)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={wealthAllocationData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} domain={[0, 100]} tickFormatter={formatPercent} />
                        <Tooltip formatter={percentTooltipFormatter} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#666" strokeDasharray="5 5" />
                        {/* LDI lines */}
                        <Line type="monotone" dataKey="ldi_bond_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_bond_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_bond_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT lines */}
                        <Line type="monotone" dataKey="rot_bond_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_bond_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_bond_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      Duration-matched bonds for LDI. Y-axis: 0-100%.
                    </div>
                  </ChartCard>

                  {/* Panel 7: Terminal Wealth Distribution (histogram) */}
                  <ChartCard title="Panel 7: Terminal Wealth Distribution">
                    <div style={{ padding: '8px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '12px', textAlign: 'center' }}>
                        <div>
                          <div style={{ fontSize: '11px', color: COLOR_LDI }}>LDI Median</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLOR_LDI }}>${Math.round(scenario.ldi.medianFinalWealth)}k</div>
                        </div>
                        <div>
                          <div style={{ fontSize: '11px', color: COLOR_ROT }}>RoT Median</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLOR_ROT }}>${Math.round(scenario.rot.medianFinalWealth)}k</div>
                        </div>
                      </div>
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={wealthHistData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="bin" fontSize={10} />
                          <YAxis fontSize={11} />
                          <Tooltip />
                          <Bar dataKey="LDI" fill={COLOR_LDI} name="LDI" />
                          <Bar dataKey="RoT" fill={COLOR_ROT} name="RoT" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </ChartCard>

                  {/* Panel 8: PV Consumption Distribution (histogram) */}
                  <ChartCard title="Panel 8: PV Consumption Distribution">
                    <div style={{ padding: '8px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '12px', textAlign: 'center' }}>
                        <div>
                          <div style={{ fontSize: '11px', color: COLOR_LDI }}>LDI Median PV</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLOR_LDI }}>${Math.round(scenario.ldi.medianPvConsumption)}k</div>
                        </div>
                        <div>
                          <div style={{ fontSize: '11px', color: COLOR_ROT }}>RoT Median PV</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLOR_ROT }}>${Math.round(scenario.rot.medianPvConsumption)}k</div>
                        </div>
                      </div>
                      <ResponsiveContainer width="100%" height={180}>
                        <BarChart data={pvHistData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="bin" fontSize={10} />
                          <YAxis fontSize={11} />
                          <Tooltip />
                          <Bar dataKey="LDI" fill={COLOR_LDI} name="LDI" />
                          <Bar dataKey="RoT" fill={COLOR_ROT} name="RoT" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </ChartCard>

                  {/* Panel 9: Net Fixed Income PV */}
                  <ChartCard title="Panel 9: Net Fixed Income PV ($k)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={netFiPvData}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={(v) => `$${Math.round(v)}k`} />
                        <Tooltip formatter={(v, name) => [`$${Math.round(v as number)}k`, name]} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#999" strokeDasharray="3 3" />
                        <ReferenceLine y={0} stroke="#000" strokeWidth={1.5} opacity={0.7} />
                        {/* LDI lines */}
                        <Line type="monotone" dataKey="ldi_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT lines */}
                        <Line type="monotone" dataKey="rot_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      Net FI PV = Bond Holdings + HC Bond - Expense Bond. Zero line = perfectly hedged.
                    </div>
                  </ChartCard>

                  {/* Panel 10: DV01 (Interest Rate Sensitivity) */}
                  <ChartCard title="Panel 10: Interest Rate Sensitivity (DV01)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={dv01Data}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={(v) => `$${Math.round(v)}`} />
                        <Tooltip formatter={(v, name) => [`$${Math.round(v as number)}`, name]} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#999" strokeDasharray="3 3" />
                        <ReferenceLine y={0} stroke="#000" strokeWidth={1.5} opacity={0.7} />
                        {/* LDI lines */}
                        <Line type="monotone" dataKey="ldi_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT lines */}
                        <Line type="monotone" dataKey="rot_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      DV01 = Dollar value change per 1pp rate move. Zero = duration matched.
                    </div>
                  </ChartCard>

                  {/* Panel 11: Net Wealth (HC + FW - PV Expenses) */}
                  <ChartCard title="Panel 11: Net Wealth (HC + FW − Expenses)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={netWealthData}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={symlogTickFormatter} />
                        <Tooltip formatter={symlogTooltipFormatter} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#999" strokeDasharray="3 3" />
                        <ReferenceLine y={0} stroke="#000" strokeWidth={1.5} opacity={0.7} />
                        {/* LDI Net Worth lines */}
                        <Line type="monotone" dataKey="ldi_nw_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_nw_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_nw_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT Net Worth lines */}
                        <Line type="monotone" dataKey="rot_nw_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_nw_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_nw_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      Net Wealth = HC + FW − PV(Expenses). Zero means exactly funded.
                    </div>
                  </ChartCard>

                  {/* Panel 12: Annual Consumption (LDI vs RoT) */}
                  <ChartCard title="Panel 12: Annual Consumption (LDI vs RoT)">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={consumptionData}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="age" fontSize={11} />
                        <YAxis fontSize={11} tickFormatter={symlogTickFormatter} />
                        <Tooltip formatter={symlogTooltipFormatter} />
                        <ReferenceLine x={scenarioRetirementAge} stroke="#999" strokeDasharray="3 3" />
                        <ReferenceLine y={0} stroke="#000" strokeWidth={1.5} opacity={0.7} />
                        {/* LDI lines */}
                        <Line type="monotone" dataKey="ldi_cons_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
                        <Line type="monotone" dataKey="ldi_cons_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
                        <Line type="monotone" dataKey="ldi_cons_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
                        {/* RoT lines */}
                        <Line type="monotone" dataKey="rot_cons_p5" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 5th" />
                        <Line type="monotone" dataKey="rot_cons_p50" stroke={COLOR_ROT} strokeWidth={2} dot={false} name="RoT Median" />
                        <Line type="monotone" dataKey="rot_cons_p95" stroke={COLOR_ROT} strokeWidth={1} strokeDasharray="2 2" dot={false} name="RoT 95th" />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', textAlign: 'center', marginTop: '4px' }}>
                      Annual total consumption (subsistence + variable). LDI adapts; RoT uses fixed 4% rule.
                    </div>
                  </ChartCard>

                </div>

                {/* Key Takeaways */}
                <div style={{
                  background: '#2c3e50',
                  color: '#fff',
                  borderRadius: '8px',
                  padding: '20px',
                  marginTop: '24px',
                }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '12px' }}>
                    Key Takeaways: {scenarioType === 'baseline' ? 'Baseline Scenario' : scenarioType === 'sequenceRisk' ? 'Sequence Risk' : 'Rate Shock'}
                  </div>
                  {scenarioType === 'baseline' && (
                    <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                      <li><strong>LDI eliminates default risk</strong> - adaptive consumption always meets subsistence</li>
                      <li><strong>RoT can default under bad sequences</strong> - fixed 4% withdrawal ignores wealth changes</li>
                      <li><strong>Similar PV consumption</strong> - LDI doesnt sacrifice much consumption for safety</li>
                      <li><strong>Human capital matters</strong> - LDI accounts for bond-like future earnings in allocation</li>
                    </ul>
                  )}
                  {scenarioType === 'sequenceRisk' && (
                    <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                      <li><strong>4% Rule ignores market conditions</strong> - withdrawing fixed amounts from a falling portfolio accelerates depletion</li>
                      <li><strong>Sequence matters</strong> - even with the same average returns, bad early years can be catastrophic</li>
                      <li><strong>Adaptive consumption protects subsistence</strong> - by reducing variable spending when wealth drops, the floor is always met</li>
                      <li><strong>Trade-off:</strong> Adaptive may mean lower consumption in good times, but eliminates default risk</li>
                    </ul>
                  )}
                  {scenarioType === 'rateShock' && (
                    <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                      <li><strong>Rate drops increase liability PV</strong> - future expenses cost more in present value terms</li>
                      <li><strong>Duration matching hedges this risk</strong> - long bonds appreciate when rates fall, offsetting the liability increase</li>
                      <li><strong>Cash/short bonds leave you exposed</strong> - they dont appreciate enough to cover the increased cost of future spending</li>
                      <li><strong>The portfolio already accounts for human capital</strong> - young workers have bond-like future earnings that offset some rate risk</li>
                    </ul>
                  )}
                </div>
              </div>
              );
            })()}
          </>
        )}
      </div>
    </div>
  );
}
