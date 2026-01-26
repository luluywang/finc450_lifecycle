// Lifecycle Path Visualizer - Claude Artifact
// Interactive visualization for lifecycle investment strategy
// Copy this entire file into a Claude artifact (React type)

import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine
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
  muExcess: 0.04,         // Equity risk premium
  bondSharpe: 0.037,      // Bond Sharpe ratio
  sigmaS: 0.18,           // Stock return volatility
  rho: 0.0,               // Correlation between rate and stock shocks
  bondDuration: 20.0,     // Duration for HC decomposition
};

/**
 * Compute bond excess return from Sharpe ratio.
 * mu_bond = bond_sharpe * bond_duration * sigma_r
 */
function computeMuBondFromEcon(econ: EconomicParams): number {
  return econ.bondSharpe * econ.bondDuration * econ.sigmaR;
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
  allowLeverage: boolean;       // Allow shorting and leverage in portfolio

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
  allowLeverage: false,         // No leverage allowed

  // Economic parameters
  riskFreeRate: 0.02,           // 2% real risk-free rate
  equityPremium: 0.04,          // 4% equity premium

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
  bondSharpe: 0.037,      // Same bond Sharpe
  sigmaS: 0.25,           // Higher stock volatility (25% vs 18%)
  rho: -0.2,              // Negative stock-rate correlation (flight to quality)
  bondDuration: 20.0,     // Same duration
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
  allowLeverage: false,

  // Economic parameters
  riskFreeRate: 0.02,
  equityPremium: 0.04,

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
  allowLeverage: true,          // Allow leverage for aggressive hedging
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
  totalWealth: number;          // HC + FW (Python: total_wealth)
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
}

// Helper function to compute muBond from bondSharpe
function computeMuBond(params: Params): number {
  return params.bondSharpe * params.bondDuration * params.sigmaR;
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
  totalWealth: number[];
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
  // Market conditions tracking
  cumulativeStockReturn: number[];  // Cumulative stock return (1 = starting value)
  interestRate: number[];           // Interest rate path per year
}

interface MonteCarloResult {
  ages: number[];
  runs: LifecycleResult[];
  // Percentiles for key variables
  consumption_p05: number[];
  consumption_p25: number[];
  consumption_p50: number[];
  consumption_p75: number[];
  consumption_p95: number[];
  financialWealth_p05: number[];
  financialWealth_p25: number[];
  financialWealth_p50: number[];
  financialWealth_p75: number[];
  financialWealth_p95: number[];
  totalWealth_p05: number[];
  totalWealth_p25: number[];
  totalWealth_p50: number[];
  totalWealth_p75: number[];
  totalWealth_p95: number[];
  // Net Worth = HC + FW - pvExpenses (total wealth including liabilities)
  netWorth_p05: number[];
  netWorth_p25: number[];
  netWorth_p50: number[];
  netWorth_p75: number[];
  netWorth_p95: number[];
  // Market conditions percentiles
  stockReturn_p05: number[];
  stockReturn_p25: number[];
  stockReturn_p50: number[];
  stockReturn_p75: number[];
  stockReturn_p95: number[];
  interestRate_p05: number[];
  interestRate_p25: number[];
  interestRate_p50: number[];
  interestRate_p75: number[];
  interestRate_p95: number[];
}

type PageType = 'base' | 'monteCarlo' | 'scenarios';

type ConsumptionRule = 'adaptive' | 'fourPercent';

interface ScenarioParams {
  consumptionRule: ConsumptionRule;
  rateShockAge: number;      // Age when rate shock occurs
  rateShockMagnitude: number; // Change in rate (e.g., -0.02 for 2% drop)
  badReturnsEarly: boolean;   // Force bad returns in first 10 years of retirement
}

// type StrategyType = 'optimal' | 'ruleOfThumb'; // Kept for documentation

interface StrategyResult {
  ages: number[];
  financialWealth: number[];
  totalConsumption: number[];
  subsistenceConsumption: number[];
  variableConsumption: number[];
  stockWeight: number[];
  bondWeight: number[];
  cashWeight: number[];
  savings: number[];
  defaulted: boolean;
  defaultAge: number | null;
  terminalWealth: number;
  // For scenario visualization
  cumulativeStockReturn: number[];  // Cumulative stock return (1 = starting value)
  interestRate: number[];           // Interest rate path
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
  rateFloor: number = -0.02,
  rateCap: number = 0.15
): [number, number] {
  // Random walk update: r_t = r_{t-1} + sigma_r * shock
  const newLatentRate = latentRate + sigmaR * rateShock;
  // Observed rate = capped latent rate
  const observedRate = Math.max(rateFloor, Math.min(rateCap, newLatentRate));
  return [newLatentRate, observedRate];
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
  rBar: number | null = null
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
  return weightedSum / pv;
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
 * @param allowLeverage - If true, allow negative/leveraged weights
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
  allowLeverage: boolean = false
): [number, number, number] {
  // Edge case: no financial wealth, return baseline targets
  if (fw <= 1e-6) {
    return [targetStock, targetBond, targetCash];
  }

  // Compute raw weights
  let wStock = targetFinStock / fw;
  let wBond = targetFinBond / fw;
  let wCash = targetFinCash / fw;

  // If leverage is allowed, return raw (unconstrained) weights
  if (allowLeverage) {
    return [wStock, wBond, wCash];
  }

  // Apply no-short constraint
  let equity = Math.max(0, wStock);
  let fixedIncome = Math.max(0, wBond + wCash);

  const totalAgg = equity + fixedIncome;
  if (totalAgg > 0) {
    equity /= totalAgg;
    fixedIncome /= totalAgg;
  } else {
    // Edge case: all target financial holdings are non-positive
    // Fall back to baseline target allocations
    equity = targetStock;
    fixedIncome = targetBond + targetCash;
  }

  // Split fixed income between bonds and cash
  let wB: number;
  let wC: number;

  if (wBond > 0 && wCash > 0) {
    const fiTotal = wBond + wCash;
    wB = fixedIncome * (wBond / fiTotal);
    wC = fixedIncome * (wCash / fiTotal);
  } else if (wBond > 0) {
    wB = fixedIncome;
    wC = 0;
  } else if (wCash > 0) {
    wB = 0;
    wC = fixedIncome;
  } else {
    // Both wBond and wCash are non-positive
    const targetFI = targetBond + targetCash;
    if (targetFI > 0) {
      wB = fixedIncome * (targetBond / targetFI);
      wC = fixedIncome * (targetCash / targetFI);
    } else {
      // Edge case: target FI is also zero/negative, split equally
      wB = fixedIncome / 2;
      wC = fixedIncome / 2;
    }
  }

  return [equity, wB, wC];
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

  // Compute target allocations from MV optimization
  const muBond = computeMuBondFromEcon(econParams);
  const [targetStock, targetBond, targetCash] = computeFullMertonAllocationConstrained(
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
      const stockReturn = ratePaths[sim][t] + econParams.muExcess +
        econParams.sigmaS * stockShocks[sim][t];
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
  // Create a Params object from LifecycleParams for the existing functions
  const legacyParams: Params = {
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
    stockBetaHC: params.stockBetaHumanCapital,
    gamma: params.gamma,
    initialWealth: params.initialWealth,
    rBar: econParams.rBar,
    muStock: econParams.muExcess,
    bondSharpe: econParams.bondSharpe,
    sigmaS: econParams.sigmaS,
    sigmaR: econParams.sigmaR,
    rho: econParams.rho,
    bondDuration: econParams.bondDuration,
    phi: econParams.phi,
  };

  const earningsProfile = computeEarningsProfile(legacyParams);
  const expenseProfile = computeExpenseProfile(legacyParams);

  const baseEarnings: number[] = Array(totalYears).fill(0);
  const expenses: number[] = Array(totalYears).fill(0);
  for (let t = 0; t < workingYears; t++) {
    baseEarnings[t] = earningsProfile[t];
    expenses[t] = expenseProfile.working[t];
  }
  for (let t = workingYears; t < totalYears; t++) {
    expenses[t] = expenseProfile.retirement[t - workingYears];
  }

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
      const durationExp = computeDuration(remainingExpenses, currentRate, phi, rBar);

      let hc = 0.0;
      let durationHc = 0.0;
      if (isWorking) {
        const remainingEarnings = baseEarnings.slice(t, workingYears);
        hc = computePresentValue(remainingEarnings, currentRate, phi, rBar);
        durationHc = computeDuration(remainingEarnings, currentRate, phi, rBar);
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
      const totalWealth = fw + hc;
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
        totalWealth: totalWealth,
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
        const stockRet = stockReturnPaths[sim][t];
        const bondRet = bondReturnPaths[sim][t];
        const cashRet = ratePaths[sim][t];

        const portfolioReturn =
          actions.stockWeight * stockRet +
          actions.bondWeight * bondRet +
          actions.cashWeight * cashRet;

        const savings = currentEarnings - actions.consumption;
        financialWealth[t + 1] = fw * (1 + portfolioReturn) + savings;
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
  /** Allow shorting and leverage in portfolio */
  allowLeverage?: boolean;
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
 * const ldi = createLDIStrategy({ allowLeverage: false });
 * const result = simulateWithStrategy(ldi, params, econ, rateShocks, stockShocks);
 */
function createLDIStrategy(options: LDIStrategyOptions = {}): Strategy {
  const { consumptionRate = null, allowLeverage = false } = options;

  const strategy = function ldiStrategy(state: SimulationState): StrategyActions {
    // Compute consumption rate if not specified
    let effectiveConsumptionRate: number;
    if (consumptionRate === null) {
      const r = state.econParams.rBar;
      const expectedStockReturn = r + state.econParams.muExcess;
      const avgReturn =
        state.targetStock * expectedStockReturn +
        state.targetBond * r +
        state.targetCash * r;
      effectiveConsumptionRate = avgReturn + state.params.consumptionBoost;
    } else {
      effectiveConsumptionRate = consumptionRate;
    }

    // Consumption: subsistence + rate * max(0, net_worth)
    let subsistence = state.expenses;
    let variable = Math.max(0, effectiveConsumptionRate * state.netWorth);
    let totalCons = subsistence + variable;

    // Apply constraints based on lifecycle stage
    if (state.isWorking) {
      // During working years: can't consume more than earnings
      if (totalCons > state.earnings) {
        totalCons = state.earnings;
        variable = Math.max(0, state.earnings - subsistence);
      }
    } else {
      // Retirement: can't consume more than financial wealth
      if (state.financialWealth <= 0) {
        return {
          consumption: 0.0,
          stockWeight: state.targetStock,
          bondWeight: state.targetBond,
          cashWeight: state.targetCash,
        };
      }
      if (totalCons > state.financialWealth) {
        totalCons = state.financialWealth;
        variable = Math.max(0, state.financialWealth - subsistence);
        if (variable < 0) {
          subsistence = state.financialWealth;
          variable = 0.0;
        }
      }
    }

    // LDI allocation: target financial holdings
    // Target = target_pct * total_wealth - HC_component + expense_component
    const targetFinStock = state.targetStock * state.totalWealth - state.hcStockComponent;
    const targetFinBond = state.targetBond * state.totalWealth - state.hcBondComponent + state.expBondComponent;
    const targetFinCash = state.targetCash * state.totalWealth - state.hcCashComponent + state.expCashComponent;

    // Normalize to weights
    const [wS, wB, wC] = normalizePortfolioWeights(
      targetFinStock,
      targetFinBond,
      targetFinCash,
      state.financialWealth,
      state.targetStock,
      state.targetBond,
      state.targetCash,
      allowLeverage
    );

    return {
      consumption: totalCons,
      stockWeight: wS,
      bondWeight: wB,
      cashWeight: wC,
    };
  } as Strategy;

  strategy.name = 'LDI';

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

  strategy.name = 'RuleOfThumb';

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
  const ldiStrategy = createLDIStrategy({ allowLeverage: false });
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
    strategyAParams: { allowLeverage: false },
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
  humanCapital: FieldPercentiles;
  interestRates: FieldPercentiles;
  stockReturns: FieldPercentiles;
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
 * const ldi = createLDIStrategy({ allowLeverage: false });
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
  const humanCapitalPaths = result.humanCapital as number[][];
  const interestRatePaths = result.interestRates as number[][];
  const stockReturnPaths = result.stockReturns as number[][];
  const defaultedFlags = result.defaulted as boolean[];
  const finalWealthValues = result.finalWealth as number[];

  // Compute percentile statistics
  const percentiles: PercentileStats = {
    financialWealth: computeFieldPercentiles(financialWealthPaths),
    consumption: computeFieldPercentiles(consumptionPaths),
    stockWeight: computeFieldPercentiles(stockWeightPaths),
    humanCapital: computeFieldPercentiles(humanCapitalPaths),
    interestRates: computeFieldPercentiles(interestRatePaths),
    stockReturns: computeFieldPercentiles(stockReturnPaths),
  };

  // Compute summary statistics
  const defaultCount = defaultedFlags.filter(d => d).length;
  const defaultRate = defaultCount / numSims;
  const medianFinalWealth = computePercentile(finalWealthValues, 50);

  // Compute PV consumption for each simulation using risk-free rate
  const pvConsumptionValues: number[] = [];
  for (let sim = 0; sim < numSims; sim++) {
    const pv = computePvConsumption(consumptionPaths[sim], econParams.rBar);
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
  const ldiStrategy = createLDIStrategy({ allowLeverage: false });
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

  // Compute percentile statistics for both
  const ldiPercentiles: PercentileStats = {
    financialWealth: computeFieldPercentiles(ldiResult.financialWealth as number[][]),
    consumption: computeFieldPercentiles(ldiResult.consumption as number[][]),
    stockWeight: computeFieldPercentiles(ldiResult.stockWeight as number[][]),
    humanCapital: computeFieldPercentiles(ldiResult.humanCapital as number[][]),
    interestRates: computeFieldPercentiles(ldiResult.interestRates as number[][]),
    stockReturns: computeFieldPercentiles(ldiResult.stockReturns as number[][]),
  };

  const rotPercentiles: PercentileStats = {
    financialWealth: computeFieldPercentiles(rotResult.financialWealth as number[][]),
    consumption: computeFieldPercentiles(rotResult.consumption as number[][]),
    stockWeight: computeFieldPercentiles(rotResult.stockWeight as number[][]),
    humanCapital: computeFieldPercentiles(rotResult.humanCapital as number[][]),
    interestRates: computeFieldPercentiles(rotResult.interestRates as number[][]),
    stockReturns: computeFieldPercentiles(rotResult.stockReturns as number[][]),
  };

  // Compute summary statistics
  const ldiDefaulted = ldiResult.defaulted as boolean[];
  const rotDefaulted = rotResult.defaulted as boolean[];
  const ldiFinalWealth = ldiResult.finalWealth as number[];
  const rotFinalWealth = rotResult.finalWealth as number[];
  const ldiConsumption = ldiResult.consumption as number[][];
  const rotConsumption = rotResult.consumption as number[][];

  // Compute PV consumption for each simulation
  const ldiPvConsumptionValues: number[] = [];
  const rotPvConsumptionValues: number[] = [];
  for (let sim = 0; sim < numSims; sim++) {
    ldiPvConsumptionValues.push(computePvConsumption(ldiConsumption[sim], econParams.rBar));
    rotPvConsumptionValues.push(computePvConsumption(rotConsumption[sim], econParams.rBar));
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
    },
    resultB: {
      result: rotResult,
      percentiles: rotPercentiles,
      numSims,
      seed,
      defaultRate: rotDefaulted.filter(d => d).length / numSims,
      medianFinalWealth: computePercentile(rotFinalWealth, 50),
      medianPvConsumption: computePercentile(rotPvConsumptionValues, 50),
    },
    strategyAParams: { allowLeverage: false },
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
  /** Stock shock magnitude during sequence risk (default: -1.5 = 1.5 std devs below mean) */
  sequenceRiskStockShock?: number;
  /** Rate shock magnitude at retirement for Rate Shock scenario (default: -0.02 = -2%) */
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
    sequenceRiskStockShock = -1.5,
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
        // Rate Shock: Apply one-time rate shock at retirement
        // Rate drops (negative shock = falling rates = higher bond prices but higher PV liabilities)
        simStockShocks.push(stockShock);
        if (t === workingYears) {
          // Convert rate change to shock units: shock = deltaR / sigmaR
          // Using large shock to represent the rate drop
          // Note: sigmaR is typically small (0.003), so to get -2% rate change
          // we need a very large shock. Instead, we pass the rate change directly
          // and handle it in the simulation. For now, use a large negative shock.
          simRateShocks.push(rateShockMagnitude / 0.003); // ~-6.67 std devs for -2% rate drop
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
  const ldiStrategy = createLDIStrategy({ allowLeverage: false });
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

    // Compute percentiles and summary stats for LDI
    const ldiPercentiles: PercentileStats = {
      financialWealth: computeFieldPercentiles(ldiResult.financialWealth as number[][]),
      consumption: computeFieldPercentiles(ldiResult.consumption as number[][]),
      stockWeight: computeFieldPercentiles(ldiResult.stockWeight as number[][]),
      humanCapital: computeFieldPercentiles(ldiResult.humanCapital as number[][]),
      interestRates: computeFieldPercentiles(ldiResult.interestRates as number[][]),
      stockReturns: computeFieldPercentiles(ldiResult.stockReturns as number[][]),
    };
    const ldiDefaulted = ldiResult.defaulted as boolean[];
    const ldiFinalWealth = ldiResult.finalWealth as number[];

    // Compute percentiles and summary stats for RoT
    const rotPercentiles: PercentileStats = {
      financialWealth: computeFieldPercentiles(rotResult.financialWealth as number[][]),
      consumption: computeFieldPercentiles(rotResult.consumption as number[][]),
      stockWeight: computeFieldPercentiles(rotResult.stockWeight as number[][]),
      humanCapital: computeFieldPercentiles(rotResult.humanCapital as number[][]),
      interestRates: computeFieldPercentiles(rotResult.interestRates as number[][]),
      stockReturns: computeFieldPercentiles(rotResult.stockReturns as number[][]),
    };
    const rotDefaulted = rotResult.defaulted as boolean[];
    const rotFinalWealth = rotResult.finalWealth as number[];
    const ldiConsumption = ldiResult.consumption as number[][];
    const rotConsumption = rotResult.consumption as number[][];

    // Compute PV consumption for each simulation
    const ldiPvConsumptionValues: number[] = [];
    const rotPvConsumptionValues: number[] = [];
    for (let sim = 0; sim < numSims; sim++) {
      ldiPvConsumptionValues.push(computePvConsumption(ldiConsumption[sim], econParams.rBar));
      rotPvConsumptionValues.push(computePvConsumption(rotConsumption[sim], econParams.rBar));
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
      },
      rot: {
        result: rotResult,
        percentiles: rotPercentiles,
        numSims,
        seed,
        defaultRate: rotDefaulted.filter(d => d).length / numSims,
        medianFinalWealth: computePercentile(rotFinalWealth, 50),
        medianPvConsumption: computePercentile(rotPvConsumptionValues, 50),
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

function computeLifecycleMedianPath(params: Params): LifecycleResult {
  const r = params.rBar;
  const phi = params.phi;
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  // Compute target allocations from MV optimization
  const muBond = computeMuBond(params);
  const [targetStock, targetBond, targetCash] = computeFullMertonAllocationConstrained(
    params.muStock, muBond, params.sigmaS, params.sigmaR,
    params.rho, params.bondDuration, params.gamma
  );

  // Initialize arrays
  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const earnings = Array(totalYears).fill(0);
  const expenses = Array(totalYears).fill(0);

  // Fill earnings and expenses
  const earningsProfile = computeEarningsProfile(params);
  const expenseProfile = computeExpenseProfile(params);

  for (let i = 0; i < workingYears; i++) {
    earnings[i] = earningsProfile[i];
    expenses[i] = expenseProfile.working[i];
  }
  for (let i = workingYears; i < totalYears; i++) {
    expenses[i] = expenseProfile.retirement[i - workingYears];
  }

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

    pvEarnings[i] = computePresentValue(remainingEarnings, r, phi, r);
    pvExpenses[i] = computePresentValue(remainingExpenses, r, phi, r);
    durationEarnings[i] = computeDuration(remainingEarnings, r, phi, r);
    durationExpenses[i] = computeDuration(remainingExpenses, r, phi, r);
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

  // Expected portfolio return
  const expectedStockReturn = r + params.muStock;
  const expectedBondReturn = r;
  const avgReturn = targetStock * expectedStockReturn +
                    targetBond * expectedBondReturn +
                    targetCash * r;

  // Consumption rate = median return + 1pp
  const consumptionRate = avgReturn + 0.01;

  // Simulate wealth accumulation
  for (let i = 0; i < totalYears; i++) {
    netWorth[i] = humanCapital[i] + financialWealth[i] - pvExpenses[i];
    variableConsumption[i] = Math.max(0, consumptionRate * netWorth[i]);
    totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

    if (earnings[i] > 0 && totalConsumption[i] > earnings[i]) {
      // Working years: cap consumption at earnings (can't borrow against HC)
      totalConsumption[i] = earnings[i];
      variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
    } else if (earnings[i] === 0) {
      // Retirement: cap consumption at financial wealth
      const fw = financialWealth[i];
      if (subsistenceConsumption[i] > fw) {
        // Bankruptcy: can't even meet subsistence, consume whatever remains
        totalConsumption[i] = fw;
        subsistenceConsumption[i] = fw;
        variableConsumption[i] = 0;
      } else if (totalConsumption[i] > fw) {
        // Can meet subsistence but not variable consumption
        totalConsumption[i] = fw;
        variableConsumption[i] = fw - subsistenceConsumption[i];
      }
    }

    const savings = earnings[i] - totalConsumption[i];

    if (i < totalYears - 1) {
      financialWealth[i + 1] = Math.max(0, financialWealth[i] * (1 + avgReturn) + savings);
    }
  }

  // Total wealth
  const totalWealth = financialWealth.map((fw, i) => fw + humanCapital[i]);

  // Target financial holdings
  const targetFinStocks = totalWealth.map((tw, i) => targetStock * tw - hcStock[i]);
  const targetFinBonds = totalWealth.map((tw, i) => targetBond * tw - hcBond[i]);
  const targetFinCash = totalWealth.map((tw, i) => targetCash * tw - hcCash[i]);

  // Apply no-short constraints to get portfolio weights using normalize helper
  const stockWeight = Array(totalYears).fill(0);
  const bondWeight = Array(totalYears).fill(0);
  const cashWeight = Array(totalYears).fill(0);

  for (let i = 0; i < totalYears; i++) {
    const [wS, wB, wC] = normalizePortfolioWeights(
      targetFinStocks[i], targetFinBonds[i], targetFinCash[i],
      financialWealth[i], targetStock, targetBond, targetCash,
      false  // no leverage for median path
    );
    stockWeight[i] = wS;
    bondWeight[i] = wB;
    cashWeight[i] = wC;
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
    totalWealth,
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
  };
}

// =============================================================================
// Monte Carlo Simulation
// =============================================================================

function computeStochasticPath(params: Params, rand: () => number): LifecycleResult {
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  // Initialize arrays
  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const earnings = Array(totalYears).fill(0);
  const expenses = Array(totalYears).fill(0);

  // Fill earnings and expenses (these don't change with market shocks)
  const earningsProfile = computeEarningsProfile(params);
  const expenseProfile = computeExpenseProfile(params);

  for (let i = 0; i < workingYears; i++) {
    earnings[i] = earningsProfile[i];
    expenses[i] = expenseProfile.working[i];
  }
  for (let i = workingYears; i < totalYears; i++) {
    expenses[i] = expenseProfile.retirement[i - workingYears];
  }

  // State variables that evolve with shocks
  let latentRate = params.rBar;  // Latent rate follows pure random walk
  let currentRate = params.rBar; // Observed rate = capped latent rate
  const phi = params.phi;

  // Track arrays for each year
  const pvEarnings = Array(totalYears).fill(0);
  const pvExpenses = Array(totalYears).fill(0);
  const durationEarnings = Array(totalYears).fill(0);
  const durationExpenses = Array(totalYears).fill(0);
  const humanCapital = Array(totalYears).fill(0);
  const hcStock = Array(totalYears).fill(0);
  const hcBond = Array(totalYears).fill(0);
  const hcCash = Array(totalYears).fill(0);
  const expBond = Array(totalYears).fill(0);
  const expCash = Array(totalYears).fill(0);
  const financialWealth = Array(totalYears).fill(0);
  const totalWealth = Array(totalYears).fill(0);
  const stockWeight = Array(totalYears).fill(0);
  const bondWeight = Array(totalYears).fill(0);
  const cashWeight = Array(totalYears).fill(0);
  const subsistenceConsumption = [...expenses];
  const variableConsumption = Array(totalYears).fill(0);
  const totalConsumption = Array(totalYears).fill(0);
  const netWorth = Array(totalYears).fill(0);
  // Market conditions tracking
  const interestRateArr = Array(totalYears).fill(params.rBar);
  const cumulativeStockReturnArr = Array(totalYears).fill(1);
  let cumStockReturn = 1;

  financialWealth[0] = params.initialWealth;

  // Simulate year by year
  for (let i = 0; i < totalYears; i++) {
    // Generate correlated shocks for this year
    const [stockShock, rateShock] = generateCorrelatedShocks(rand, params.rho);

    // Update interest rate using random walk (phi=1.0)
    [latentRate, currentRate] = updateInterestRate(latentRate, params.sigmaR, rateShock);

    // Track interest rate
    interestRateArr[i] = currentRate;

    // Compute PVs and durations with current rate
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }
    const remainingExpenses = expenses.slice(i);

    pvEarnings[i] = computePresentValue(remainingEarnings, currentRate, phi, params.rBar);
    pvExpenses[i] = computePresentValue(remainingExpenses, currentRate, phi, params.rBar);
    durationEarnings[i] = computeDuration(remainingEarnings, currentRate, phi, params.rBar);
    durationExpenses[i] = computeDuration(remainingExpenses, currentRate, phi, params.rBar);

    humanCapital[i] = pvEarnings[i];

    // Decompose human capital
    hcStock[i] = humanCapital[i] * params.stockBetaHC;
    const nonStockHC = humanCapital[i] * (1 - params.stockBetaHC);

    if (params.bondDuration > 0 && nonStockHC > 0) {
      const bondFraction = Math.min(1, durationEarnings[i] / params.bondDuration);
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

    // Compute target allocation based on current market conditions
    const [targetStock, targetBond, targetCash] = computeFullMertonAllocationConstrained(
      params.muStock, computeMuBond(params), params.sigmaS, params.sigmaR,
      params.rho, params.bondDuration, params.gamma
    );

    totalWealth[i] = financialWealth[i] + humanCapital[i];

    // Portfolio weights using normalize helper
    const targetFinStocks = targetStock * totalWealth[i] - hcStock[i];
    const targetFinBonds = targetBond * totalWealth[i] - hcBond[i];
    const targetFinCash = targetCash * totalWealth[i] - hcCash[i];

    const [wS, wB, wC] = normalizePortfolioWeights(
      targetFinStocks, targetFinBonds, targetFinCash,
      financialWealth[i], targetStock, targetBond, targetCash,
      false  // no leverage
    );
    stockWeight[i] = wS;
    bondWeight[i] = wB;
    cashWeight[i] = wC;

    // Consumption decision
    netWorth[i] = humanCapital[i] + financialWealth[i] - pvExpenses[i];

    // Consumption rate based on expected return + 1pp
    const expectedReturn = currentRate + stockWeight[i] * params.muStock;
    const consumptionRate = expectedReturn + 0.01;

    variableConsumption[i] = Math.max(0, consumptionRate * netWorth[i]);
    totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

    if (earnings[i] > 0 && totalConsumption[i] > earnings[i]) {
      // Working years: cap consumption at earnings (can't borrow against HC)
      totalConsumption[i] = earnings[i];
      variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
    } else if (earnings[i] === 0) {
      // Retirement: cap consumption at financial wealth
      const fw = financialWealth[i];
      if (subsistenceConsumption[i] > fw) {
        // Bankruptcy: can't even meet subsistence, consume whatever remains
        totalConsumption[i] = fw;
        subsistenceConsumption[i] = fw;
        variableConsumption[i] = 0;
      } else if (totalConsumption[i] > fw) {
        // Can meet subsistence but not variable consumption
        totalConsumption[i] = fw;
        variableConsumption[i] = fw - subsistenceConsumption[i];
      }
    }

    // Wealth accumulation with stochastic returns
    // Track cumulative stock return at start of this year
    cumulativeStockReturnArr[i] = cumStockReturn;

    if (i < totalYears - 1) {
      const savings = earnings[i] - totalConsumption[i];

      // Realized returns with shocks
      const stockReturn = currentRate + params.muStock + params.sigmaS * stockShock;
      const bondReturn = currentRate - params.bondDuration * params.sigmaR * rateShock;
      const cashReturn = currentRate;

      // Update cumulative stock return
      cumStockReturn *= (1 + stockReturn);

      const portfolioReturn = stockWeight[i] * stockReturn +
                             bondWeight[i] * bondReturn +
                             cashWeight[i] * cashReturn;

      financialWealth[i + 1] = Math.max(0, financialWealth[i] * (1 + portfolioReturn) + savings);
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
    totalWealth,
    stockWeight,
    bondWeight,
    cashWeight,
    subsistenceConsumption,
    variableConsumption,
    totalConsumption,
    netWorth,
    targetStock: 0,
    targetBond: 0,
    targetCash: 0,
    cumulativeStockReturn: cumulativeStockReturnArr,
    interestRate: interestRateArr,
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

function computeMonteCarloSimulation(params: Params, numRuns: number = 50, baseSeed: number = 42): MonteCarloResult {
  const runs: LifecycleResult[] = [];

  // Run simulations with different seeds
  for (let run = 0; run < numRuns; run++) {
    const rand = mulberry32(baseSeed + run * 1000);
    runs.push(computeStochasticPath(params, rand));
  }

  const totalYears = params.endAge - params.startAge;
  const ages = runs[0].ages;

  // Compute percentiles for each year
  const consumption_p05 = Array(totalYears).fill(0);
  const consumption_p25 = Array(totalYears).fill(0);
  const consumption_p50 = Array(totalYears).fill(0);
  const consumption_p75 = Array(totalYears).fill(0);
  const consumption_p95 = Array(totalYears).fill(0);

  const financialWealth_p05 = Array(totalYears).fill(0);
  const financialWealth_p25 = Array(totalYears).fill(0);
  const financialWealth_p50 = Array(totalYears).fill(0);
  const financialWealth_p75 = Array(totalYears).fill(0);
  const financialWealth_p95 = Array(totalYears).fill(0);

  const totalWealth_p05 = Array(totalYears).fill(0);
  const totalWealth_p25 = Array(totalYears).fill(0);
  const totalWealth_p50 = Array(totalYears).fill(0);
  const totalWealth_p75 = Array(totalYears).fill(0);
  const totalWealth_p95 = Array(totalYears).fill(0);

  const netWorth_p05 = Array(totalYears).fill(0);
  const netWorth_p25 = Array(totalYears).fill(0);
  const netWorth_p50 = Array(totalYears).fill(0);
  const netWorth_p75 = Array(totalYears).fill(0);
  const netWorth_p95 = Array(totalYears).fill(0);

  // Market conditions percentiles
  const stockReturn_p05 = Array(totalYears).fill(0);
  const stockReturn_p25 = Array(totalYears).fill(0);
  const stockReturn_p50 = Array(totalYears).fill(0);
  const stockReturn_p75 = Array(totalYears).fill(0);
  const stockReturn_p95 = Array(totalYears).fill(0);

  const interestRate_p05 = Array(totalYears).fill(0);
  const interestRate_p25 = Array(totalYears).fill(0);
  const interestRate_p50 = Array(totalYears).fill(0);
  const interestRate_p75 = Array(totalYears).fill(0);
  const interestRate_p95 = Array(totalYears).fill(0);

  for (let i = 0; i < totalYears; i++) {
    const consumptionValues = runs.map(r => r.totalConsumption[i]);
    consumption_p05[i] = computePercentile(consumptionValues, 5);
    consumption_p25[i] = computePercentile(consumptionValues, 25);
    consumption_p50[i] = computePercentile(consumptionValues, 50);
    consumption_p75[i] = computePercentile(consumptionValues, 75);
    consumption_p95[i] = computePercentile(consumptionValues, 95);

    const fwValues = runs.map(r => r.financialWealth[i]);
    financialWealth_p05[i] = computePercentile(fwValues, 5);
    financialWealth_p25[i] = computePercentile(fwValues, 25);
    financialWealth_p50[i] = computePercentile(fwValues, 50);
    financialWealth_p75[i] = computePercentile(fwValues, 75);
    financialWealth_p95[i] = computePercentile(fwValues, 95);

    const twValues = runs.map(r => r.totalWealth[i]);
    totalWealth_p05[i] = computePercentile(twValues, 5);
    totalWealth_p25[i] = computePercentile(twValues, 25);
    totalWealth_p50[i] = computePercentile(twValues, 50);
    totalWealth_p75[i] = computePercentile(twValues, 75);
    totalWealth_p95[i] = computePercentile(twValues, 95);

    const nwValues = runs.map(r => r.netWorth[i]);
    netWorth_p05[i] = computePercentile(nwValues, 5);
    netWorth_p25[i] = computePercentile(nwValues, 25);
    netWorth_p50[i] = computePercentile(nwValues, 50);
    netWorth_p75[i] = computePercentile(nwValues, 75);
    netWorth_p95[i] = computePercentile(nwValues, 95);

    // Stock return percentiles (cumulative)
    const srValues = runs.map(r => r.cumulativeStockReturn[i]);
    stockReturn_p05[i] = computePercentile(srValues, 5);
    stockReturn_p25[i] = computePercentile(srValues, 25);
    stockReturn_p50[i] = computePercentile(srValues, 50);
    stockReturn_p75[i] = computePercentile(srValues, 75);
    stockReturn_p95[i] = computePercentile(srValues, 95);

    // Interest rate percentiles
    const irValues = runs.map(r => r.interestRate[i]);
    interestRate_p05[i] = computePercentile(irValues, 5);
    interestRate_p25[i] = computePercentile(irValues, 25);
    interestRate_p50[i] = computePercentile(irValues, 50);
    interestRate_p75[i] = computePercentile(irValues, 75);
    interestRate_p95[i] = computePercentile(irValues, 95);
  }

  return {
    ages,
    runs,
    consumption_p05,
    consumption_p25,
    consumption_p50,
    consumption_p75,
    consumption_p95,
    financialWealth_p05,
    financialWealth_p25,
    financialWealth_p50,
    financialWealth_p75,
    financialWealth_p95,
    totalWealth_p05,
    totalWealth_p25,
    totalWealth_p50,
    totalWealth_p75,
    totalWealth_p95,
    netWorth_p05,
    netWorth_p25,
    netWorth_p50,
    netWorth_p75,
    netWorth_p95,
    stockReturn_p05,
    stockReturn_p25,
    stockReturn_p50,
    stockReturn_p75,
    stockReturn_p95,
    interestRate_p05,
    interestRate_p25,
    interestRate_p50,
    interestRate_p75,
    interestRate_p95,
  };
}

// =============================================================================
// Scenario Simulation (for teaching concepts)
// =============================================================================

interface ScenarioResult {
  ages: number[];
  financialWealth: number[];
  totalWealth: number[];      // financialWealth + humanCapital
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
  const earnings = Array(totalYears).fill(0);
  const expenses = Array(totalYears).fill(0);

  // Fill earnings and expenses
  const earningsProfile = computeEarningsProfile(params);
  const expenseProfile = computeExpenseProfile(params);

  for (let i = 0; i < workingYears; i++) {
    earnings[i] = earningsProfile[i];
    expenses[i] = expenseProfile.working[i];
  }
  for (let i = workingYears; i < totalYears; i++) {
    expenses[i] = expenseProfile.retirement[i - workingYears];
  }

  // State tracking
  let latentRate = params.rBar;  // Latent rate follows pure random walk
  let currentRate = params.rBar; // Observed rate = capped latent rate
  const phi = params.phi;

  const financialWealth = Array(totalYears).fill(0);
  const totalWealthArr = Array(totalYears).fill(0);
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

    // Update interest rate using random walk (phi=1.0)
    [latentRate, currentRate] = updateInterestRate(latentRate, params.sigmaR, rateShock);
    rateHistory[i] = currentRate;

    // Compute PVs with current rate
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }
    const remainingExpenses = expenses.slice(i);

    const pvEarnings = computePresentValue(remainingEarnings, currentRate, phi, params.rBar);
    const pvExpenses = computePresentValue(remainingExpenses, currentRate, phi, params.rBar);
    const durationEarnings = computeDuration(remainingEarnings, currentRate, phi, params.rBar);

    const humanCapital = pvEarnings;

    // HC decomposition for portfolio
    const hcStock = humanCapital * params.stockBetaHC;
    const nonStockHC = humanCapital * (1 - params.stockBetaHC);
    let hcBond = 0;
    let hcCash = 0;
    if (params.bondDuration > 0 && nonStockHC > 0) {
      const bondFraction = Math.min(1, durationEarnings / params.bondDuration);
      hcBond = nonStockHC * bondFraction;
      hcCash = nonStockHC * (1 - bondFraction);
    } else {
      hcCash = nonStockHC;
    }

    // Target allocation
    const [targetStock, targetBond, targetCash] = computeFullMertonAllocationConstrained(
      params.muStock, computeMuBond(params), params.sigmaS, params.sigmaR,
      params.rho, params.bondDuration, params.gamma
    );

    const totalWealth = financialWealth[i] + humanCapital;
    totalWealthArr[i] = totalWealth;

    // Portfolio weights using normalize helper
    const targetFinStocks = targetStock * totalWealth - hcStock;
    const targetFinBonds = targetBond * totalWealth - hcBond;
    const targetFinCash = targetCash * totalWealth - hcCash;

    const [stockWeight, bondWeight, cashWeight] = normalizePortfolioWeights(
      targetFinStocks, targetFinBonds, targetFinCash,
      financialWealth[i], targetStock, targetBond, targetCash,
      false  // no leverage
    );

    // Consumption decision based on rule
    const netWorth = humanCapital + financialWealth[i] - pvExpenses;

    if (scenario.consumptionRule === 'adaptive') {
      // Adaptive: consume based on net worth, always protect subsistence
      const expectedReturn = currentRate + stockWeight * params.muStock;
      const consumptionRate = expectedReturn + 0.01;
      variableConsumption[i] = Math.max(0, consumptionRate * netWorth);
      totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

      if (earnings[i] > 0 && totalConsumption[i] > earnings[i]) {
        // Working years: cap consumption at earnings (can't borrow against HC)
        totalConsumption[i] = earnings[i];
        variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
      } else if (earnings[i] === 0) {
        // Retirement: cap consumption at financial wealth
        const fw = financialWealth[i];
        if (subsistenceConsumption[i] > fw) {
          // Bankruptcy: can't even meet subsistence
          if (!defaulted) {
            defaulted = true;
            defaultAge = params.startAge + i;
          }
          // Consume whatever wealth remains
          totalConsumption[i] = fw;
          subsistenceConsumption[i] = fw;
          variableConsumption[i] = 0;
        } else if (totalConsumption[i] > fw) {
          // Can meet subsistence but not variable consumption
          totalConsumption[i] = fw;
          variableConsumption[i] = fw - subsistenceConsumption[i];
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
          // Fallback: consume from earnings
          variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
          totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];
          if (totalConsumption[i] > earnings[i]) {
            totalConsumption[i] = earnings[i];
            variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
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

    // Realized returns with shocks
    const stockReturn = currentRate + params.muStock + params.sigmaS * stockShock;
    cumStockReturn *= (1 + stockReturn);
    cumulativeStockReturn[i] = cumStockReturn;

    if (i < totalYears - 1) {
      const bondReturn = currentRate - params.bondDuration * params.sigmaR * rateShock;
      const cashReturn = currentRate;

      const portfolioReturn = stockWeight * stockReturn +
                             bondWeight * bondReturn +
                             cashWeight * cashReturn;

      financialWealth[i + 1] = Math.max(0, financialWealth[i] * (1 + portfolioReturn) + savings);
    }
  }

  return {
    ages,
    financialWealth,
    totalWealth: totalWealthArr,
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
// Strategy Comparison: Optimal vs Rule of Thumb
// =============================================================================

function computeOptimalStrategy(
  params: Params,
  rand: () => number,
  badReturnsEarly: boolean = false
): StrategyResult {
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const earnings = Array(totalYears).fill(0);
  const expenses = Array(totalYears).fill(0);

  const earningsProfile = computeEarningsProfile(params);
  const expenseProfile = computeExpenseProfile(params);

  for (let i = 0; i < workingYears; i++) {
    earnings[i] = earningsProfile[i];
    expenses[i] = expenseProfile.working[i];
  }
  for (let i = workingYears; i < totalYears; i++) {
    expenses[i] = expenseProfile.retirement[i - workingYears];
  }

  let latentRate = params.rBar;  // Latent rate follows pure random walk
  let currentRate = params.rBar; // Observed rate = capped latent rate
  const phi = params.phi;

  const financialWealth = Array(totalYears).fill(0);
  const subsistenceConsumption = [...expenses];
  const variableConsumption = Array(totalYears).fill(0);
  const totalConsumption = Array(totalYears).fill(0);
  const stockWeightArr = Array(totalYears).fill(0);
  const bondWeightArr = Array(totalYears).fill(0);
  const cashWeightArr = Array(totalYears).fill(0);
  const savingsArr = Array(totalYears).fill(0);
  const cumulativeStockReturn = Array(totalYears).fill(1);
  const interestRateArr = Array(totalYears).fill(params.rBar);

  financialWealth[0] = params.initialWealth;

  let defaulted = false;
  let defaultAge: number | null = null;
  let cumStockReturn = 1;

  for (let i = 0; i < totalYears; i++) {
    let [stockShock, rateShock] = generateCorrelatedShocks(rand, params.rho);

    if (badReturnsEarly && i >= workingYears && i < workingYears + 10) {
      stockShock = -Math.abs(stockShock) * 0.5 - 0.3; // Force moderately negative (~-5% avg/year)
    }

    // Update interest rate using random walk (phi=1.0)
    [latentRate, currentRate] = updateInterestRate(latentRate, params.sigmaR, rateShock);
    interestRateArr[i] = currentRate;

    // Compute PVs for optimal strategy
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }
    const remainingExpenses = expenses.slice(i);

    const pvEarnings = computePresentValue(remainingEarnings, currentRate, phi, params.rBar);
    const pvExpenses = computePresentValue(remainingExpenses, currentRate, phi, params.rBar);
    const durationEarnings = computeDuration(remainingEarnings, currentRate, phi, params.rBar);

    const humanCapital = pvEarnings;

    // HC decomposition
    const hcStock = humanCapital * params.stockBetaHC;
    const nonStockHC = humanCapital * (1 - params.stockBetaHC);
    let hcBond = 0;
    let hcCash = 0;
    if (params.bondDuration > 0 && nonStockHC > 0) {
      const bondFraction = Math.min(1, durationEarnings / params.bondDuration);
      hcBond = nonStockHC * bondFraction;
      hcCash = nonStockHC * (1 - bondFraction);
    } else {
      hcCash = nonStockHC;
    }

    // Optimal allocation
    const [targetStock, targetBond, targetCash] = computeFullMertonAllocationConstrained(
      params.muStock, computeMuBond(params), params.sigmaS, params.sigmaR,
      params.rho, params.bondDuration, params.gamma
    );

    const totalWealth = financialWealth[i] + humanCapital;

    // Portfolio weights using normalize helper
    const targetFinStocks = targetStock * totalWealth - hcStock;
    const targetFinBonds = targetBond * totalWealth - hcBond;
    const targetFinCash = targetCash * totalWealth - hcCash;

    const [stockWeight, bondWeight, cashWeight] = normalizePortfolioWeights(
      targetFinStocks, targetFinBonds, targetFinCash,
      financialWealth[i], targetStock, targetBond, targetCash,
      false  // no leverage
    );

    stockWeightArr[i] = stockWeight;
    bondWeightArr[i] = bondWeight;
    cashWeightArr[i] = cashWeight;

    // Adaptive consumption
    const netWorth = humanCapital + financialWealth[i] - pvExpenses;
    const expectedReturn = currentRate + stockWeight * params.muStock;
    const consumptionRate = expectedReturn + 0.01;

    variableConsumption[i] = Math.max(0, consumptionRate * netWorth);
    totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

    if (earnings[i] > 0 && totalConsumption[i] > earnings[i]) {
      // Working years: cap consumption at earnings (can't borrow against HC)
      totalConsumption[i] = earnings[i];
      variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
    } else if (earnings[i] === 0) {
      // Retirement: cap consumption at financial wealth
      const fw = financialWealth[i];
      if (subsistenceConsumption[i] > fw) {
        // Bankruptcy: can't even meet subsistence
        if (!defaulted) {
          defaulted = true;
          defaultAge = params.startAge + i;
        }
        // Consume whatever wealth remains
        totalConsumption[i] = fw;
        subsistenceConsumption[i] = fw;
        variableConsumption[i] = 0;
      } else if (totalConsumption[i] > fw) {
        // Can meet subsistence but not variable consumption
        totalConsumption[i] = fw;
        variableConsumption[i] = fw - subsistenceConsumption[i];
      }
    }

    savingsArr[i] = earnings[i] - totalConsumption[i];

    // Wealth accumulation
    const stockReturn = currentRate + params.muStock + params.sigmaS * stockShock;
    cumStockReturn *= (1 + stockReturn);
    cumulativeStockReturn[i] = cumStockReturn;

    if (i < totalYears - 1) {
      const bondReturn = currentRate - params.bondDuration * params.sigmaR * rateShock;
      const cashReturn = currentRate;

      const portfolioReturn = stockWeight * stockReturn +
                             bondWeight * bondReturn +
                             cashWeight * cashReturn;

      financialWealth[i + 1] = Math.max(0, financialWealth[i] * (1 + portfolioReturn) + savingsArr[i]);
    }
  }

  return {
    ages,
    financialWealth,
    totalConsumption,
    subsistenceConsumption,
    variableConsumption,
    stockWeight: stockWeightArr,
    bondWeight: bondWeightArr,
    cashWeight: cashWeightArr,
    savings: savingsArr,
    defaulted,
    defaultAge,
    terminalWealth: financialWealth[financialWealth.length - 1],
    cumulativeStockReturn,
    interestRate: interestRateArr,
  };
}

function computeRuleOfThumbStrategy(
  params: Params,
  rand: () => number,
  badReturnsEarly: boolean = false
): StrategyResult {
  const totalYears = params.endAge - params.startAge;
  const workingYears = params.retirementAge - params.startAge;

  const ages = Array.from({ length: totalYears }, (_, i) => params.startAge + i);
  const earnings = Array(totalYears).fill(0);
  const expenses = Array(totalYears).fill(0);

  const earningsProfile = computeEarningsProfile(params);
  const expenseProfile = computeExpenseProfile(params);

  for (let i = 0; i < workingYears; i++) {
    earnings[i] = earningsProfile[i];
    expenses[i] = expenseProfile.working[i];
  }
  for (let i = workingYears; i < totalYears; i++) {
    expenses[i] = expenseProfile.retirement[i - workingYears];
  }

  let latentRate = params.rBar;  // Latent rate follows pure random walk
  let currentRate = params.rBar; // Observed rate = capped latent rate

  const financialWealth = Array(totalYears).fill(0);
  const subsistenceConsumption = [...expenses];
  const variableConsumption = Array(totalYears).fill(0);
  const totalConsumption = Array(totalYears).fill(0);
  const stockWeightArr = Array(totalYears).fill(0);
  const bondWeightArr = Array(totalYears).fill(0);
  const cashWeightArr = Array(totalYears).fill(0);
  const savingsArr = Array(totalYears).fill(0);
  const cumulativeStockReturn = Array(totalYears).fill(1);
  const interestRateArr = Array(totalYears).fill(params.rBar);

  financialWealth[0] = params.initialWealth;

  let defaulted = false;
  let defaultAge: number | null = null;
  let fourPercentWithdrawal = 0;
  let retirementAllocation = { stock: 0, bond: 0, cash: 0 };
  let cumStockReturn = 1;

  for (let i = 0; i < totalYears; i++) {
    const age = params.startAge + i;

    let [stockShock, rateShock] = generateCorrelatedShocks(rand, params.rho);

    if (badReturnsEarly && i >= workingYears && i < workingYears + 10) {
      stockShock = -Math.abs(stockShock) * 0.5 - 0.3; // Force moderately negative (~-5% avg/year)
    }

    // Update interest rate using random walk (phi=1.0)
    [latentRate, currentRate] = updateInterestRate(latentRate, params.sigmaR, rateShock);
    interestRateArr[i] = currentRate;

    // Rule of Thumb allocation: (100 - age)% stocks, rest split 50/50 cash/bonds
    let stockWeight: number;
    let bondWeight: number;
    let cashWeight: number;

    if (i < workingYears) {
      // During working years: 100 - age in stocks
      stockWeight = Math.max(0, Math.min(1, (100 - age) / 100));
      const fixedIncome = 1 - stockWeight;
      bondWeight = fixedIncome * 0.5;  // Half in long-term bonds
      cashWeight = fixedIncome * 0.5;  // Half in cash
    } else {
      // At retirement: freeze allocation
      if (i === workingYears) {
        retirementAllocation.stock = Math.max(0, Math.min(1, (100 - age) / 100));
        const fixedIncome = 1 - retirementAllocation.stock;
        retirementAllocation.bond = fixedIncome * 0.5;
        retirementAllocation.cash = fixedIncome * 0.5;
        // Set 4% withdrawal
        fourPercentWithdrawal = 0.04 * financialWealth[i];
      }
      stockWeight = retirementAllocation.stock;
      bondWeight = retirementAllocation.bond;
      cashWeight = retirementAllocation.cash;
    }

    stockWeightArr[i] = stockWeight;
    bondWeightArr[i] = bondWeight;
    cashWeightArr[i] = cashWeight;

    // Consumption
    if (i < workingYears) {
      // Save 20% of income, consume the rest
      const targetSavings = 0.20 * earnings[i];
      totalConsumption[i] = earnings[i] - targetSavings;
      variableConsumption[i] = Math.max(0, totalConsumption[i] - subsistenceConsumption[i]);
      savingsArr[i] = targetSavings;
    } else {
      // 4% rule in retirement
      totalConsumption[i] = fourPercentWithdrawal;
      variableConsumption[i] = Math.max(0, fourPercentWithdrawal - subsistenceConsumption[i]);

      // Check if we can meet the withdrawal
      if (financialWealth[i] < fourPercentWithdrawal) {
        totalConsumption[i] = Math.max(0, financialWealth[i]);
        variableConsumption[i] = Math.max(0, totalConsumption[i] - subsistenceConsumption[i]);

        if (financialWealth[i] < subsistenceConsumption[i]) {
          if (!defaulted) {
            defaulted = true;
            defaultAge = age;
          }
          totalConsumption[i] = Math.max(0, financialWealth[i]);
          variableConsumption[i] = 0;
        }
      }

      savingsArr[i] = -totalConsumption[i]; // Negative savings = withdrawal
    }

    // Wealth accumulation
    const stockReturn = currentRate + params.muStock + params.sigmaS * stockShock;
    cumStockReturn *= (1 + stockReturn);
    cumulativeStockReturn[i] = cumStockReturn;

    if (i < totalYears - 1) {
      const bondReturn = currentRate - params.bondDuration * params.sigmaR * rateShock;
      const cashReturn = currentRate;

      const portfolioReturn = stockWeight * stockReturn +
                             bondWeight * bondReturn +
                             cashWeight * cashReturn;

      if (i < workingYears) {
        financialWealth[i + 1] = Math.max(0, financialWealth[i] * (1 + portfolioReturn) + savingsArr[i]);
      } else {
        financialWealth[i + 1] = Math.max(0, financialWealth[i] * (1 + portfolioReturn) - totalConsumption[i]);
      }
    }
  }

  return {
    ages,
    financialWealth,
    totalConsumption,
    subsistenceConsumption,
    variableConsumption,
    stockWeight: stockWeightArr,
    bondWeight: bondWeightArr,
    cashWeight: cashWeightArr,
    savings: savingsArr,
    defaulted,
    defaultAge,
    terminalWealth: financialWealth[financialWealth.length - 1],
    cumulativeStockReturn,
    interestRate: interestRateArr,
  };
}

function runStrategyComparison(
  params: Params,
  numRuns: number = 50,
  baseSeed: number = 42,
  badReturnsEarly: boolean = false
): { optimal: StrategyResult[]; ruleOfThumb: StrategyResult[] } {
  const optimalRuns: StrategyResult[] = [];
  const ruleOfThumbRuns: StrategyResult[] = [];

  for (let run = 0; run < numRuns; run++) {
    const rand1 = mulberry32(baseSeed + run * 1000);
    const rand2 = mulberry32(baseSeed + run * 1000);

    optimalRuns.push(computeOptimalStrategy(params, rand1, badReturnsEarly));
    ruleOfThumbRuns.push(computeRuleOfThumbStrategy(params, rand2, badReturnsEarly));
  }

  return { optimal: optimalRuns, ruleOfThumb: ruleOfThumbRuns };
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
  subsistence: '#95a5a6',
  variable: '#f39c12',
};

// Number formatters
const formatDollarM = (value: number) => `$${(value / 1000).toFixed(2)}M`;
const formatDollarK = (value: number) => `$${Math.round(value)}k`;
const formatDollar = (value: number) => Math.round(value).toLocaleString();
const formatPercent = (value: number) => `${Math.round(value)}`;
const formatYears = (value: number) => value.toFixed(1);

const dollarMTooltipFormatter = (value: number | undefined) => value !== undefined ? formatDollarM(value) : '';
const dollarKTooltipFormatter = (value: number | undefined) => value !== undefined ? formatDollarK(value) : '';
const dollarTooltipFormatter = (value: number | undefined) => value !== undefined ? `$${formatDollar(value)}k` : '';
const percentTooltipFormatter = (value: number | undefined) => value !== undefined ? `${Math.round(value)}%` : '';
const yearsTooltipFormatter = (value: number | undefined) => value !== undefined ? `${formatYears(value)} yrs` : '';

function ChartSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '24px' }}>
      <h3 style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '12px', color: '#2c3e50' }}>
        {title}
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '16px' }}>
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
      <div style={{ fontSize: '12px', fontWeight: '500', marginBottom: '8px', color: '#555' }}>
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

  // Default parameters
  const [params, setParams] = useState<Params>({
    startAge: 25,
    retirementAge: 65,
    endAge: 85,
    initialEarnings: 100,
    earningsGrowth: 0.02,
    earningsHumpAge: 50,
    earningsDecline: 0.01,
    baseExpenses: 60,
    expenseGrowth: 0.01,
    retirementExpenses: 80,
    stockBetaHC: 0.1,
    gamma: 2,
    initialWealth: 1,
    rBar: 0.02,
    muStock: 0.03,
    bondSharpe: 0.10,
    sigmaS: 0.18,
    sigmaR: 0.006,
    rho: 0.0,
    bondDuration: 20,
    phi: 1.0,
  });

  const updateParam = (key: keyof Params, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Compute lifecycle results
  const result = useMemo(() => computeLifecycleMedianPath(params), [params]);

  // Compute Monte Carlo results (only when on that page for performance)
  const mcResult = useMemo(() => {
    if (currentPage !== 'monteCarlo') return null;
    return computeMonteCarloSimulation(params, 50);
  }, [params, currentPage]);

  // Scenario state
  const [scenarioType, setScenarioType] = useState<'sequenceRisk' | 'rateShock' | 'optimalVsRuleOfThumb'>('optimalVsRuleOfThumb');
  const [rateShockAge, setRateShockAge] = useState(50);
  const [rateShockMagnitude, setRateShockMagnitude] = useState(-0.02);
  const [scenarioRetirementAge, setScenarioRetirementAge] = useState(params.retirementAge);
  const [scenarioEndAge, setScenarioEndAge] = useState(params.endAge);
  // scenarioBadReturns is handled by the scenarioType - sequenceRisk forces bad returns

  // Create modified params for scenarios with custom ages
  const scenarioParams = useMemo(() => ({
    ...params,
    retirementAge: scenarioRetirementAge,
    endAge: scenarioEndAge,
  }), [params, scenarioRetirementAge, scenarioEndAge]);

  // Compute scenario comparisons
  const scenarioResults = useMemo(() => {
    if (currentPage !== 'scenarios') return null;

    if (scenarioType === 'sequenceRisk') {
      // Compare adaptive vs 4% rule with bad early returns
      return runScenarioComparison(scenarioParams, 50, 42, true, 0, 0);
    } else if (scenarioType === 'rateShock') {
      // Rate shock scenario
      return runScenarioComparison(scenarioParams, 50, 42, false, rateShockAge, rateShockMagnitude);
    }
    return null;
  }, [scenarioParams, currentPage, scenarioType, rateShockAge, rateShockMagnitude]);

  // Strategy comparison (Optimal vs Rule of Thumb) - Normal market conditions
  const strategyResults = useMemo(() => {
    if (currentPage !== 'scenarios' || scenarioType !== 'optimalVsRuleOfThumb') return null;
    return runStrategyComparison(scenarioParams, 50, 42, false); // Normal market, no forced bad returns
  }, [scenarioParams, currentPage, scenarioType]);

  // Compute percentile data for scenario charts (fan charts instead of 50 lines)
  const scenarioPercentileData = useMemo(() => {
    if (!scenarioResults) return null;
    return scenarioResults.adaptive[0].ages.map((age, i) => {
      const returns = scenarioResults.adaptive.map(r => Math.log(r.cumulativeStockReturn[i]));
      const rates = scenarioResults.adaptive.map(r => r.interestRate[i] * 100);

      // Financial wealth - Adaptive
      const fwAdaptive = scenarioResults.adaptive.map(r => r.financialWealth[i]);
      const fw_adapt_p05 = computePercentile(fwAdaptive, 5);
      const fw_adapt_p25 = computePercentile(fwAdaptive, 25);
      const fw_adapt_p50 = computePercentile(fwAdaptive, 50);
      const fw_adapt_p75 = computePercentile(fwAdaptive, 75);
      const fw_adapt_p95 = computePercentile(fwAdaptive, 95);

      // Financial wealth - 4% Rule
      const fw4Pct = scenarioResults.fourPercent.map(r => r.financialWealth[i]);
      const fw_4pct_p05 = computePercentile(fw4Pct, 5);
      const fw_4pct_p25 = computePercentile(fw4Pct, 25);
      const fw_4pct_p50 = computePercentile(fw4Pct, 50);
      const fw_4pct_p75 = computePercentile(fw4Pct, 75);
      const fw_4pct_p95 = computePercentile(fw4Pct, 95);

      // Total wealth - Adaptive
      const twAdaptive = scenarioResults.adaptive.map(r => r.totalWealth[i]);
      const tw_adapt_p05 = computePercentile(twAdaptive, 5);
      const tw_adapt_p25 = computePercentile(twAdaptive, 25);
      const tw_adapt_p50 = computePercentile(twAdaptive, 50);
      const tw_adapt_p75 = computePercentile(twAdaptive, 75);
      const tw_adapt_p95 = computePercentile(twAdaptive, 95);

      // Total wealth - 4% Rule
      const tw4Pct = scenarioResults.fourPercent.map(r => r.totalWealth[i]);
      const tw_4pct_p05 = computePercentile(tw4Pct, 5);
      const tw_4pct_p25 = computePercentile(tw4Pct, 25);
      const tw_4pct_p50 = computePercentile(tw4Pct, 50);
      const tw_4pct_p75 = computePercentile(tw4Pct, 75);
      const tw_4pct_p95 = computePercentile(tw4Pct, 95);

      // Consumption - Adaptive
      const consAdaptive = scenarioResults.adaptive.map(r => r.totalConsumption[i]);
      const cons_adapt_p05 = computePercentile(consAdaptive, 5);
      const cons_adapt_p25 = computePercentile(consAdaptive, 25);
      const cons_adapt_p50 = computePercentile(consAdaptive, 50);
      const cons_adapt_p75 = computePercentile(consAdaptive, 75);
      const cons_adapt_p95 = computePercentile(consAdaptive, 95);

      // Consumption - 4% Rule
      const cons4Pct = scenarioResults.fourPercent.map(r => r.totalConsumption[i]);
      const cons_4pct_p05 = computePercentile(cons4Pct, 5);
      const cons_4pct_p25 = computePercentile(cons4Pct, 25);
      const cons_4pct_p50 = computePercentile(cons4Pct, 50);
      const cons_4pct_p75 = computePercentile(cons4Pct, 75);
      const cons_4pct_p95 = computePercentile(cons4Pct, 95);

      return {
        age,
        // Stock return percentiles (log scale)
        sr_p05: computePercentile(returns, 5),
        sr_p25: computePercentile(returns, 25),
        sr_p50: computePercentile(returns, 50),
        sr_p75: computePercentile(returns, 75),
        sr_p95: computePercentile(returns, 95),
        // Bands for stacking
        sr_band_5_25: computePercentile(returns, 25) - computePercentile(returns, 5),
        sr_band_25_75: computePercentile(returns, 75) - computePercentile(returns, 25),
        sr_band_75_95: computePercentile(returns, 95) - computePercentile(returns, 75),
        // Interest rate percentiles
        rate_p05: computePercentile(rates, 5),
        rate_p25: computePercentile(rates, 25),
        rate_p50: computePercentile(rates, 50),
        rate_p75: computePercentile(rates, 75),
        rate_p95: computePercentile(rates, 95),
        // Bands
        rate_band_5_25: computePercentile(rates, 25) - computePercentile(rates, 5),
        rate_band_25_75: computePercentile(rates, 75) - computePercentile(rates, 25),
        rate_band_75_95: computePercentile(rates, 95) - computePercentile(rates, 75),
        // Financial Wealth - Adaptive
        fw_adapt_p05, fw_adapt_p25, fw_adapt_p50, fw_adapt_p75, fw_adapt_p95,
        fw_adapt_band_5_25: fw_adapt_p25 - fw_adapt_p05,
        fw_adapt_band_25_75: fw_adapt_p75 - fw_adapt_p25,
        fw_adapt_band_75_95: fw_adapt_p95 - fw_adapt_p75,
        // Financial Wealth - 4% Rule
        fw_4pct_p05, fw_4pct_p25, fw_4pct_p50, fw_4pct_p75, fw_4pct_p95,
        fw_4pct_band_5_25: fw_4pct_p25 - fw_4pct_p05,
        fw_4pct_band_25_75: fw_4pct_p75 - fw_4pct_p25,
        fw_4pct_band_75_95: fw_4pct_p95 - fw_4pct_p75,
        // Total Wealth - Adaptive
        tw_adapt_p05, tw_adapt_p25, tw_adapt_p50, tw_adapt_p75, tw_adapt_p95,
        tw_adapt_band_5_25: tw_adapt_p25 - tw_adapt_p05,
        tw_adapt_band_25_75: tw_adapt_p75 - tw_adapt_p25,
        tw_adapt_band_75_95: tw_adapt_p95 - tw_adapt_p75,
        // Total Wealth - 4% Rule
        tw_4pct_p05, tw_4pct_p25, tw_4pct_p50, tw_4pct_p75, tw_4pct_p95,
        tw_4pct_band_5_25: tw_4pct_p25 - tw_4pct_p05,
        tw_4pct_band_25_75: tw_4pct_p75 - tw_4pct_p25,
        tw_4pct_band_75_95: tw_4pct_p95 - tw_4pct_p75,
        // Consumption - Adaptive
        cons_adapt_p05, cons_adapt_p25, cons_adapt_p50, cons_adapt_p75, cons_adapt_p95,
        cons_adapt_band_5_25: cons_adapt_p25 - cons_adapt_p05,
        cons_adapt_band_25_75: cons_adapt_p75 - cons_adapt_p25,
        cons_adapt_band_75_95: cons_adapt_p95 - cons_adapt_p75,
        // Consumption - 4% Rule
        cons_4pct_p05, cons_4pct_p25, cons_4pct_p50, cons_4pct_p75, cons_4pct_p95,
        cons_4pct_band_5_25: cons_4pct_p25 - cons_4pct_p05,
        cons_4pct_band_25_75: cons_4pct_p75 - cons_4pct_p25,
        cons_4pct_band_75_95: cons_4pct_p95 - cons_4pct_p75,
        // Subsistence floor (same for both strategies)
        subsistence: scenarioResults.adaptive[0].subsistenceConsumption[i],
      };
    });
  }, [scenarioResults]);

  // Prepare chart data
  const chartData = useMemo(() => {
    return result.ages.map((age, i) => ({
      age,
      earnings: result.earnings[i],
      expenses: result.expenses[i],
      pvEarnings: result.pvEarnings[i],
      pvExpenses: -result.pvExpenses[i],
      durationEarnings: result.durationEarnings[i],
      durationExpenses: -result.durationExpenses[i],
      humanCapital: result.humanCapital[i],
      financialWealth: result.financialWealth[i],
      hcStock: result.hcStock[i],
      hcBond: result.hcBond[i],
      hcCash: result.hcCash[i],
      expBond: -result.expBond[i],
      expCash: -result.expCash[i],
      netStock: result.hcStock[i],
      netBond: result.hcBond[i] - result.expBond[i],
      netCash: result.hcCash[i] - result.expCash[i],
      stockWeight: result.stockWeight[i] * 100,
      bondWeight: result.bondWeight[i] * 100,
      cashWeight: result.cashWeight[i] * 100,
      subsistence: result.subsistenceConsumption[i],
      variable: result.variableConsumption[i],
      totalConsumption: result.totalConsumption[i],
    }));
  }, [result]);

  // Prepare Monte Carlo chart data with band ranges for proper stacking
  const mcChartData = useMemo(() => {
    if (!mcResult) return [];
    return mcResult.ages.map((age, i) => ({
      age,
      // Consumption percentiles and bands
      consumption_p05: mcResult.consumption_p05[i],
      consumption_p25: mcResult.consumption_p25[i],
      consumption_p50: mcResult.consumption_p50[i],
      consumption_p75: mcResult.consumption_p75[i],
      consumption_p95: mcResult.consumption_p95[i],
      // Band ranges for proper area rendering
      consumption_band_5_25: mcResult.consumption_p25[i] - mcResult.consumption_p05[i],
      consumption_band_25_75: mcResult.consumption_p75[i] - mcResult.consumption_p25[i],
      consumption_band_75_95: mcResult.consumption_p95[i] - mcResult.consumption_p75[i],
      // Financial wealth percentiles and bands
      fw_p05: mcResult.financialWealth_p05[i],
      fw_p25: mcResult.financialWealth_p25[i],
      fw_p50: mcResult.financialWealth_p50[i],
      fw_p75: mcResult.financialWealth_p75[i],
      fw_p95: mcResult.financialWealth_p95[i],
      fw_band_5_25: mcResult.financialWealth_p25[i] - mcResult.financialWealth_p05[i],
      fw_band_25_75: mcResult.financialWealth_p75[i] - mcResult.financialWealth_p25[i],
      fw_band_75_95: mcResult.financialWealth_p95[i] - mcResult.financialWealth_p75[i],
      // Total wealth percentiles (FW + HC)
      tw_p05: mcResult.totalWealth_p05[i],
      tw_p25: mcResult.totalWealth_p25[i],
      tw_p50: mcResult.totalWealth_p50[i],
      tw_p75: mcResult.totalWealth_p75[i],
      tw_p95: mcResult.totalWealth_p95[i],
      tw_band_5_25: mcResult.totalWealth_p25[i] - mcResult.totalWealth_p05[i],
      tw_band_25_75: mcResult.totalWealth_p75[i] - mcResult.totalWealth_p25[i],
      tw_band_75_95: mcResult.totalWealth_p95[i] - mcResult.totalWealth_p75[i],
      // Net Worth percentiles (HC + FW - expenses)
      nw_p05: mcResult.netWorth_p05[i],
      nw_p25: mcResult.netWorth_p25[i],
      nw_p50: mcResult.netWorth_p50[i],
      nw_p75: mcResult.netWorth_p75[i],
      nw_p95: mcResult.netWorth_p95[i],
      nw_band_5_25: mcResult.netWorth_p25[i] - mcResult.netWorth_p05[i],
      nw_band_25_75: mcResult.netWorth_p75[i] - mcResult.netWorth_p25[i],
      nw_band_75_95: mcResult.netWorth_p95[i] - mcResult.netWorth_p75[i],
      // Stock return percentiles (cumulative, using log scale for bands)
      sr_p05: Math.log(mcResult.stockReturn_p05[i]),
      sr_p25: Math.log(mcResult.stockReturn_p25[i]),
      sr_p50: Math.log(mcResult.stockReturn_p50[i]),
      sr_p75: Math.log(mcResult.stockReturn_p75[i]),
      sr_p95: Math.log(mcResult.stockReturn_p95[i]),
      sr_band_5_25: Math.log(mcResult.stockReturn_p25[i]) - Math.log(mcResult.stockReturn_p05[i]),
      sr_band_25_75: Math.log(mcResult.stockReturn_p75[i]) - Math.log(mcResult.stockReturn_p25[i]),
      sr_band_75_95: Math.log(mcResult.stockReturn_p95[i]) - Math.log(mcResult.stockReturn_p75[i]),
      // Interest rate percentiles (in percentage)
      ir_p05: mcResult.interestRate_p05[i] * 100,
      ir_p25: mcResult.interestRate_p25[i] * 100,
      ir_p50: mcResult.interestRate_p50[i] * 100,
      ir_p75: mcResult.interestRate_p75[i] * 100,
      ir_p95: mcResult.interestRate_p95[i] * 100,
      ir_band_5_25: (mcResult.interestRate_p25[i] - mcResult.interestRate_p05[i]) * 100,
      ir_band_25_75: (mcResult.interestRate_p75[i] - mcResult.interestRate_p25[i]) * 100,
      ir_band_75_95: (mcResult.interestRate_p95[i] - mcResult.interestRate_p75[i]) * 100,
    }));
  }, [mcResult]);

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
            min={params.retirementAge + 5} max={100} step={1} suffix="" decimals={0}
          />
          <StepperInput
            label="Current Wealth"
            value={params.initialWealth}
            onChange={(v) => updateParam('initialWealth', v)}
            min={0} max={5000} step={10} suffix="$k" decimals={0}
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
            min={0} max={5} step={0.5} suffix="%" decimals={1}
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
            min={0} max={0.5} step={0.05} suffix="" decimals={2}
          />
          <StepperInput
            label="Stock volatility (σ)"
            value={params.sigmaS * 100}
            onChange={(v) => updateParam('sigmaS', v / 100)}
            min={10} max={30} step={1} suffix="%" decimals={0}
          />
          <StepperInput
            label="Rate shock vol (σᵣ)"
            value={params.sigmaR * 100}
            onChange={(v) => updateParam('sigmaR', v / 100)}
            min={0.5} max={3} step={0.1} suffix="%" decimals={1}
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
        </ParamGroup>

        <ParamGroup title="Income">
          <StepperInput
            label="Initial earnings"
            value={params.initialEarnings}
            onChange={(v) => updateParam('initialEarnings', v)}
            min={50} max={200} step={10} suffix="$k" decimals={0}
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
            min={40} max={60} step={1} suffix="" decimals={0}
          />
        </ParamGroup>

        <ParamGroup title="Expenses">
          <StepperInput
            label="Working expenses"
            value={params.baseExpenses}
            onChange={(v) => updateParam('baseExpenses', v)}
            min={30} max={100} step={5} suffix="$k" decimals={0}
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
            min={40} max={120} step={5} suffix="$k" decimals={0}
          />
        </ParamGroup>

        <ParamGroup title="Risk Preferences">
          <StepperInput
            label="Risk aversion (γ)"
            value={params.gamma}
            onChange={(v) => updateParam('gamma', v)}
            min={1} max={10} step={0.5} suffix="" decimals={1}
          />
          <StepperInput
            label="HC stock beta"
            value={params.stockBetaHC}
            onChange={(v) => updateParam('stockBetaHC', v)}
            min={0} max={0.5} step={0.05} suffix="" decimals={2}
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
              onClick={() => setCurrentPage('monteCarlo')}
              style={{
                padding: '8px 16px',
                border: 'none',
                borderRadius: '6px',
                fontSize: '13px',
                fontWeight: currentPage === 'monteCarlo' ? 'bold' : 'normal',
                background: currentPage === 'monteCarlo' ? '#fff' : 'transparent',
                color: currentPage === 'monteCarlo' ? '#2c3e50' : '#666',
                cursor: 'pointer',
                boxShadow: currentPage === 'monteCarlo' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              }}
            >
              Monte Carlo (50 runs)
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
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Line type="monotone" dataKey="earnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="Earnings" />
                <Line type="monotone" dataKey="expenses" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="Expenses" />
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
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Line type="monotone" dataKey="pvEarnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="PV Earnings" />
                <Line type="monotone" dataKey="pvExpenses" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="PV Expenses" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Durations (years)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatYears} />
                <Tooltip formatter={yearsTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
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
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Area type="monotone" dataKey="financialWealth" stackId="1" stroke={COLORS.fw} fill={COLORS.fw} name="Financial Wealth" />
                <Area type="monotone" dataKey="humanCapital" stackId="1" stroke={COLORS.hc} fill={COLORS.hc} name="Human Capital" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Human Capital Decomposition ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Area type="monotone" dataKey="hcCash" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="HC Cash" />
                <Area type="monotone" dataKey="hcBond" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="HC Bond" />
                <Area type="monotone" dataKey="hcStock" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="HC Stock" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Expense Liability Decomposition ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Area type="monotone" dataKey="expCash" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Expense Cash" />
                <Area type="monotone" dataKey="expBond" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Expense Bond" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Net HC minus Expenses ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Area type="monotone" dataKey="netCash" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Net Cash" />
                <Area type="monotone" dataKey="netBond" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Net Bond" />
                <Area type="monotone" dataKey="netStock" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="Net Stock" />
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
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Area type="monotone" dataKey="subsistence" stackId="1" stroke={COLORS.subsistence} fill={COLORS.subsistence} name="Subsistence" />
                <Area type="monotone" dataKey="variable" stackId="1" stroke={COLORS.variable} fill={COLORS.variable} name="Variable" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Portfolio Allocation (%)">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} domain={[0, 100]} tickFormatter={formatPercent} />
                <Tooltip formatter={percentTooltipFormatter} />
                <Legend wrapperStyle={{ fontSize: '10px' }} />
                <Area type="monotone" dataKey="cashWeight" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Cash" />
                <Area type="monotone" dataKey="bondWeight" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Bonds" />
                <Area type="monotone" dataKey="stockWeight" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="Stocks" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </ChartSection>
          </>
        )}

        {currentPage === 'monteCarlo' && mcResult && (
          <>
            <div style={{
              background: '#e8f4f8',
              padding: '12px 16px',
              borderRadius: '8px',
              marginBottom: '24px',
              fontSize: '13px',
              color: '#2c3e50',
            }}>
              <strong>Monte Carlo Simulation:</strong> 50 independent runs with stochastic shocks to interest rates and returns.
              Shocks affect rates, which change forward-looking PVs and durations, affecting consumption and portfolio decisions.
              Charts show percentile lines: 5th, 25th, median (50th), 75th, and 95th percentiles.
            </div>

            {/* Consumption Fan Chart */}
            <ChartSection title="Consumption Evolution">
              <ChartCard title="Total Consumption Distribution ($k)">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={mcChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarK} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarKTooltipFormatter} />
                    <Line type="monotone" dataKey="consumption_p05" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="consumption_p25" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="consumption_p50" stroke={COLORS.variable} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="consumption_p75" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="consumption_p95" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </ChartSection>

            {/* Financial Wealth Fan Chart */}
            <ChartSection title="Financial Wealth Evolution">
              <ChartCard title="Financial Wealth Distribution ($M)">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={mcChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarMTooltipFormatter} />
                    <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="fw_p05" stroke={COLORS.fw} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="fw_p25" stroke={COLORS.fw} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="fw_p50" stroke={COLORS.fw} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="fw_p75" stroke={COLORS.fw} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="fw_p95" stroke={COLORS.fw} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </ChartSection>

            {/* Net Worth Fan Chart */}
            <ChartSection title="Net Worth Evolution (HC + FW - Expenses)">
              <ChartCard title="Net Worth Distribution ($M)">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={mcChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarMTooltipFormatter} />
                    <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="nw_p05" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="nw_p25" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="nw_p50" stroke={COLORS.variable} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="nw_p75" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="nw_p95" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            </ChartSection>

            {/* Terminal Wealth Distribution */}
            <ChartSection title={`Terminal Values (Age ${params.endAge - 1})`}>
              <ChartCard title="Terminal Financial Wealth Distribution">
                <div style={{ padding: '16px' }}>
                  <div style={{ marginBottom: '16px', fontSize: '12px', color: '#666' }}>
                    Distribution of financial wealth at age {params.endAge - 1} across 50 runs:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', textAlign: 'center' }}>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>5th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.fw }}>
                        {formatDollarM(mcResult.financialWealth_p05[mcResult.financialWealth_p05.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>25th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.fw }}>
                        {formatDollarM(mcResult.financialWealth_p25[mcResult.financialWealth_p25.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>Median</div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.fw }}>
                        {formatDollarM(mcResult.financialWealth_p50[mcResult.financialWealth_p50.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>75th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.fw }}>
                        {formatDollarM(mcResult.financialWealth_p75[mcResult.financialWealth_p75.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>95th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.fw }}>
                        {formatDollarM(mcResult.financialWealth_p95[mcResult.financialWealth_p95.length - 1])}
                      </div>
                    </div>
                  </div>
                  <div style={{ marginTop: '24px' }}>
                    <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px' }}>
                      Runs depleted (FW &lt; $10k): {mcResult.runs.filter(r => r.financialWealth[r.financialWealth.length - 1] < 10).length} of 50
                    </div>
                  </div>
                </div>
              </ChartCard>

              <ChartCard title="Terminal Consumption Distribution">
                <div style={{ padding: '16px' }}>
                  <div style={{ marginBottom: '16px', fontSize: '12px', color: '#666' }}>
                    Distribution of annual consumption at age {params.endAge - 1} across 50 runs:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', textAlign: 'center' }}>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>5th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.variable }}>
                        {formatDollarM(mcResult.consumption_p05[mcResult.consumption_p05.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>25th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.variable }}>
                        {formatDollarM(mcResult.consumption_p25[mcResult.consumption_p25.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>Median</div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.variable }}>
                        {formatDollarM(mcResult.consumption_p50[mcResult.consumption_p50.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>75th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.variable }}>
                        {formatDollarM(mcResult.consumption_p75[mcResult.consumption_p75.length - 1])}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#999' }}>95th %ile</div>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.variable }}>
                        {formatDollarM(mcResult.consumption_p95[mcResult.consumption_p95.length - 1])}
                      </div>
                    </div>
                  </div>
                </div>
              </ChartCard>
            </ChartSection>

            {/* Market Conditions Charts */}
            <ChartSection title="Market Conditions">
              <ChartCard title="Cumulative Stock Returns - Percentile Bands (Log Scale)">
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={mcChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={(v) => `${(Math.exp(v) * 100).toFixed(0)}%`} domain={['auto', 'auto']} />
                    <Tooltip formatter={(v) => v !== undefined ? [`${(Math.exp(v as number) * 100).toFixed(0)}%`, 'Cumulative Return'] : ['', '']} />
                    <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                    <Area type="monotone" dataKey="sr_p05" stackId="sr" fill="transparent" stroke="transparent" />
                    <Area type="monotone" dataKey="sr_band_5_25" stackId="sr" fill={COLORS.stock} fillOpacity={0.15} stroke="transparent" name="5th-25th" />
                    <Area type="monotone" dataKey="sr_band_25_75" stackId="sr" fill={COLORS.stock} fillOpacity={0.3} stroke="transparent" name="25th-75th" />
                    <Area type="monotone" dataKey="sr_band_75_95" stackId="sr" fill={COLORS.stock} fillOpacity={0.15} stroke="transparent" name="75th-95th" />
                    <Line type="monotone" dataKey="sr_p50" stroke={COLORS.stock} strokeWidth={2} dot={false} name="Median" />
                    <Line type="monotone" dataKey="sr_p25" stroke={COLORS.stock} strokeWidth={1} strokeDasharray="4 4" dot={false} name="25th %ile" />
                    <Line type="monotone" dataKey="sr_p75" stroke={COLORS.stock} strokeWidth={1} strokeDasharray="4 4" dot={false} name="75th %ile" />
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                  Cumulative stock returns across 50 Monte Carlo runs. Y-axis: 100% = starting value.
                </div>
              </ChartCard>

              <ChartCard title="Interest Rate Paths - Percentile Bands">
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={mcChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={(v) => `${v.toFixed(1)}%`} domain={['auto', 'auto']} />
                    <Tooltip formatter={(v) => v !== undefined ? [`${(v as number).toFixed(2)}%`, 'Interest Rate'] : ['', '']} />
                    <Area type="monotone" dataKey="ir_p05" stackId="ir" fill="transparent" stroke="transparent" />
                    <Area type="monotone" dataKey="ir_band_5_25" stackId="ir" fill={COLORS.bond} fillOpacity={0.15} stroke="transparent" name="5th-25th" />
                    <Area type="monotone" dataKey="ir_band_25_75" stackId="ir" fill={COLORS.bond} fillOpacity={0.3} stroke="transparent" name="25th-75th" />
                    <Area type="monotone" dataKey="ir_band_75_95" stackId="ir" fill={COLORS.bond} fillOpacity={0.15} stroke="transparent" name="75th-95th" />
                    <Line type="monotone" dataKey="ir_p50" stroke={COLORS.bond} strokeWidth={2} dot={false} name="Median" />
                    <Line type="monotone" dataKey="ir_p25" stroke={COLORS.bond} strokeWidth={1} strokeDasharray="4 4" dot={false} name="25th %ile" />
                    <Line type="monotone" dataKey="ir_p75" stroke={COLORS.bond} strokeWidth={1} strokeDasharray="4 4" dot={false} name="75th %ile" />
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                  Interest rate paths showing how rates evolve stochastically over time.
                </div>
              </ChartCard>
            </ChartSection>
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
                Optimal vs Rule of Thumb
              </div>
              <div style={{ fontSize: '12px', color: '#666', marginBottom: '12px' }}>
                Comparing lifecycle-optimal strategy vs common "rules of thumb" (save 20%, 100-age in stocks, 4% withdrawal)
              </div>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                <button
                  onClick={() => setScenarioType('optimalVsRuleOfThumb')}
                  style={{
                    padding: '8px 16px',
                    border: 'none',
                    borderBottom: scenarioType === 'optimalVsRuleOfThumb' ? '3px solid #3498db' : '3px solid transparent',
                    background: scenarioType === 'optimalVsRuleOfThumb' ? '#e8f4f8' : 'transparent',
                    cursor: 'pointer',
                    fontWeight: scenarioType === 'optimalVsRuleOfThumb' ? 'bold' : 'normal',
                    color: scenarioType === 'optimalVsRuleOfThumb' ? '#2c3e50' : '#666',
                  }}
                >
                  Normal Market
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
                  Bad Early Returns
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
                  Interest Rate Shock
                </button>
              </div>

              {/* Age Controls */}
              <div style={{ display: 'flex', gap: '24px', padding: '12px', background: '#f5f5f5', borderRadius: '6px', marginBottom: '12px' }}>
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
            </div>

            {/* Concept Explanation */}
            {scenarioType === 'optimalVsRuleOfThumb' && (
              <div style={{
                background: '#e3f2fd',
                border: '1px solid #2196f3',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '24px',
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#1565c0' }}>
                  Comparing Two Strategies
                </div>
                <div style={{ fontSize: '13px', color: '#1565c0', lineHeight: 1.5 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                    <div>
                      <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>Optimal Strategy:</p>
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
                        <li>Save 20% of income</li>
                        <li>(100 - age)% in stocks</li>
                        <li>Rest split 50/50 cash & bonds</li>
                        <li>Freeze allocation at retirement</li>
                        <li>4% withdrawal rule</li>
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

            {/* Loading state */}
            {scenarioType === 'optimalVsRuleOfThumb' && !strategyResults && (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                Computing 50 simulation runs...
              </div>
            )}
            {(scenarioType === 'sequenceRisk' || scenarioType === 'rateShock') && !scenarioResults && (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                Computing 50 simulation runs...
              </div>
            )}

            {/* Strategy Comparison: Optimal vs Rule of Thumb */}
            {scenarioType === 'optimalVsRuleOfThumb' && strategyResults && (
              <>
                {/* Summary Statistics */}
                <ChartSection title="Comparison Summary">
                  <ChartCard title="Default Risk (Failure to Meet Subsistence)">
                    <div style={{ padding: '24px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
                        <div style={{ textAlign: 'center', padding: '20px', background: '#e8f5e9', borderRadius: '8px' }}>
                          <div style={{ fontSize: '13px', color: '#2e7d32', marginBottom: '8px' }}>Optimal Strategy</div>
                          <div style={{ fontSize: '36px', fontWeight: 'bold', color: '#1b5e20' }}>
                            {strategyResults.optimal.filter(r => r.defaulted).length}
                          </div>
                          <div style={{ fontSize: '12px', color: '#388e3c' }}>of 50 runs defaulted</div>
                        </div>
                        <div style={{ textAlign: 'center', padding: '20px', background: strategyResults.ruleOfThumb.filter(r => r.defaulted).length > 0 ? '#ffebee' : '#e8f5e9', borderRadius: '8px' }}>
                          <div style={{ fontSize: '13px', color: strategyResults.ruleOfThumb.filter(r => r.defaulted).length > 0 ? '#c62828' : '#2e7d32', marginBottom: '8px' }}>Rule of Thumb</div>
                          <div style={{ fontSize: '36px', fontWeight: 'bold', color: strategyResults.ruleOfThumb.filter(r => r.defaulted).length > 0 ? '#b71c1c' : '#1b5e20' }}>
                            {strategyResults.ruleOfThumb.filter(r => r.defaulted).length}
                          </div>
                          <div style={{ fontSize: '12px', color: strategyResults.ruleOfThumb.filter(r => r.defaulted).length > 0 ? '#d32f2f' : '#388e3c' }}>of 50 runs defaulted</div>
                          {strategyResults.ruleOfThumb.filter(r => r.defaulted).length > 0 && (
                            <div style={{ fontSize: '11px', color: '#666', marginTop: '8px' }}>
                              Avg default age: {Math.round(
                                strategyResults.ruleOfThumb
                                  .filter(r => r.defaulted && r.defaultAge !== null)
                                  .reduce((sum, r) => sum + (r.defaultAge || 0), 0) /
                                Math.max(1, strategyResults.ruleOfThumb.filter(r => r.defaulted).length)
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </ChartCard>

                  <ChartCard title="Lifetime Consumption Comparison">
                    <div style={{ padding: '24px' }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
                        <div>
                          <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '12px', color: COLORS.hc }}>Optimal - Total Consumption</div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontSize: '12px' }}>
                            <div>
                              <div style={{ color: '#999' }}>5th %ile</div>
                              <div style={{ fontWeight: 'bold' }}>${formatDollar(computePercentile(strategyResults.optimal.map(r => r.totalConsumption.reduce((a, b) => a + b, 0)), 5))}k</div>
                            </div>
                            <div>
                              <div style={{ color: '#999' }}>Median</div>
                              <div style={{ fontWeight: 'bold' }}>${formatDollar(computePercentile(strategyResults.optimal.map(r => r.totalConsumption.reduce((a, b) => a + b, 0)), 50))}k</div>
                            </div>
                            <div>
                              <div style={{ color: '#999' }}>95th %ile</div>
                              <div style={{ fontWeight: 'bold' }}>${formatDollar(computePercentile(strategyResults.optimal.map(r => r.totalConsumption.reduce((a, b) => a + b, 0)), 95))}k</div>
                            </div>
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '12px', color: COLORS.expenses }}>Rule of Thumb - Total Consumption</div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontSize: '12px' }}>
                            <div>
                              <div style={{ color: '#999' }}>5th %ile</div>
                              <div style={{ fontWeight: 'bold' }}>${formatDollar(computePercentile(strategyResults.ruleOfThumb.map(r => r.totalConsumption.reduce((a, b) => a + b, 0)), 5))}k</div>
                            </div>
                            <div>
                              <div style={{ color: '#999' }}>Median</div>
                              <div style={{ fontWeight: 'bold' }}>${formatDollar(computePercentile(strategyResults.ruleOfThumb.map(r => r.totalConsumption.reduce((a, b) => a + b, 0)), 50))}k</div>
                            </div>
                            <div>
                              <div style={{ color: '#999' }}>95th %ile</div>
                              <div style={{ fontWeight: 'bold' }}>${formatDollar(computePercentile(strategyResults.ruleOfThumb.map(r => r.totalConsumption.reduce((a, b) => a + b, 0)), 95))}k</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </ChartCard>
                </ChartSection>

                {/* Financial Wealth Paths */}
                <ChartSection title="Financial Wealth Over Time">
                  <ChartCard title="Optimal Strategy - Financial Wealth ($M)">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={strategyResults.optimal[0].ages.map((age, i) => {
                        const fwValues = strategyResults.optimal.map(r => r.financialWealth[i]);
                        const p05 = computePercentile(fwValues, 5);
                        const p25 = computePercentile(fwValues, 25);
                        const p50 = computePercentile(fwValues, 50);
                        const p75 = computePercentile(fwValues, 75);
                        const p95 = computePercentile(fwValues, 95);
                        return { age, fw_p05: p05, fw_p25: p25, fw_p50: p50, fw_p75: p75, fw_p95: p95 };
                      })}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                        <Tooltip formatter={dollarMTooltipFormatter} />
                        <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="fw_p05" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                        <Line type="monotone" dataKey="fw_p25" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="fw_p50" stroke={COLORS.hc} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="fw_p75" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                        <Line type="monotone" dataKey="fw_p95" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartCard>

                  <ChartCard title="Rule of Thumb - Financial Wealth ($M)">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={strategyResults.ruleOfThumb[0].ages.map((age, i) => {
                        const fwValues = strategyResults.ruleOfThumb.map(r => r.financialWealth[i]);
                        const p05 = computePercentile(fwValues, 5);
                        const p25 = computePercentile(fwValues, 25);
                        const p50 = computePercentile(fwValues, 50);
                        const p75 = computePercentile(fwValues, 75);
                        const p95 = computePercentile(fwValues, 95);
                        return { age, fw_p05: p05, fw_p25: p25, fw_p50: p50, fw_p75: p75, fw_p95: p95 };
                      })}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                        <Tooltip formatter={dollarMTooltipFormatter} />
                        <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="fw_p05" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                        <Line type="monotone" dataKey="fw_p25" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="fw_p50" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="fw_p75" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                        <Line type="monotone" dataKey="fw_p95" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartCard>
                </ChartSection>

                {/* Total Wealth (Net Worth) = HC + FW - Expenses */}
                <ChartSection title="Total Wealth Over Time (HC + FW - Expenses)">
                  <ChartCard title="Optimal Strategy - Total Wealth ($M)">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={strategyResults.optimal[0].ages.map((age, i) => {
                        const fwValues = strategyResults.optimal.map(r => r.financialWealth[i]);
                        const p05 = computePercentile(fwValues, 5);
                        const p25 = computePercentile(fwValues, 25);
                        const p50 = computePercentile(fwValues, 50);
                        const p75 = computePercentile(fwValues, 75);
                        const p95 = computePercentile(fwValues, 95);
                        return { age, tw_p05: p05, tw_p25: p25, tw_p50: p50, tw_p75: p75, tw_p95: p95 };
                      })}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                        <Tooltip formatter={dollarMTooltipFormatter} />
                        <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="tw_p05" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                        <Line type="monotone" dataKey="tw_p25" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="tw_p50" stroke={COLORS.hc} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="tw_p75" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                        <Line type="monotone" dataKey="tw_p95" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                      Total wealth stays positive - optimal strategy prevents bankruptcy.
                    </div>
                  </ChartCard>

                  <ChartCard title="Rule of Thumb - Total Wealth ($M)">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={strategyResults.ruleOfThumb[0].ages.map((age, i) => {
                        const fwValues = strategyResults.ruleOfThumb.map(r => r.financialWealth[i]);
                        const p05 = computePercentile(fwValues, 5);
                        const p25 = computePercentile(fwValues, 25);
                        const p50 = computePercentile(fwValues, 50);
                        const p75 = computePercentile(fwValues, 75);
                        const p95 = computePercentile(fwValues, 95);
                        return { age, tw_p05: p05, tw_p25: p25, tw_p50: p50, tw_p75: p75, tw_p95: p95 };
                      })}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                        <Tooltip formatter={dollarMTooltipFormatter} />
                        <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="tw_p05" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                        <Line type="monotone" dataKey="tw_p25" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="tw_p50" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="tw_p75" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                        <Line type="monotone" dataKey="tw_p95" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#e74c3c', textAlign: 'center', marginTop: '8px' }}>
                      Note: 5th percentile can go negative - total wealth bankruptcy risk.
                    </div>
                  </ChartCard>
                </ChartSection>

                {/* Consumption Paths */}
                <ChartSection title="Consumption Over Time">
                  <ChartCard title="Optimal Strategy - Consumption Distribution ($k)">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={strategyResults.optimal[0].ages.map((age, i) => {
                        const consumptionValues = strategyResults.optimal.map(r => r.totalConsumption[i]);
                        const p05 = computePercentile(consumptionValues, 5);
                        const p25 = computePercentile(consumptionValues, 25);
                        const p50 = computePercentile(consumptionValues, 50);
                        const p75 = computePercentile(consumptionValues, 75);
                        const p95 = computePercentile(consumptionValues, 95);
                        return {
                          age,
                          subsistence: strategyResults.optimal[0].subsistenceConsumption[i],
                          c_p05: p05,
                          c_p25: p25,
                          c_p50: p50,
                          c_p75: p75,
                          c_p95: p95,
                        };
                      })}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={formatDollarK} domain={['auto', 'auto']} />
                        <Tooltip formatter={dollarKTooltipFormatter} />
                        <Line type="monotone" dataKey="subsistence" stroke="#999" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Subsistence Floor" />
                        <Line type="monotone" dataKey="c_p05" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                        <Line type="monotone" dataKey="c_p25" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="c_p50" stroke={COLORS.variable} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="c_p75" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                        <Line type="monotone" dataKey="c_p95" stroke={COLORS.variable} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                      Consumption adjusts with wealth. Shows percentile distribution across 50 runs.
                    </div>
                  </ChartCard>

                  <ChartCard title="Rule of Thumb - Consumption Distribution ($k)">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={strategyResults.ruleOfThumb[0].ages.map((age, i) => {
                        const consumptionValues = strategyResults.ruleOfThumb.map(r => r.totalConsumption[i]);
                        const p05 = computePercentile(consumptionValues, 5);
                        const p25 = computePercentile(consumptionValues, 25);
                        const p50 = computePercentile(consumptionValues, 50);
                        const p75 = computePercentile(consumptionValues, 75);
                        const p95 = computePercentile(consumptionValues, 95);
                        return {
                          age,
                          subsistence: strategyResults.ruleOfThumb[0].subsistenceConsumption[i],
                          c_p05: p05,
                          c_p25: p25,
                          c_p50: p50,
                          c_p75: p75,
                          c_p95: p95,
                        };
                      })}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={formatDollarK} domain={['auto', 'auto']} />
                        <Tooltip formatter={dollarKTooltipFormatter} />
                        <Line type="monotone" dataKey="subsistence" stroke="#999" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Subsistence Floor" />
                        <Line type="monotone" dataKey="c_p05" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                        <Line type="monotone" dataKey="c_p25" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="c_p50" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="c_p75" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                        <Line type="monotone" dataKey="c_p95" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                      </LineChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                      Fixed 4% withdrawal. Shows percentile distribution across 50 runs.
                    </div>
                  </ChartCard>
                </ChartSection>

                {/* Portfolio Allocation Comparison */}
                <ChartSection title="Portfolio Allocation (Single Run Example)">
                  <ChartCard title="Optimal Strategy - Allocation (%)">
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={strategyResults.optimal[0].ages.map((age, i) => ({
                        age,
                        cash: strategyResults.optimal[0].cashWeight[i] * 100,
                        bonds: strategyResults.optimal[0].bondWeight[i] * 100,
                        stocks: strategyResults.optimal[0].stockWeight[i] * 100,
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} domain={[0, 100]} tickFormatter={formatPercent} />
                        <Tooltip formatter={percentTooltipFormatter} />
                        <Legend wrapperStyle={{ fontSize: '10px' }} />
                        <Area type="monotone" dataKey="cash" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Cash" />
                        <Area type="monotone" dataKey="bonds" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Bonds" />
                        <Area type="monotone" dataKey="stocks" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="Stocks" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                      Accounts for human capital. Young = more bonds (HC is stock-like), Old = more stocks.
                    </div>
                  </ChartCard>

                  <ChartCard title="Rule of Thumb - Allocation (%)">
                    <ResponsiveContainer width="100%" height={280}>
                      <AreaChart data={strategyResults.ruleOfThumb[0].ages.map((age, i) => ({
                        age,
                        cash: strategyResults.ruleOfThumb[0].cashWeight[i] * 100,
                        bonds: strategyResults.ruleOfThumb[0].bondWeight[i] * 100,
                        stocks: strategyResults.ruleOfThumb[0].stockWeight[i] * 100,
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} domain={[0, 100]} tickFormatter={formatPercent} />
                        <Tooltip formatter={percentTooltipFormatter} />
                        <Legend wrapperStyle={{ fontSize: '10px' }} />
                        <Area type="monotone" dataKey="cash" stackId="1" stroke={COLORS.cash} fill={COLORS.cash} name="Cash" />
                        <Area type="monotone" dataKey="bonds" stackId="1" stroke={COLORS.bond} fill={COLORS.bond} name="Bonds" />
                        <Area type="monotone" dataKey="stocks" stackId="1" stroke={COLORS.stock} fill={COLORS.stock} name="Stocks" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                      (100 - age)% stocks, rest split 50/50 cash & bonds. Frozen at retirement.
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
                    Key Takeaways: Optimal vs Rule of Thumb
                  </div>
                  <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                    <li><strong>Adaptive consumption eliminates default risk</strong> — by adjusting spending to wealth, you never fail to meet subsistence</li>
                    <li><strong>Human capital matters for allocation</strong> — young workers have bond-like future earnings, so their financial portfolio should hold more stocks</li>
                    <li><strong>Duration matching hedges interest rate risk</strong> — optimal strategy matches asset duration to liability duration</li>
                    <li><strong>Rule of thumb ignores personal circumstances</strong> — it doesn't account for earnings profile, expenses, or risk preferences</li>
                    <li><strong>4% rule fails under bad sequence</strong> — fixed withdrawals from a falling portfolio can lead to ruin</li>
                  </ul>
                </div>
              </>
            )}

            {/* Scenario-specific visualizations */}
            {(scenarioType === 'sequenceRisk' || scenarioType === 'rateShock') && scenarioResults && (
              <>
            {/* Market Conditions - Show what the scenario looks like */}
            <ChartSection title={scenarioType === 'sequenceRisk' ? "Stock Return Paths (Log Scale)" : "Interest Rate Paths"}>
              {scenarioType === 'sequenceRisk' ? (
                <>
                  <ChartCard title="Cumulative Stock Returns - Percentile Bands">
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={scenarioPercentileData || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={(v) => `${(Math.exp(v) * 100).toFixed(0)}%`} domain={['auto', 'auto']} />
                        <Tooltip formatter={(v) => v !== undefined ? [`${(Math.exp(v as number) * 100).toFixed(0)}%`, 'Cumulative Return'] : ['', '']} />
                        <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                        <Area type="monotone" dataKey="sr_p05" stackId="sr" fill="transparent" stroke="transparent" />
                        <Area type="monotone" dataKey="sr_band_5_25" stackId="sr" fill={COLORS.stock} fillOpacity={0.15} stroke="transparent" name="5th-25th" />
                        <Area type="monotone" dataKey="sr_band_25_75" stackId="sr" fill={COLORS.stock} fillOpacity={0.3} stroke="transparent" name="25th-75th" />
                        <Area type="monotone" dataKey="sr_band_75_95" stackId="sr" fill={COLORS.stock} fillOpacity={0.15} stroke="transparent" name="75th-95th" />
                        <Line type="monotone" dataKey="sr_p50" stroke={COLORS.stock} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="sr_p25" stroke={COLORS.stock} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="sr_p75" stroke={COLORS.stock} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#c0392b', textAlign: 'center', marginTop: '8px' }}>
                      Stocks averaging ~-5%/year for first 10 years of retirement. Y-axis: cumulative return (100% = starting value).
                    </div>
                  </ChartCard>
                  <ChartCard title="Cumulative Return Distribution at Retirement+10">
                    <div style={{ padding: '24px' }}>
                      {(() => {
                        const retirementIdx = scenarioRetirementAge - params.startAge;
                        const idx10 = Math.min(retirementIdx + 10, scenarioResults.adaptive[0].ages.length - 1);
                        const returns = scenarioResults.adaptive.map(r => r.cumulativeStockReturn[idx10]);
                        return (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', textAlign: 'center' }}>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>5th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.stock }}>{(computePercentile(returns, 5) * 100).toFixed(0)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>25th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.stock }}>{(computePercentile(returns, 25) * 100).toFixed(0)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>Median</div><div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.stock }}>{(computePercentile(returns, 50) * 100).toFixed(0)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>75th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.stock }}>{(computePercentile(returns, 75) * 100).toFixed(0)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>95th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.stock }}>{(computePercentile(returns, 95) * 100).toFixed(0)}%</div></div>
                          </div>
                        );
                      })()}
                      <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '16px' }}>
                        Cumulative stock return after 10 years of forced bad returns. 100% = no change from start.
                      </div>
                    </div>
                  </ChartCard>
                </>
              ) : (
                <>
                  <ChartCard title="Interest Rate Paths - Percentile Bands">
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={scenarioPercentileData || []}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="age" fontSize={10} />
                        <YAxis fontSize={10} tickFormatter={(v) => `${v.toFixed(1)}%`} domain={['auto', 'auto']} />
                        <Tooltip formatter={(v) => v !== undefined ? [`${(v as number).toFixed(2)}%`, 'Interest Rate'] : ['', '']} />
                        <ReferenceLine x={rateShockAge} stroke="#e74c3c" strokeWidth={2} strokeDasharray="5 5" />
                        <Area type="monotone" dataKey="rate_p05" stackId="rate" fill="transparent" stroke="transparent" />
                        <Area type="monotone" dataKey="rate_band_5_25" stackId="rate" fill={COLORS.bond} fillOpacity={0.15} stroke="transparent" name="5th-25th" />
                        <Area type="monotone" dataKey="rate_band_25_75" stackId="rate" fill={COLORS.bond} fillOpacity={0.3} stroke="transparent" name="25th-75th" />
                        <Area type="monotone" dataKey="rate_band_75_95" stackId="rate" fill={COLORS.bond} fillOpacity={0.15} stroke="transparent" name="75th-95th" />
                        <Line type="monotone" dataKey="rate_p50" stroke={COLORS.bond} strokeWidth={2} dot={false} name="(3) Median" />
                        <Line type="monotone" dataKey="rate_p25" stroke={COLORS.bond} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                        <Line type="monotone" dataKey="rate_p75" stroke={COLORS.bond} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                      </AreaChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '11px', color: '#d68910', textAlign: 'center', marginTop: '8px' }}>
                      Rate shock of {(rateShockMagnitude * 100).toFixed(1)}% at age {rateShockAge}. Red dashed line marks the shock.
                    </div>
                  </ChartCard>
                  <ChartCard title="Interest Rate Distribution After Shock">
                    <div style={{ padding: '24px' }}>
                      {(() => {
                        const shockIdx = Math.min(rateShockAge - params.startAge + 1, scenarioResults.adaptive[0].ages.length - 1);
                        const rates = scenarioResults.adaptive.map(r => r.interestRate[shockIdx] * 100);
                        return (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', textAlign: 'center' }}>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>5th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.bond }}>{computePercentile(rates, 5).toFixed(2)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>25th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.bond }}>{computePercentile(rates, 25).toFixed(2)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>Median</div><div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.bond }}>{computePercentile(rates, 50).toFixed(2)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>75th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.bond }}>{computePercentile(rates, 75).toFixed(2)}%</div></div>
                            <div><div style={{ fontSize: '11px', color: '#999' }}>95th %ile</div><div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.bond }}>{computePercentile(rates, 95).toFixed(2)}%</div></div>
                          </div>
                        );
                      })()}
                      <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '16px' }}>
                        Interest rate distribution immediately after the shock at age {rateShockAge}.
                      </div>
                    </div>
                  </ChartCard>
                </>
              )}
            </ChartSection>

            {/* Default Risk and Consumption Comparison */}
            <ChartSection title="Outcomes: Optimal vs Rule of Thumb">
              <ChartCard title="Default Risk (Failure to Meet Subsistence)">
                <div style={{ padding: '24px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
                    <div style={{ textAlign: 'center', padding: '20px', background: '#e8f5e9', borderRadius: '8px' }}>
                      <div style={{ fontSize: '13px', color: '#2e7d32', marginBottom: '8px' }}>Optimal Strategy</div>
                      <div style={{ fontSize: '48px', fontWeight: 'bold', color: '#1b5e20' }}>
                        {scenarioResults.adaptive.filter(r => r.defaulted).length}
                      </div>
                      <div style={{ fontSize: '12px', color: '#388e3c' }}>of 50 runs defaulted</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '20px', background: '#ffebee', borderRadius: '8px' }}>
                      <div style={{ fontSize: '13px', color: '#c62828', marginBottom: '8px' }}>Rule of Thumb (4% Rule)</div>
                      <div style={{ fontSize: '48px', fontWeight: 'bold', color: '#b71c1c' }}>
                        {scenarioResults.fourPercent.filter(r => r.defaulted).length}
                      </div>
                      <div style={{ fontSize: '12px', color: '#d32f2f' }}>of 50 runs defaulted</div>
                      {scenarioResults.fourPercent.filter(r => r.defaulted).length > 0 && (
                        <div style={{ fontSize: '11px', color: '#666', marginTop: '8px' }}>
                          Avg default age: {Math.round(
                            scenarioResults.fourPercent
                              .filter(r => r.defaulted && r.defaultAge !== null)
                              .reduce((sum, r) => sum + (r.defaultAge || 0), 0) /
                            Math.max(1, scenarioResults.fourPercent.filter(r => r.defaulted).length)
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </ChartCard>

              <ChartCard title="Average Annual Consumption ($k)">
                <div style={{ padding: '24px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '12px', color: COLORS.hc }}>Optimal Strategy</div>
                      {(() => {
                        const avgConsumption = scenarioResults.adaptive.map(r => r.totalConsumption.reduce((a, b) => a + b, 0) / r.totalConsumption.length);
                        return (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontSize: '12px' }}>
                            <div><div style={{ color: '#999' }}>5th %ile</div><div style={{ fontWeight: 'bold' }}>{formatDollarK(computePercentile(avgConsumption, 5))}</div></div>
                            <div><div style={{ color: '#999' }}>Median</div><div style={{ fontWeight: 'bold', fontSize: '14px' }}>{formatDollarK(computePercentile(avgConsumption, 50))}</div></div>
                            <div><div style={{ color: '#999' }}>95th %ile</div><div style={{ fontWeight: 'bold' }}>{formatDollarK(computePercentile(avgConsumption, 95))}</div></div>
                          </div>
                        );
                      })()}
                    </div>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '12px', color: COLORS.expenses }}>Rule of Thumb (4% Rule)</div>
                      {(() => {
                        const avgConsumption = scenarioResults.fourPercent.map(r => r.totalConsumption.reduce((a, b) => a + b, 0) / r.totalConsumption.length);
                        return (
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', fontSize: '12px' }}>
                            <div><div style={{ color: '#999' }}>5th %ile</div><div style={{ fontWeight: 'bold' }}>{formatDollarK(computePercentile(avgConsumption, 5))}</div></div>
                            <div><div style={{ color: '#999' }}>Median</div><div style={{ fontWeight: 'bold', fontSize: '14px' }}>{formatDollarK(computePercentile(avgConsumption, 50))}</div></div>
                            <div><div style={{ color: '#999' }}>95th %ile</div><div style={{ fontWeight: 'bold' }}>{formatDollarK(computePercentile(avgConsumption, 95))}</div></div>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                </div>
              </ChartCard>
            </ChartSection>

            {/* Financial Wealth Over Time */}
            <ChartSection title="Financial Wealth Over Time">
              <ChartCard title="Adaptive Strategy - Financial Wealth ($M)">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scenarioPercentileData || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarMTooltipFormatter} />
                    <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="fw_adapt_p05" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="fw_adapt_p25" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="fw_adapt_p50" stroke={COLORS.hc} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="fw_adapt_p75" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="fw_adapt_p95" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                  Adaptive strategy adjusts consumption to preserve wealth.
                </div>
              </ChartCard>
              <ChartCard title="4% Rule - Financial Wealth ($M)">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scenarioPercentileData || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarMTooltipFormatter} />
                    <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="fw_4pct_p05" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="fw_4pct_p25" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="fw_4pct_p50" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="fw_4pct_p75" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="fw_4pct_p95" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#e74c3c', textAlign: 'center', marginTop: '8px' }}>
                  Fixed 4% withdrawal can deplete wealth faster in bad markets.
                </div>
              </ChartCard>
            </ChartSection>

            {/* Total Wealth Over Time */}
            <ChartSection title="Total Wealth Over Time (FW + Human Capital)">
              <ChartCard title="Adaptive Strategy - Total Wealth ($M)">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scenarioPercentileData || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarMTooltipFormatter} />
                    <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="tw_adapt_p05" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="tw_adapt_p25" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="tw_adapt_p50" stroke={COLORS.hc} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="tw_adapt_p75" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="tw_adapt_p95" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                  Total wealth = Financial wealth + Human capital. HC declines to zero at retirement.
                </div>
              </ChartCard>
              <ChartCard title="4% Rule - Total Wealth ($M)">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scenarioPercentileData || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarM} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarMTooltipFormatter} />
                    <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="tw_4pct_p05" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="tw_4pct_p25" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="tw_4pct_p50" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="tw_4pct_p75" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="tw_4pct_p95" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#e74c3c', textAlign: 'center', marginTop: '8px' }}>
                  Total wealth can approach zero, indicating bankruptcy risk.
                </div>
              </ChartCard>
            </ChartSection>

            {/* Consumption Over Time */}
            <ChartSection title="Consumption Over Time">
              <ChartCard title="Adaptive Strategy - Consumption ($k)">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scenarioPercentileData || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarK} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarKTooltipFormatter} />
                    <Line type="monotone" dataKey="subsistence" stroke="#999" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Subsistence Floor" />
                    <Line type="monotone" dataKey="cons_adapt_p05" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="cons_adapt_p25" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="cons_adapt_p50" stroke={COLORS.hc} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="cons_adapt_p75" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="cons_adapt_p95" stroke={COLORS.hc} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#666', textAlign: 'center', marginTop: '8px' }}>
                  Adaptive consumption stays above subsistence floor. Gray dashed line = subsistence level.
                </div>
              </ChartCard>
              <ChartCard title="4% Rule - Consumption ($k)">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={scenarioPercentileData || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" fontSize={10} />
                    <YAxis fontSize={10} tickFormatter={formatDollarK} domain={['auto', 'auto']} />
                    <Tooltip formatter={dollarKTooltipFormatter} />
                    <Line type="monotone" dataKey="subsistence" stroke="#999" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Subsistence Floor" />
                    <Line type="monotone" dataKey="cons_4pct_p05" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(1) 5th %ile" />
                    <Line type="monotone" dataKey="cons_4pct_p25" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(2) 25th %ile" />
                    <Line type="monotone" dataKey="cons_4pct_p50" stroke={COLORS.expenses} strokeWidth={2} dot={false} name="(3) Median" />
                    <Line type="monotone" dataKey="cons_4pct_p75" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="4 4" dot={false} name="(4) 75th %ile" />
                    <Line type="monotone" dataKey="cons_4pct_p95" stroke={COLORS.expenses} strokeWidth={1} strokeDasharray="2 2" dot={false} name="(5) 95th %ile" />
                  </LineChart>
                </ResponsiveContainer>
                <div style={{ fontSize: '11px', color: '#e74c3c', textAlign: 'center', marginTop: '8px' }}>
                  4% Rule can force consumption below subsistence when wealth depletes.
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
              {scenarioType === 'sequenceRisk' ? (
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                  <li><strong>4% Rule ignores market conditions</strong> — withdrawing fixed amounts from a falling portfolio accelerates depletion</li>
                  <li><strong>Sequence matters</strong> — even with the same average returns, bad early years can be catastrophic</li>
                  <li><strong>Adaptive consumption protects subsistence</strong> — by reducing variable spending when wealth drops, the floor is always met</li>
                  <li><strong>Trade-off:</strong> Adaptive may mean lower consumption in good times, but eliminates default risk</li>
                </ul>
              ) : (
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', lineHeight: 1.6 }}>
                  <li><strong>Rate drops increase liability PV</strong> — future expenses cost more in present value terms</li>
                  <li><strong>Duration matching hedges this risk</strong> — long bonds appreciate when rates fall, offsetting the liability increase</li>
                  <li><strong>Cash/short bonds leave you exposed</strong> — they don't appreciate enough to cover the increased cost of future spending</li>
                  <li><strong>The portfolio already accounts for human capital</strong> — young workers have bond-like future earnings that offset some rate risk</li>
                </ul>
              )}
            </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
