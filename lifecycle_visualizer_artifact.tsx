// Lifecycle Path Visualizer - Claude Artifact
// Interactive visualization for lifecycle investment strategy
// Copy this entire file into a Claude artifact (React type)

import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// =============================================================================
// Types
// =============================================================================

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
  bondDurationBenchmark: number;
  gamma: number;
  initialWealth: number;

  // Market parameters (VCV Merton)
  rBar: number;
  muStock: number;
  muBond: number;
  sigmaS: number;
  sigmaR: number;
  rho: number;
  bondDuration: number;
  phi: number;
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
}

// =============================================================================
// Core Calculation Functions
// =============================================================================

function effectiveDuration(tau: number, phi: number): number {
  if (Math.abs(phi - 1.0) < 1e-10) return tau;
  return (1 - Math.pow(phi, tau)) / (1 - phi);
}

function zeroCouponPrice(r: number, tau: number, rBar: number, phi: number): number {
  if (tau <= 0) return 1.0;
  const B = effectiveDuration(tau, phi);
  return Math.exp(-tau * rBar - B * (r - rBar));
}

function computePresentValue(
  cashflows: number[],
  rate: number,
  phi: number,
  rBar: number
): number {
  let pv = 0;
  for (let t = 0; t < cashflows.length; t++) {
    pv += cashflows[t] * zeroCouponPrice(rate, t + 1, rBar, phi);
  }
  return pv;
}

function computeDuration(
  cashflows: number[],
  rate: number,
  phi: number,
  rBar: number
): number {
  if (cashflows.length === 0) return 0;

  const pv = computePresentValue(cashflows, rate, phi, rBar);
  if (pv < 1e-10) return 0;

  let weightedSum = 0;
  for (let t = 0; t < cashflows.length; t++) {
    const P_t = zeroCouponPrice(rate, t + 1, rBar, phi);
    const B_t = effectiveDuration(t + 1, phi);
    weightedSum += cashflows[t] * P_t * B_t;
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
  const [targetStock, targetBond, targetCash] = computeFullMertonAllocationConstrained(
    params.muStock, params.muBond, params.sigmaS, params.sigmaR,
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

  for (let i = 0; i < totalYears; i++) {
    if (params.bondDurationBenchmark > 0 && nonStockHC[i] > 0) {
      const bondFraction = Math.min(1, durationEarnings[i] / params.bondDurationBenchmark);
      hcBond[i] = nonStockHC[i] * bondFraction;
      hcCash[i] = nonStockHC[i] * (1 - bondFraction);
    } else {
      hcCash[i] = nonStockHC[i];
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

    // Cap at earnings during working years
    if (earnings[i] > 0 && totalConsumption[i] > earnings[i]) {
      totalConsumption[i] = earnings[i];
      variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
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

  // Apply no-short constraints to get portfolio weights
  const stockWeight = Array(totalYears).fill(0);
  const bondWeight = Array(totalYears).fill(0);
  const cashWeight = Array(totalYears).fill(0);

  for (let i = 0; i < totalYears; i++) {
    const fw = financialWealth[i];
    if (fw > 1e-6) {
      let wStock = targetFinStocks[i] / fw;
      let wBond = targetFinBonds[i] / fw;
      let wCash = targetFinCash[i] / fw;

      // Aggregate to equity vs fixed income
      let equity = Math.max(0, wStock);
      let fixedIncome = Math.max(0, wBond + wCash);

      // Normalize
      const totalAgg = equity + fixedIncome;
      if (totalAgg > 0) {
        equity /= totalAgg;
        fixedIncome /= totalAgg;
      } else {
        equity = targetStock;
        fixedIncome = targetBond + targetCash;
      }

      // Split fixed income
      if (wBond > 0 && wCash > 0) {
        const fiTotal = wBond + wCash;
        bondWeight[i] = fixedIncome * (wBond / fiTotal);
        cashWeight[i] = fixedIncome * (wCash / fiTotal);
      } else if (wBond > 0) {
        bondWeight[i] = fixedIncome;
        cashWeight[i] = 0;
      } else if (wCash > 0) {
        bondWeight[i] = 0;
        cashWeight[i] = fixedIncome;
      } else {
        const targetFI = targetBond + targetCash;
        if (targetFI > 0) {
          bondWeight[i] = fixedIncome * (targetBond / targetFI);
          cashWeight[i] = fixedIncome * (targetCash / targetFI);
        } else {
          bondWeight[i] = fixedIncome / 2;
          cashWeight[i] = fixedIncome / 2;
        }
      }
      stockWeight[i] = equity;
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
const formatDollar = (value: number) => Math.round(value).toLocaleString();
const formatPercent = (value: number) => Math.round(value);
const formatYears = (value: number) => value.toFixed(1);

const dollarTooltipFormatter = (value: number) => `$${formatDollar(value)}k`;
const percentTooltipFormatter = (value: number) => `${formatPercent(value)}%`;
const yearsTooltipFormatter = (value: number) => `${formatYears(value)} yrs`;

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
    bondDurationBenchmark: 20,
    gamma: 2,
    initialWealth: 1,
    rBar: 0.02,
    muStock: 0.04,
    muBond: 0.005,
    sigmaS: 0.18,
    sigmaR: 0.012,
    rho: -0.2,
    bondDuration: 7,
    phi: 1.0,
  });

  const updateParam = (key: keyof Params, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  // Compute lifecycle results
  const result = useMemo(() => computeLifecycleMedianPath(params), [params]);

  // Prepare chart data
  const chartData = useMemo(() => {
    return result.ages.map((age, i) => ({
      age,
      earnings: result.earnings[i],
      expenses: result.expenses[i],
      pvEarnings: result.pvEarnings[i],
      pvExpenses: result.pvExpenses[i],
      durationEarnings: result.durationEarnings[i],
      durationExpenses: result.durationExpenses[i],
      humanCapital: result.humanCapital[i],
      financialWealth: result.financialWealth[i],
      hcStock: result.hcStock[i],
      hcBond: result.hcBond[i],
      hcCash: result.hcCash[i],
      stockWeight: result.stockWeight[i] * 100,
      bondWeight: result.bondWeight[i] * 100,
      cashWeight: result.cashWeight[i] * 100,
      subsistence: result.subsistenceConsumption[i],
      variable: result.variableConsumption[i],
      totalConsumption: result.totalConsumption[i],
    }));
  }, [result]);

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
            label="Bond excess return"
            value={params.muBond * 100}
            onChange={(v) => updateParam('muBond', v / 100)}
            min={0} max={2} step={0.1} suffix="%" decimals={2}
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
          <StepperInput
            label="HC duration benchmark"
            value={params.bondDurationBenchmark}
            onChange={(v) => updateParam('bondDurationBenchmark', v)}
            min={5} max={30} step={1} suffix="yrs" decimals={0}
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
        <h1 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '16px', color: '#2c3e50' }}>
          Lifecycle Path Visualizer
        </h1>

        {/* Section 1: Assumptions */}
        <ChartSection title="Section 1: Assumptions">
          <ChartCard title="Earnings Profile ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
                <Line type="monotone" dataKey="earnings" stroke={COLORS.earnings} strokeWidth={2} dot={false} name="Earnings" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Expense Profile ($k)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age" fontSize={10} />
                <YAxis fontSize={10} tickFormatter={formatDollar} />
                <Tooltip formatter={dollarTooltipFormatter} />
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
      </div>
    </div>
  );
}
