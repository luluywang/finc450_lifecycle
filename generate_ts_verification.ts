/**
 * Standalone TypeScript verification data generator.
 *
 * This file extracts the pure calculation functions from LifecycleVisualizer.tsx
 * and runs them in Node.js to generate verification data without needing the browser.
 *
 * Usage:
 *   npx ts-node generate_ts_verification.ts > output/typescript_verification.json
 *   OR
 *   npx tsx generate_ts_verification.ts > output/typescript_verification.json
 */

// =============================================================================
// Core Calculation Functions (copied from LifecycleVisualizer.tsx)
// =============================================================================

function effectiveDuration(tau: number, phi: number): number {
  if (tau <= 0) return 0.0;
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
  phi: number | null = null,
  rBar: number | null = null
): number {
  if (cashflows.length === 0) return 0;

  let pv = 0;
  if (rBar !== null && phi !== null) {
    for (let t = 0; t < cashflows.length; t++) {
      pv += cashflows[t] * zeroCouponPrice(rate, t + 1, rBar, phi);
    }
  } else {
    for (let t = 0; t < cashflows.length; t++) {
      pv += cashflows[t] / Math.pow(1 + rate, t + 1);
    }
  }
  return pv;
}

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
    for (let t = 0; t < cashflows.length; t++) {
      const P_t = zeroCouponPrice(rate, t + 1, rBar, phi);
      const B_t = effectiveDuration(t + 1, phi);
      weightedSum += cashflows[t] * P_t * B_t;
    }
  } else {
    for (let t = 0; t < cashflows.length; t++) {
      weightedSum += (t + 1) * cashflows[t] / Math.pow(1 + rate, t + 1);
    }
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

  const sigmaB = duration * sigmaR;
  const covSB = -duration * sigmaS * sigmaR * rho;
  const varS = sigmaS * sigmaS;
  const varB = sigmaB * sigmaB;
  const det = varS * varB - covSB * covSB;

  let stockWeight: number;
  let bondWeight: number;

  if (Math.abs(det) < 1e-12) {
    stockWeight = muStock / (gamma * varS);
    bondWeight = varB > 1e-12 ? muBond / (gamma * varB) : 0;
  } else {
    const inv00 = varB / det;
    const inv01 = -covSB / det;
    const inv10 = -covSB / det;
    const inv11 = varS / det;

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

  wStock = Math.max(0, wStock);
  wBond = Math.max(0, wBond);
  wCash = Math.max(0, wCash);

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

// =============================================================================
// Default Parameters (matching Python)
// =============================================================================

interface Params {
  startAge: number;
  retirementAge: number;
  endAge: number;
  initialEarnings: number;
  earningsGrowth: number;
  earningsHumpAge: number;
  earningsDecline: number;
  baseExpenses: number;
  expenseGrowth: number;
  retirementExpenses: number;
  stockBetaHC: number;
  gamma: number;
  initialWealth: number;
  rBar: number;
  muStock: number;
  bondSharpe: number;
  sigmaS: number;
  sigmaR: number;
  rho: number;
  bondDuration: number;
  phi: number;
}

const DEFAULT_PARAMS: Params = {
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
  gamma: 2.0,
  initialWealth: 100,
  rBar: 0.02,
  muStock: 0.045,
  bondSharpe: 0.0,
  sigmaS: 0.18,
  sigmaR: 0.007,
  rho: 0.0,
  bondDuration: 20.0,
  phi: 1.0,
};

function computeMuBond(params: Params): number {
  return params.bondSharpe * params.bondDuration * params.sigmaR;
}

function computeEarningsProfile(params: Params): number[] {
  const workingYears = params.retirementAge - params.startAge;
  const earnings: number[] = [];

  for (let year = 0; year < workingYears; year++) {
    const age = params.startAge + year;
    let income = params.initialEarnings;

    // Apply growth
    income *= Math.pow(1 + params.earningsGrowth, year);

    // Apply hump (decline after humpAge)
    if (age > params.earningsHumpAge) {
      const yearsAfterHump = age - params.earningsHumpAge;
      income *= Math.pow(1 - params.earningsDecline, yearsAfterHump);
    }

    earnings.push(income);
  }

  return earnings;
}

function computeExpenseProfile(params: Params): { working: number[], retirement: number[] } {
  const workingYears = params.retirementAge - params.startAge;
  const retirementYears = params.endAge - params.retirementAge;

  const working: number[] = [];
  for (let year = 0; year < workingYears; year++) {
    let expense = params.baseExpenses;
    expense *= Math.pow(1 + params.expenseGrowth, year);
    working.push(expense);
  }

  const retirement = Array(retirementYears).fill(params.retirementExpenses);

  return { working, retirement };
}

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
  if (fw <= 1e-6) {
    return [targetStock, targetBond, targetCash];
  }

  let wStock = targetFinStock / fw;
  let wBond = targetFinBond / fw;
  let wCash = targetFinCash / fw;

  if (allowLeverage) {
    return [wStock, wBond, wCash];
  }

  let equity = Math.max(0, wStock);
  let fixedIncome = Math.max(0, wBond + wCash);

  const totalAgg = equity + fixedIncome;
  if (totalAgg > 0) {
    equity /= totalAgg;
    fixedIncome /= totalAgg;
  } else {
    equity = targetStock;
    fixedIncome = targetBond + targetCash;
  }

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
    const targetFI = targetBond + targetCash;
    if (targetFI > 0) {
      wB = fixedIncome * (targetBond / targetFI);
      wC = fixedIncome * (targetCash / targetFI);
    } else {
      wB = fixedIncome / 2;
      wC = fixedIncome / 2;
    }
  }

  return [equity, wB, wC];
}

// =============================================================================
// Lifecycle Median Path Computation
// =============================================================================

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
    let remainingEarnings: number[] = [];
    if (i < workingYears) {
      remainingEarnings = earnings.slice(i, workingYears);
    }
    const remainingExpenses = expenses.slice(i);

    // Use VCV term structure (phi, rBar) to match Python compute_static_pvs
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
    if (params.bondDuration > 0 && nonStockHC[i] > 0) {
      const bondFraction = durationEarnings[i] / params.bondDuration;
      hcBond[i] = nonStockHC[i] * bondFraction;
      hcCash[i] = nonStockHC[i] * (1 - bondFraction);
    } else {
      hcCash[i] = nonStockHC[i];
    }
  }

  // Decompose expenses
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

  // Initialize total wealth and weight arrays (computed inside the loop)
  const totalWealth = Array(totalYears).fill(0);
  const stockWeight = Array(totalYears).fill(0);
  const bondWeight = Array(totalYears).fill(0);
  const cashWeight = Array(totalYears).fill(0);

  // Simulate wealth accumulation with step-by-step portfolio weight calculation
  // This matches Python's simulate_paths which computes weights at each time step
  for (let i = 0; i < totalYears; i++) {
    // Current financial wealth and human capital
    const fw = financialWealth[i];
    const hc = humanCapital[i];
    const tw = fw + hc;
    totalWealth[i] = tw;
    netWorth[i] = hc + fw - pvExpenses[i];

    // Compute portfolio weights FIRST (needed for dynamic consumption rate)
    const targetFinStock = targetStock * tw - hcStock[i];
    const targetFinBond = targetBond * tw - hcBond[i] + expBond[i];
    const targetFinCash = targetCash * tw - hcCash[i] + expCash[i];

    const [wS, wB, wC] = normalizePortfolioWeights(
      targetFinStock, targetFinBond, targetFinCash,
      fw, targetStock, targetBond, targetCash,
      false  // no leverage for median path
    );

    stockWeight[i] = wS;
    bondWeight[i] = wB;
    cashWeight[i] = wC;

    // Dynamic consumption rate using realized weights and current rate (r for median path)
    const expectedReturnI = (
      wS * (r + params.muStock) +
      wB * (r + muBond) +
      wC * r
    );
    const sigmaBi = params.bondDuration * params.sigmaR;
    const covSBi = -params.bondDuration * params.sigmaS * params.sigmaR * params.rho;
    const portfolioVarI = (
      wS * wS * params.sigmaS * params.sigmaS +
      wB * wB * sigmaBi * sigmaBi +
      2 * wS * wB * covSBi
    );
    const consumptionRate = expectedReturnI - 0.5 * portfolioVarI;

    // Compute consumption using dynamic rate
    variableConsumption[i] = Math.max(0, consumptionRate * netWorth[i]);
    totalConsumption[i] = subsistenceConsumption[i] + variableConsumption[i];

    if (earnings[i] > 0 && totalConsumption[i] > earnings[i]) {
      totalConsumption[i] = earnings[i];
      variableConsumption[i] = Math.max(0, earnings[i] - subsistenceConsumption[i]);
    } else if (earnings[i] === 0) {
      if (subsistenceConsumption[i] > fw) {
        totalConsumption[i] = fw;
        subsistenceConsumption[i] = fw;
        variableConsumption[i] = 0;
      } else if (totalConsumption[i] > fw) {
        totalConsumption[i] = fw;
        variableConsumption[i] = fw - subsistenceConsumption[i];
      }
    }

    // Compute portfolio return using CURRENT weights
    const portfolioReturn = wS * stockReturn + wB * bondReturn + wC * cashReturn;

    const savings = earnings[i] - totalConsumption[i];

    if (i < totalYears - 1) {
      financialWealth[i + 1] = Math.max(0, fw * (1 + portfolioReturn) + savings);
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
    targetStock,
    targetBond,
    targetCash,
  };
}

// =============================================================================
// Generate Verification Data
// =============================================================================

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
    total_wealth: result.totalWealth,
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

  return {
    metadata: {
      source: "TypeScript (standalone generator)",
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
  };
}

// =============================================================================
// Main
// =============================================================================

const data = generateVerificationData(DEFAULT_PARAMS);
console.log(JSON.stringify(data, null, 2));
