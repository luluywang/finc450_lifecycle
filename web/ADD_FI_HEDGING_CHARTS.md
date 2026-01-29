# Prompt: Add Net FI PV and DV01 Charts to Lifecycle Visualizer

Copy this prompt to Claude when editing your deployed artifact to add fixed-income hedging metric charts to the Teaching Scenarios page.

---

## Prompt

Please add two new charts (Panel 9: Net Fixed Income PV and Panel 10: DV01) to the Teaching Scenarios page. These charts show fixed-income hedging metrics comparing LDI vs Rule-of-Thumb strategies.

### 1. Add fields to `PercentileStats` interface

Find the `PercentileStats` interface and add:

```typescript
interface PercentileStats {
  // ... existing fields ...
  netFiPv: FieldPercentiles;
  dv01: FieldPercentiles;
}
```

### 2. Add helper function to compute Net FI PV and DV01 paths

Add this function after `computeFieldPercentiles()`:

```typescript
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

  // Get base earnings and expenses
  const legacyParams: Params = {
    startAge: params.startAge, retirementAge: params.retirementAge, endAge: params.endAge,
    initialEarnings: params.initialEarnings, earningsGrowth: params.earningsGrowth,
    earningsHumpAge: params.earningsHumpAge, earningsDecline: params.earningsDecline,
    baseExpenses: params.baseExpenses, expenseGrowth: params.expenseGrowth,
    retirementExpenses: params.retirementExpenses, stockBetaHC: params.stockBetaHumanCapital,
    gamma: params.gamma, initialWealth: params.initialWealth, rBar: econParams.rBar,
    muStock: econParams.muExcess, bondSharpe: econParams.bondSharpe, sigmaS: econParams.sigmaS,
    sigmaR: econParams.sigmaR, rho: econParams.rho, bondDuration: econParams.bondDuration,
    phi: econParams.phi,
  };

  const earningsProfile = computeEarningsProfile(legacyParams);
  const expenseProfile = computeExpenseProfile(legacyParams);

  const baseEarnings: number[] = Array(nPeriods).fill(0);
  const expenses: number[] = Array(nPeriods).fill(0);
  for (let t = 0; t < workingYears; t++) {
    baseEarnings[t] = earningsProfile[t];
    expenses[t] = expenseProfile.working[t];
  }
  for (let t = workingYears; t < nPeriods; t++) {
    expenses[t] = expenseProfile.retirement[t - workingYears];
  }

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

      const remainingExpenses = expenses.slice(t);
      const pvExp = computePresentValue(remainingExpenses, currentRate, phi, rBar);
      const durationExp = computeDuration(remainingExpenses, currentRate, phi, rBar);

      let durationHc = 0;
      if (isWorking) {
        const remainingEarnings = baseEarnings.slice(t, workingYears);
        durationHc = computeDuration(remainingEarnings, currentRate, phi, rBar);
      }

      let hcBond = 0;
      if (isWorking && hc > 0) {
        const nonStockHc = hc * (1 - stockBeta);
        if (bondDuration > 0) hcBond = nonStockHc * (durationHc / bondDuration);
      }

      let expBond = 0;
      if (bondDuration > 0 && pvExp > 0) expBond = pvExp * (durationExp / bondDuration);

      const bondHoldings = wB * fw;
      netFiPv.push(bondHoldings + hcBond - expBond);

      const assetDur = durationHc * hcBond + bondDuration * bondHoldings;
      const liabDur = durationExp * expBond;
      dv01.push((assetDur - liabDur) * 0.01);
    }
    netFiPvPaths.push(netFiPv);
    dv01Paths.push(dv01);
  }
  return { netFiPv: netFiPvPaths, dv01: dv01Paths };
}
```

### 3. Update percentile computation in `runTeachingScenarios`

In the `runBothStrategies` helper inside `runTeachingScenarios`, after computing cumulative stock returns, add:

```typescript
// Compute Net FI PV and DV01 paths for both strategies
const { netFiPv: ldiNetFiPvPaths, dv01: ldiDv01Paths } = computeNetFiPvAndDv01Paths(ldiResult, params, econParams);
const { netFiPv: rotNetFiPvPaths, dv01: rotDv01Paths } = computeNetFiPvAndDv01Paths(rotResult, params, econParams);
```

Then add to both `ldiPercentiles` and `rotPercentiles`:

```typescript
netFiPv: computeFieldPercentiles(ldiNetFiPvPaths),
dv01: computeFieldPercentiles(ldiDv01Paths),
```

### 4. Add chart data preparation

In the individual scenario view (where `wealthAllocationData` is defined), add:

```typescript
// Panel 9 data: Net FI PV
const netFiPvData = ages.map((age, i) => ({
  age,
  ldi_p50: scenario.ldi.percentiles.netFiPv.p50[i],
  rot_p50: scenario.rot.percentiles.netFiPv.p50[i],
  ldi_p5: scenario.ldi.percentiles.netFiPv.p5[i],
  ldi_p95: scenario.ldi.percentiles.netFiPv.p95[i],
  rot_p5: scenario.rot.percentiles.netFiPv.p5[i],
  rot_p95: scenario.rot.percentiles.netFiPv.p95[i],
}));

// Panel 10 data: DV01
const dv01Data = ages.map((age, i) => ({
  age,
  ldi_p50: scenario.ldi.percentiles.dv01.p50[i],
  rot_p50: scenario.rot.percentiles.dv01.p50[i],
  ldi_p5: scenario.ldi.percentiles.dv01.p5[i],
  ldi_p95: scenario.ldi.percentiles.dv01.p95[i],
  rot_p5: scenario.rot.percentiles.dv01.p5[i],
  rot_p95: scenario.rot.percentiles.dv01.p95[i],
}));
```

### 5. Add the two chart panels after Panel 8

After the Panel 8 (PV Consumption Distribution) ChartCard, add:

```tsx
{/* Panel 9: Net Fixed Income PV */}
<ChartCard title="Panel 9: Net Fixed Income PV ($k)">
  <ResponsiveContainer width="100%" height={280}>
    <LineChart data={netFiPvData}>
      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
      <XAxis dataKey="age" fontSize={11} />
      <YAxis fontSize={11} tickFormatter={(v) => `$${Math.round(v)}k`} />
      <Tooltip formatter={(v: number, name: string) => [`$${Math.round(v)}k`, name]} />
      <ReferenceLine x={scenarioRetirementAge} stroke="#999" strokeDasharray="3 3" />
      <ReferenceLine y={0} stroke="#000" strokeWidth={1.5} opacity={0.7} />
      <Line type="monotone" dataKey="ldi_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
      <Line type="monotone" dataKey="ldi_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
      <Line type="monotone" dataKey="ldi_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
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
      <Tooltip formatter={(v: number, name: string) => [`$${Math.round(v)}`, name]} />
      <ReferenceLine x={scenarioRetirementAge} stroke="#999" strokeDasharray="3 3" />
      <ReferenceLine y={0} stroke="#000" strokeWidth={1.5} opacity={0.7} />
      <Line type="monotone" dataKey="ldi_p5" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 5th" />
      <Line type="monotone" dataKey="ldi_p50" stroke={COLOR_LDI} strokeWidth={2} dot={false} name="LDI Median" />
      <Line type="monotone" dataKey="ldi_p95" stroke={COLOR_LDI} strokeWidth={1} strokeDasharray="2 2" dot={false} name="LDI 95th" />
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
```

### Expected Result

After these changes, the Teaching Scenarios page will show:
- **Panel 9: Net Fixed Income PV** - Shows bond position minus liabilities. Zero line = perfectly hedged.
- **Panel 10: DV01** - Shows interest rate sensitivity. Zero = duration matched.

Both charts compare LDI (blue) vs RoT (gold) with 5th/50th/95th percentile bands.
