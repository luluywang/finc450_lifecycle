# Duration Type Analysis: Modified vs Macaulay

## Your Question
You asked whether `bond_duration` represents modified duration, and whether the forward-looking duration computations (`duration_hc`, `duration_exp`) are computing modified or Macaulay duration.

## Answer: All Duration Computations Use MODIFIED DURATION

### 1. How Effective Duration Works

The code uses `effective_duration(tau, phi)` which computes:

```
B(tau) = (1 - phi^tau) / (1 - phi)
```

This is **not** Macaulay duration—it's **rate duration** in the Vasicek model context.

With `phi = 1.0` (no mean reversion):
```
B(tau) = tau
```

### 2. Critical: This IS Modified Duration

The Vasicek bond price formula is:
```
P(tau) = exp(-tau*r_bar - B(tau)*(r - r_bar))
```

Taking the derivative with respect to r:
```
dP/P = -B(tau) * dr
```

This `B(tau)` is **price elasticity**—the percentage change in price per unit change in rate. This is **modified duration by definition**.

### 3. The Consistency Chain

The code uses modified duration consistently:

| Function | Returns | Type |
|----------|---------|------|
| `effective_duration(t, phi)` | `B(t) = (1-phi^t)/(1-phi)` | **Modified duration** (price elasticity) |
| `compute_duration()` | PV-weighted average of `B(t)` | **Modified duration** |
| `bond_duration` parameter | 20.0 | **Reference modified duration** |

### 4. The Bond Return Formula

```python
bond_return = r - duration * Δr
```

This uses `duration` directly as the price sensitivity:
- `duration = bond_duration = 20.0` (a modified duration)
- When `Δr` increases by 1%, bond return decreases by 20%

This is the **modified duration approximation**.

### 5. The HC Decomposition

```python
hc_bond_frac = duration_hc / bond_duration
```

Both are in the same units (modified duration):
- `duration_hc = 17.86` (modified duration of earnings)
- `bond_duration = 20.0` (reference modified duration)
- Ratio = 0.893 → HC is 89.3% bond-like

This is **correct** because both durations are comparable.

## Example Verification

With the earnings profile:
- **Computed duration = 17.86 years** (modified duration)
- This is properly comparable to **bond_duration = 20.0**
- The ratio (0.893) makes sense: earnings are shorter duration than long bonds

## Conclusion

**The "modified vs Macaulay duration" issue is a RED HERRING.**

The code consistently uses **effective duration**, which in the Vasicek model is equivalent to **modified duration** (price elasticity). There is no mismatch.

### So What Causes the Retirement Kink?

The kink is NOT due to duration type mismatches. Instead, it's due to:

1. **HC discontinuity**: Human capital goes from 196 → 0 at retirement (economically correct but mathematically discontinuous)
2. **Loss of HC offset**: The HC_bond component (which reduces target bond position) disappears
3. **Expense stream break**: The shift from working to retirement expenses also changes expense duration

The computed durations themselves are correct; the kink is caused by the structural discontinuity in the economic model at retirement, not a bug in the duration calculation.
