# Root Cause Analysis: Portfolio Weight Kink at Retirement

## Summary
There is a **-1.61% discontinuous drop in bond weight** at retirement (age 65) for the LDI strategy.

## Root Cause: Loss of Human Capital Bond Offset

### The Problem

The portfolio allocation formula in LDI is:
```
target_fin_bond = target_bond × surplus - HC_bond + Exp_bond
```

At retirement, there is a discontinuous jump in the components:

| Component | Age 64 | Age 65 | Change |
|-----------|--------|--------|--------|
| HC | 196.04 | 0.00 | -196.04 (discontinuous!) |
| HC_bond | 9.80 | 0.00 | -9.80 (offset disappears) |
| Exp_bond | 1647.98 | 1564.60 | -83.38 (duration changes) |
| Surplus | 1862.93 | 1897.88 | +34.96 (continuous) |
| **Target_fin_bond** | **1638.18** | **1564.60** | **-73.58** |
| Bond Weight | 0.4102 | 0.3941 | **-0.0161** |

### Why This Happens

1. **Human Capital becomes 0 at retirement** (economically correct - no more future earnings)
2. **HC duration becomes 0** (because HC = 0, so there's nothing to compute duration for)
3. **HC_bond offset disappears**: The HC was partially bond-like (`HC_bond = HC × (1-β) × duration_HC/bond_duration`), which created a hedge
4. **At the same time, expense duration changes** due to the shift from working→retirement expense profile

### Is This Correct?

**Theoretically, YES** - but there's a potential issue:

The human capital didn't suddenly become short-duration. Instead, **it ceased to exist**. However, the way this is handled might not be smooth:

1. Just before retirement (age 64): HC_bond = 9.80 is reducing the target bond position
2. Just after retirement (age 65): HC_bond = 0, so that offset is gone

This creates an instantaneous demand for MORE bonds (to compensate for the lost HC bond offset).

## Potential Issues

### Issue 1: Modified Duration vs Duration
The code uses "effective duration" from the mean-reverting rate model, but applies it to decompose HC:
```python
hc_bond_frac = duration_hc / bond_duration
```

With `phi=1.0` (no mean reversion), this is equivalent to Macaulay duration. But **modified duration** (which accounts for the price sensitivity to rate changes) is:
```
Modified Duration = Macaulay Duration / (1 + r)
```

If the code is using Macaulay duration where modified duration should be used, this could cause allocation errors.

### Issue 2: Duration of HC Computation
The HC duration is computed as a PV-weighted average. But at the retirement boundary:
- **Age 64**: HC has earnings for 1 more year + then 0 (working years end)
- **Age 65**: HC is 0 (no more earnings)

This sudden drop in HC is fundamentally discontinuous. The issue is: **should the last year of HC before retirement have zero or positive duration?**

Currently, the earnings stream is cut off exactly at retirement, so the last year (age 64) has 1 year of future earnings left, giving it positive duration.

### Issue 3: Normalization Issues?
The normalization function (lines 201-254) clips stocks/bonds at zero and enforces leverage constraints. But with the sudden jump in target_fin_bond, this might be causing:

1. **Numerical clipping** if the constraint is tight
2. **Re-scaling** if leverage constraint is hit
3. **Residual cash becoming negative** (borrowing) if leverage is capped

Let me check if the leverage constraint is being hit... Looking at the numbers:
- Age 64: stock + bond = 1293.70 + 1638.18 = 2931.88, FW = 3954.14, ratio = 0.741 ✓
- Age 65: stock + bond = 1317.98 + 1564.60 = 2882.58, FW = 4131.35, ratio = 0.698 ✓

With `max_leverage = 1.0`, these are both below 1.0, so **no clipping is happening**. The kink is real!

## Solutions

### Solution 1: Smooth HC Duration at Retirement Boundary
Instead of dropping HC to zero abruptly, interpolate the HC duration smoothly over a period before and after retirement.

**Risk**: Makes the model more complex; may not be theoretically justified.

### Solution 2: Check if Modified Duration Should Be Used
Verify whether `effective_duration()` should return **modified** duration instead of Macaulay duration. Look at:
1. How bond returns are computed (line 148-183 of economics.py)
2. Whether the DC_bond price sensitivity matches the duration approximation

### Solution 3: Pre-retirement Duration Adjustment
Keep the HC non-zero just before retirement, allowing a smooth glide into retirement. For example:
- Age 64.5: HC is 50% of expected value
- Age 65: HC is 0%

This would smooth out the duration transition.

### Solution 4: Nothing - It's Correct!
The kink might be theoretically justified:
- You DO lose the HC bond offset at retirement
- This DOES mean you need less bond exposure post-retirement
- The weights should change

## Recommendation

Check the following:
1. **Is `effective_duration()` computing modified or Macaulay duration?** If it's Macaulay, should it be modified for price sensitivity calculations?
2. **Is the expense decomposition correct at the retirement boundary?** The duration_exp changes from 14.41 → 14.01; is this expected?
3. **Test smoothness**: Run the same simulation with a gradual transition to retirement (e.g., phased earnings decline) and see if the bond weight changes are smooth.

If the answer to #1 is "should be modified" and the answer to #2 is "no issue found," then the kink is theoretically correct and shouldn't be changed.
