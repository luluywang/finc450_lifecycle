# Teaching Scenarios: LDI vs Rule-of-Thumb Strategy Comparison

## Overview

This PDF compares two lifecycle investment strategies across three Monte Carlo scenarios designed to illustrate key concepts in lifecycle investing:

- **LDI (Liability-Driven Investment)**: Dynamic strategy that accounts for human capital composition
- **Rule-of-Thumb (RoT)**: Simple (100 - Age)% stock allocation with 4% withdrawal rule

## PDF Structure

### Page 1: Summary Comparison

Aggregate metrics across all scenarios for both β=0 (bond-like HC) and β=0.4 (risky HC):

| Metric | Description |
|--------|-------------|
| Default Rates | Percentage of simulations where portfolio is depleted |
| PV Lifetime Consumption | Present value of total consumption over lifetime |
| Terminal Wealth | Median wealth at age 95 |

### Pages 2-7: Individual Scenario Pages

Each scenario page contains 8 panels:

1. **Cumulative Stock Market Returns** - Fan chart showing return distribution
2. **Interest Rate Paths** - Fan chart of rate evolution
3. **Financial Wealth** - Overlaid LDI vs RoT wealth trajectories
4. **Default Timing** - Histogram of when defaults occur
5. **Stock Allocation** - LDI dynamic vs RoT static weights
6. **Bond Allocation** - Portfolio bond weights over time
7. **Terminal Wealth Distribution** - Histogram of final wealth
8. **PV Consumption Distribution** - Histogram of lifetime consumption value

## Scenarios

### Baseline (Pages 2, 5)
- Standard Monte Carlo with random shocks
- No scenario manipulation
- Tests normal market conditions

### Sequence-of-Returns Risk (Pages 3, 6)
- Bad stock returns (~-12%/yr) in first 5 years OF retirement
- Demonstrates the classic problem of withdrawing from a declining portfolio
- Shows why sequence of returns matters for retirees

### Pre-Retirement Rate Shock (Pages 4, 7)
- Interest rate drop (~4% cumulative) in 5 years BEFORE retirement
- Affects bond valuations and discount rates
- Tests strategy robustness to rate environment changes

## Key Parameters

- **Simulations**: 500 Monte Carlo paths
- **Career**: Age 25-65 (40 years working)
- **Retirement**: Age 65-95 (30 years)
- **Beta values**: 0.0 (bond-like HC), 0.4 (risky HC)
- **RoT withdrawal rate**: 4%
- **Random seed**: 42 (reproducible)

## Key Findings

| Scenario | LDI Default Rate (β=0) | RoT Default Rate (β=0) |
|----------|------------------------|------------------------|
| Baseline | 2.8% | 26.2% |
| Sequence Risk | 10.6% | 50.6% |
| Rate Shock | 7.4% | 48.0% |

LDI consistently outperforms Rule-of-Thumb on default risk across all scenarios, with the advantage most pronounced under stress conditions.
