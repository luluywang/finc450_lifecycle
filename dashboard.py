#!/usr/bin/env python3
"""
Dashboard comparing LDI vs Rule-of-Thumb strategies.

Shows distribution of net worth (FW + HC - PV Expenses) over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from lifecycle_strategy import (
    LifecycleParams,
    EconomicParams,
    compute_lifecycle_median_path,
    compute_rule_of_thumb_strategy,
    compute_pv_consumption,
    compute_present_value,
    run_strategy_comparison,
    generate_correlated_shocks,
    simulate_interest_rates,
    simulate_stock_returns,
    compute_mv_optimal_allocation,
)
from retirement_simulation import generate_correlated_shocks, simulate_interest_rates, simulate_stock_returns


def run_net_worth_comparison(
    params: LifecycleParams = None,
    econ_params: EconomicParams = None,
    n_simulations: int = 200,
    random_seed: int = 42,
    rot_savings_rate: float = 0.15,
    rot_target_duration: float = 6.0,
    rot_withdrawal_rate: float = 0.04,
):
    """
    Run Monte Carlo comparison tracking net worth for both strategies.

    Net Worth = Financial Wealth + Human Capital - PV(Future Expenses)
    """
    if params is None:
        params = LifecycleParams()
    if econ_params is None:
        econ_params = EconomicParams()

    total_years = params.end_age - params.start_age
    working_years = params.retirement_age - params.start_age
    ages = np.arange(params.start_age, params.end_age)

    # Get median path for human capital and PV expenses (same for both strategies)
    median_result = compute_lifecycle_median_path(params, econ_params)
    human_capital = median_result.human_capital
    pv_expenses = median_result.pv_expenses
    earnings = median_result.earnings
    expenses = median_result.expenses

    # Compute optimal target allocations
    if params.gamma > 0:
        target_stock, target_bond, target_cash = compute_mv_optimal_allocation(
            mu_stock=econ_params.mu_excess,
            mu_bond=econ_params.mu_bond,
            sigma_s=econ_params.sigma_s,
            sigma_r=econ_params.sigma_r,
            rho=econ_params.rho,
            duration=econ_params.bond_duration,
            gamma=params.gamma
        )
    else:
        target_stock = params.target_stock_allocation
        target_bond = params.target_bond_allocation
        target_cash = 1.0 - target_stock - target_bond

    # Generate random shocks
    rng = np.random.default_rng(random_seed)
    rate_shocks, stock_shocks = generate_correlated_shocks(
        total_years, n_simulations, econ_params.rho, rng
    )

    # Simulate paths
    initial_rate = econ_params.r_bar
    rate_paths = simulate_interest_rates(
        initial_rate, total_years, n_simulations, econ_params, rate_shocks
    )
    stock_return_paths = simulate_stock_returns(rate_paths, econ_params, stock_shocks)

    # Initialize arrays
    ldi_wealth_paths = np.zeros((n_simulations, total_years))
    ldi_net_worth_paths = np.zeros((n_simulations, total_years))
    ldi_consumption_paths = np.zeros((n_simulations, total_years))

    rot_wealth_paths = np.zeros((n_simulations, total_years))
    rot_net_worth_paths = np.zeros((n_simulations, total_years))
    rot_consumption_paths = np.zeros((n_simulations, total_years))

    # Consumption rate for LDI
    r = econ_params.r_bar
    avg_median_return = (
        target_stock * (r + econ_params.mu_excess) +
        target_bond * r +
        target_cash * r
    )
    consumption_rate = avg_median_return + params.consumption_boost

    # Run simulations
    for sim in range(n_simulations):
        # ---- LDI STRATEGY ----
        ldi_wealth_paths[sim, 0] = params.initial_wealth
        ldi_defaulted = False

        for t in range(total_years):
            fw = ldi_wealth_paths[sim, t]

            # Dynamic revaluation using current simulated rate
            current_rate = rate_paths[sim, t]

            # PV of remaining expenses
            remaining_expenses = expenses[t:]
            pv_exp = compute_present_value(remaining_expenses, current_rate,
                                           econ_params.phi, econ_params.r_bar)

            # Human capital = PV of remaining earnings
            if t < working_years:
                remaining_earnings = earnings[t:working_years]
                hc = compute_present_value(remaining_earnings, current_rate,
                                           econ_params.phi, econ_params.r_bar)
            else:
                hc = 0.0

            net_worth = hc + fw - pv_exp

            ldi_net_worth_paths[sim, t] = net_worth

            # Consumption
            subsistence = expenses[t]
            variable = max(0, consumption_rate * net_worth)
            total_cons = subsistence + variable

            if t < working_years:
                if total_cons > earnings[t]:
                    total_cons = earnings[t]
            else:
                if ldi_defaulted:
                    total_cons = 0
                elif fw <= 0:
                    ldi_defaulted = True
                    total_cons = 0
                elif total_cons > fw:
                    total_cons = fw

            ldi_consumption_paths[sim, t] = total_cons

            # Evolve wealth
            if t < total_years - 1 and not ldi_defaulted:
                savings = earnings[t] - total_cons

                stock_ret = stock_return_paths[sim, t]
                bond_ret = rate_paths[sim, t] + econ_params.mu_bond
                cash_ret = rate_paths[sim, t]

                portfolio_return = (
                    target_stock * stock_ret +
                    target_bond * bond_ret +
                    target_cash * cash_ret
                )
                ldi_wealth_paths[sim, t + 1] = fw * (1 + portfolio_return) + savings

        # ---- RULE OF THUMB STRATEGY ----
        rot_result = compute_rule_of_thumb_strategy(
            params=params,
            econ_params=econ_params,
            savings_rate=rot_savings_rate,
            withdrawal_rate=rot_withdrawal_rate,
            target_duration=rot_target_duration,
            stock_returns=stock_return_paths[sim, :],
            interest_rates=rate_paths[sim, :],
        )

        rot_wealth_paths[sim, :] = rot_result.financial_wealth
        rot_consumption_paths[sim, :] = rot_result.total_consumption

        # Compute net worth for RoT with dynamic revaluation
        for t in range(total_years):
            current_rate = rate_paths[sim, t]

            # PV of remaining expenses
            remaining_expenses = expenses[t:]
            pv_exp = compute_present_value(remaining_expenses, current_rate,
                                           econ_params.phi, econ_params.r_bar)

            # Human capital = PV of remaining earnings
            if t < working_years:
                remaining_earnings = earnings[t:working_years]
                hc = compute_present_value(remaining_earnings, current_rate,
                                           econ_params.phi, econ_params.r_bar)
            else:
                hc = 0.0

            rot_net_worth_paths[sim, t] = rot_result.financial_wealth[t] + hc - pv_exp

    return {
        'ages': ages,
        'n_simulations': n_simulations,
        'human_capital': human_capital,
        'pv_expenses': pv_expenses,
        # LDI
        'ldi_wealth_paths': ldi_wealth_paths,
        'ldi_net_worth_paths': ldi_net_worth_paths,
        'ldi_consumption_paths': ldi_consumption_paths,
        # RoT
        'rot_wealth_paths': rot_wealth_paths,
        'rot_net_worth_paths': rot_net_worth_paths,
        'rot_consumption_paths': rot_consumption_paths,
        # Params
        'params': params,
        'econ_params': econ_params,
    }


def create_dashboard(results, figsize=(16, 12)):
    """Create a dashboard comparing LDI vs RoT strategies."""

    ages = results['ages']
    params = results['params']
    n_sims = results['n_simulations']

    x = np.arange(len(ages))
    retirement_x = params.retirement_age - params.start_age

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Colors
    color_ldi = '#2ecc71'
    color_rot = '#3498db'
    alpha_fan = 0.25

    percentiles = [5, 25, 50, 75, 95]

    # Helper function for fan charts
    def plot_fan(ax, paths, color, label):
        p5 = np.percentile(paths, 5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        ax.fill_between(x, p5, p95, alpha=alpha_fan, color=color)
        ax.fill_between(x, p25, p75, alpha=alpha_fan + 0.15, color=color)
        ax.plot(x, p50, color=color, linewidth=2, label=f'{label} Median')

        return p50

    # ---- (0,0): Net Worth Distribution ----
    ax = axes[0, 0]
    plot_fan(ax, results['ldi_net_worth_paths'], color_ldi, 'LDI')
    plot_fan(ax, results['rot_net_worth_paths'], color_rot, 'RoT')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('$ (000s)')
    ax.set_title('Net Worth (FW + HC - PV Expenses)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=50, linscale=0.5)

    # ---- (0,1): Financial Wealth Distribution ----
    ax = axes[0, 1]
    plot_fan(ax, results['ldi_wealth_paths'], color_ldi, 'LDI')
    plot_fan(ax, results['rot_wealth_paths'], color_rot, 'RoT')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('$ (000s)')
    ax.set_title('Financial Wealth')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=50, linscale=0.5)

    # ---- (0,2): Consumption Distribution ----
    ax = axes[0, 2]
    plot_fan(ax, results['ldi_consumption_paths'], color_ldi, 'LDI')
    plot_fan(ax, results['rot_consumption_paths'], color_rot, 'RoT')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('$ (000s)')
    ax.set_title('Consumption')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog', linthresh=50, linscale=0.5)

    # ---- (1,0): Human Capital and PV Expenses (deterministic) ----
    ax = axes[1, 0]
    ax.fill_between(x, 0, results['human_capital'], alpha=0.5, color='#f39c12', label='Human Capital')
    ax.fill_between(x, 0, -results['pv_expenses'], alpha=0.5, color='#e74c3c', label='PV Expenses (neg)')
    ax.plot(x, results['human_capital'] - results['pv_expenses'],
            color='purple', linewidth=2, label='HC - PV Exp')
    ax.axvline(x=retirement_x, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Years from Career Start')
    ax.set_ylabel('$ (000s)')
    ax.set_title('Human Capital & PV Expenses (Same for Both)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (1,1): Terminal Wealth Distribution ----
    ax = axes[1, 1]
    ldi_final = results['ldi_wealth_paths'][:, -1]
    rot_final = results['rot_wealth_paths'][:, -1]

    # Use log scale with floor at $10k
    floor_val = 10
    ldi_floored = np.maximum(ldi_final, floor_val)
    rot_floored = np.maximum(rot_final, floor_val)
    max_val = max(np.percentile(rot_floored, 99), np.percentile(ldi_floored, 99))
    bins = np.geomspace(floor_val, max_val, 40)
    ax.hist(ldi_floored, bins=bins, alpha=0.6, color=color_ldi, label=f'LDI (med=${np.median(ldi_final):,.0f}k)', edgecolor='white')
    ax.hist(rot_floored, bins=bins, alpha=0.6, color=color_rot, label=f'RoT (med=${np.median(rot_final):,.0f}k)', edgecolor='white')
    ax.axvline(x=max(np.median(ldi_final), floor_val), color=color_ldi, linestyle='--', linewidth=2)
    ax.axvline(x=max(np.median(rot_final), floor_val), color=color_rot, linestyle='--', linewidth=2)
    ax.set_xlabel('Final Wealth at Age 85 ($ 000s)')
    ax.set_ylabel('Count')
    ax.set_title('Terminal Wealth Distribution')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xscale('log')

    # ---- (1,2): PV Consumption Distribution ----
    ax = axes[1, 2]
    r = results['econ_params'].r_bar

    ldi_pv = np.array([compute_pv_consumption(results['ldi_consumption_paths'][i], r)
                       for i in range(n_sims)])
    rot_pv = np.array([compute_pv_consumption(results['rot_consumption_paths'][i], r)
                       for i in range(n_sims)])

    # Use log scale with floor at $100k
    floor_val = 100
    ldi_pv_floored = np.maximum(ldi_pv, floor_val)
    rot_pv_floored = np.maximum(rot_pv, floor_val)
    min_val = min(ldi_pv_floored.min(), rot_pv_floored.min())
    max_val = max(np.percentile(ldi_pv_floored, 99), np.percentile(rot_pv_floored, 99))
    bins = np.geomspace(min_val, max_val, 40)
    ax.hist(ldi_pv_floored, bins=bins, alpha=0.6, color=color_ldi,
            label=f'LDI (med=${np.median(ldi_pv):,.0f}k)', edgecolor='white')
    ax.hist(rot_pv_floored, bins=bins, alpha=0.6, color=color_rot,
            label=f'RoT (med=${np.median(rot_pv):,.0f}k)', edgecolor='white')
    ax.axvline(x=max(np.median(ldi_pv), floor_val), color=color_ldi, linestyle='--', linewidth=2)
    ax.axvline(x=max(np.median(rot_pv), floor_val), color=color_rot, linestyle='--', linewidth=2)
    ax.set_xlabel('PV Lifetime Consumption ($ 000s)')
    ax.set_ylabel('Count')
    ax.set_title('PV Consumption Distribution')
    ax.set_xscale('log')
    ax.legend(loc='upper right', fontsize=9)

    # Add summary stats as text
    ldi_default_rate = np.mean(results['ldi_wealth_paths'][:, -1] <= 0) * 100
    rot_default_rate = np.mean(results['rot_wealth_paths'][:, -1] <= 0) * 100

    summary = f"""
Summary Statistics (n={n_sims} simulations)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    LDI         RoT
Median PV Cons:   ${np.median(ldi_pv):,.0f}k    ${np.median(rot_pv):,.0f}k
Median Final FW:  ${np.median(ldi_final):,.0f}k    ${np.median(rot_final):,.0f}k
Default Rate:     {ldi_default_rate:.1f}%        {rot_default_rate:.1f}%

LDI advantage in PV consumption: +${np.median(ldi_pv) - np.median(rot_pv):,.0f}k ({(np.median(ldi_pv)/np.median(rot_pv)-1)*100:+.1f}%)
"""

    fig.text(0.02, 0.02, summary, fontsize=10, fontfamily='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('LDI vs Rule-of-Thumb Strategy Dashboard', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


if __name__ == '__main__':
    print("Running strategy comparison simulations...")

    params = LifecycleParams(consumption_boost=0.0)
    econ_params = EconomicParams()

    results = run_net_worth_comparison(
        params=params,
        econ_params=econ_params,
        n_simulations=500,
        random_seed=42,
    )

    print("Creating dashboard...")
    fig = create_dashboard(results)

    # Save to file
    fig.savefig('strategy_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Dashboard saved to strategy_dashboard.png")
