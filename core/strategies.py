"""
Strategy implementations for lifecycle investment simulation.

This module contains strategy classes that implement the StrategyProtocol.
Strategies are simple functions mapping SimulationState -> StrategyActions.

Available strategies:
- LDIStrategy: Liability-Driven Investment with optional leverage
- RuleOfThumbStrategy: Classic 100-age rule with 4% withdrawal
- FixedConsumptionStrategy: 4% rule with LDI-style allocation
"""

from dataclasses import dataclass
from typing import Tuple

from .params import (
    SimulationState,
    StrategyActions,
    LifecycleParams,
    EconomicParams,
)


def _normalize_weights_no_leverage(
    target_fin_stock: float,
    target_fin_bond: float,
    target_fin_cash: float,
    fw: float,
    target_stock: float,
    target_bond: float,
    target_cash: float,
) -> Tuple[float, float, float]:
    """
    Normalize portfolio weights with no-short constraint.

    Clips negative weights to 0 and normalizes to sum to 1.0.
    """
    if fw <= 1e-6:
        # Clip MV targets (may be negative with unconstrained optimization)
        ws = max(0.0, target_stock)
        wb = max(0.0, target_bond)
        wc = max(0.0, target_cash)
        total = ws + wb + wc
        if total > 0:
            return ws / total, wb / total, wc / total
        return 0.0, 0.0, 1.0

    # Clip each component independently at 0
    w_s = max(0.0, target_fin_stock)
    w_b = max(0.0, target_fin_bond)
    w_c = max(0.0, target_fin_cash)

    total = w_s + w_b + w_c
    if total > 0:
        return w_s / total, w_b / total, w_c / total
    else:
        # All targets non-positive: clip MV targets and normalize
        ws = max(0.0, target_stock)
        wb = max(0.0, target_bond)
        wc = max(0.0, target_cash)
        total = ws + wb + wc
        if total > 0:
            return ws / total, wb / total, wc / total
        return 0.0, 0.0, 1.0


@dataclass
class LDIStrategy:
    """
    Liability-Driven Investment strategy.

    This implements the optimal lifecycle strategy:
    - Consumption: subsistence + consumption_rate * net_worth
    - Allocation: LDI hedge (target - HC component + expense hedge)

    The allocation adjusts financial holdings to offset implicit positions
    in human capital (an asset) and expense liabilities (a liability).

    Attributes:
        consumption_rate: Share of net worth consumed above subsistence (default from expected return)
        allow_leverage: If True, allows shorting and leverage (default False)
        name: Strategy name for display
    """
    consumption_rate: float = None  # None means derive from expected return
    allow_leverage: bool = False
    name: str = "LDI"

    def __call__(self, state: SimulationState) -> StrategyActions:
        """Compute LDI strategy actions given current state."""
        # Compute consumption rate if not specified
        if self.consumption_rate is None:
            r = state.current_rate
            sigma_s = state.econ_params.sigma_s
            sigma_r = state.econ_params.sigma_r
            rho = state.econ_params.rho
            D = state.econ_params.bond_duration
            mu_bond = state.econ_params.mu_bond

            w_s = state.target_stock
            w_b = state.target_bond

            # Expected (arithmetic mean) portfolio return - includes bond excess return
            expected_return = (
                w_s * (r + state.econ_params.mu_excess) +
                w_b * (r + mu_bond) +
                state.target_cash * r
            )

            # Full portfolio variance with correlation
            # Bond volatility = duration * rate volatility
            sigma_b = D * sigma_r
            # Stock-bond covariance (negative when rho > 0: rising rates hurt bonds)
            cov_sb = -D * sigma_s * sigma_r * rho
            portfolio_var = w_s**2 * sigma_s**2 + w_b**2 * sigma_b**2 + 2 * w_s * w_b * cov_sb

            # Median portfolio return = expected - 0.5 * variance (Jensen's correction)
            median_return = expected_return - 0.5 * portfolio_var
            consumption_rate = median_return + state.params.consumption_boost
        else:
            consumption_rate = self.consumption_rate

        # Consumption: subsistence + rate * net_worth
        subsistence = state.expenses
        variable = max(0, consumption_rate * state.net_worth)
        total_cons = subsistence + variable

        # Apply constraints based on lifecycle stage
        if state.is_working:
            # During working years: can't consume more than earnings
            if total_cons > state.earnings:
                total_cons = state.earnings
                variable = max(0, state.earnings - subsistence)
        else:
            # Retirement: can't consume more than financial wealth
            if state.financial_wealth <= 0:
                return StrategyActions(
                    total_consumption=0.0,
                    subsistence_consumption=0.0,
                    variable_consumption=0.0,
                    stock_weight=state.target_stock,
                    bond_weight=state.target_bond,
                    cash_weight=state.target_cash,
                )
            if total_cons > state.financial_wealth:
                total_cons = state.financial_wealth
                variable = max(0, state.financial_wealth - subsistence)
                if variable < 0:
                    subsistence = state.financial_wealth
                    variable = 0.0

        # LDI allocation: surplus optimization
        # Surplus = max(0, net_worth) where net_worth = HC + FW - PV(expenses)
        # Target = target_pct * surplus - HC_component + expense_component
        # Sum of targets = FW (no leverage needed). When surplus=0, all FW in bonds.
        surplus = max(0, state.net_worth)
        target_fin_stock = state.target_stock * surplus - state.hc_stock_component
        target_fin_bond = state.target_bond * surplus - state.hc_bond_component + state.exp_bond_component
        target_fin_cash = state.target_cash * surplus - state.hc_cash_component + state.exp_cash_component

        # Normalize to weights
        fw = state.financial_wealth
        if self.allow_leverage:
            # Raw weights (can be negative or >1)
            if fw > 1e-6:
                w_s = target_fin_stock / fw
                w_b = target_fin_bond / fw
                w_c = target_fin_cash / fw
            else:
                w_s, w_b, w_c = state.target_stock, state.target_bond, state.target_cash
        else:
            # No-short constraint
            w_s, w_b, w_c = _normalize_weights_no_leverage(
                target_fin_stock, target_fin_bond, target_fin_cash, fw,
                state.target_stock, state.target_bond, state.target_cash
            )

        return StrategyActions(
            total_consumption=total_cons,
            subsistence_consumption=subsistence,
            variable_consumption=variable,
            stock_weight=w_s,
            bond_weight=w_b,
            cash_weight=w_c,
        )


@dataclass
class RuleOfThumbStrategy:
    """
    Classic "rule of thumb" financial advisor strategy.

    This implements traditional heuristics:
    - Working years: Save savings_rate of income
    - Allocation: (100 - age)% in stocks, rest in bonds/cash to achieve target duration
    - Retirement: Allocation frozen at retirement age
    - Withdrawal: withdrawal_rate of initial retirement wealth (fixed)

    This is NOT optimal but represents common retail advice.

    Attributes:
        savings_rate: Fraction of income to save during working years (default 0.15)
        withdrawal_rate: Fixed withdrawal rate in retirement (default 0.04)
        target_duration: Target duration for fixed income allocation (default 6.0)
        name: Strategy name for display
    """
    savings_rate: float = 0.15
    withdrawal_rate: float = 0.04
    target_duration: float = 6.0
    name: str = "RoT"

    # Internal state for fixed retirement values
    _retirement_consumption: float = None
    _retirement_stock_weight: float = None
    _retirement_bond_weight: float = None
    _retirement_cash_weight: float = None

    def reset(self):
        """Reset internal state for a new simulation."""
        self._retirement_consumption = None
        self._retirement_stock_weight = None
        self._retirement_bond_weight = None
        self._retirement_cash_weight = None

    def __call__(self, state: SimulationState) -> StrategyActions:
        """Compute Rule-of-Thumb strategy actions given current state."""
        fw = state.financial_wealth
        age = state.age

        # Compute allocation: (100 - age)% stocks
        bond_duration = state.econ_params.bond_duration
        bond_weight_in_fi = min(1.0, self.target_duration / bond_duration) if bond_duration > 0 else 0.0

        if state.is_working:
            # Working years allocation
            stock_pct = max(0.0, min(1.0, (100 - age) / 100.0))
            fixed_income_pct = 1.0 - stock_pct
            bond_pct = fixed_income_pct * bond_weight_in_fi
            cash_pct = fixed_income_pct * (1.0 - bond_weight_in_fi)
        else:
            # Retirement: freeze allocation at retirement age
            if self._retirement_stock_weight is None:
                retirement_age = state.params.retirement_age
                self._retirement_stock_weight = max(0.0, min(1.0, (100 - retirement_age) / 100.0))
                retirement_fi = 1.0 - self._retirement_stock_weight
                self._retirement_bond_weight = retirement_fi * bond_weight_in_fi
                self._retirement_cash_weight = retirement_fi * (1.0 - bond_weight_in_fi)

            stock_pct = self._retirement_stock_weight
            bond_pct = self._retirement_bond_weight
            cash_pct = self._retirement_cash_weight

        # Compute consumption
        subsistence = state.expenses

        if state.is_working:
            # Working years: save savings_rate of earnings
            baseline_consumption = state.earnings * (1.0 - self.savings_rate)

            if baseline_consumption >= subsistence:
                total_cons = baseline_consumption
                variable = baseline_consumption - subsistence
            else:
                # Can't meet baseline, try to cover subsistence
                available = state.earnings + fw
                if available >= subsistence:
                    total_cons = subsistence
                    variable = 0.0
                else:
                    total_cons = max(0, available)
                    subsistence = total_cons
                    variable = 0.0
        else:
            # Retirement: 4% of initial retirement wealth
            if self._retirement_consumption is None:
                self._retirement_consumption = self.withdrawal_rate * fw

            if fw <= 0:
                return StrategyActions(
                    total_consumption=0.0,
                    subsistence_consumption=0.0,
                    variable_consumption=0.0,
                    stock_weight=stock_pct,
                    bond_weight=bond_pct,
                    cash_weight=cash_pct,
                )

            target_consumption = max(self._retirement_consumption, subsistence)
            if fw < target_consumption:
                total_cons = fw
                subsistence = min(fw, state.expenses)
                variable = max(0, fw - state.expenses)
            else:
                total_cons = target_consumption
                variable = target_consumption - subsistence

        return StrategyActions(
            total_consumption=total_cons,
            subsistence_consumption=subsistence,
            variable_consumption=variable,
            stock_weight=stock_pct,
            bond_weight=bond_pct,
            cash_weight=cash_pct,
        )


@dataclass
class FixedConsumptionStrategy:
    """
    Fixed consumption (4% rule) strategy with LDI-style allocation.

    This combines:
    - Working years: consume only subsistence, save the rest
    - Retirement: fixed percentage of retirement wealth each year
    - Allocation: LDI-style (adjusts for HC and expenses)

    This differs from RuleOfThumbStrategy in using optimal LDI allocation
    rather than the naive (100-age) rule.

    Attributes:
        withdrawal_rate: Fixed withdrawal rate in retirement (default 0.04)
        allow_leverage: If True, allows shorting and leverage (default False)
        name: Strategy name for display
    """
    withdrawal_rate: float = 0.04
    allow_leverage: bool = False
    name: str = "Fixed4%"

    # Internal state
    _retirement_consumption: float = None

    def reset(self):
        """Reset internal state for a new simulation."""
        self._retirement_consumption = None

    def __call__(self, state: SimulationState) -> StrategyActions:
        """Compute Fixed Consumption strategy actions given current state."""
        fw = state.financial_wealth
        subsistence = state.expenses

        if state.is_working:
            # Working years: consume only subsistence
            total_cons = subsistence
            variable = 0.0
        else:
            # Retirement: fixed consumption
            if self._retirement_consumption is None:
                self._retirement_consumption = self.withdrawal_rate * fw

            if fw <= 0:
                return StrategyActions(
                    total_consumption=0.0,
                    subsistence_consumption=0.0,
                    variable_consumption=0.0,
                    stock_weight=state.target_stock,
                    bond_weight=state.target_bond,
                    cash_weight=state.target_cash,
                )

            if fw < self._retirement_consumption:
                total_cons = fw
                subsistence = min(state.expenses, fw)
                variable = max(0, total_cons - subsistence)
            else:
                total_cons = self._retirement_consumption
                variable = max(0, total_cons - subsistence)

        # LDI allocation: surplus optimization (same as LDIStrategy)
        surplus = max(0, state.net_worth)
        target_fin_stock = state.target_stock * surplus - state.hc_stock_component
        target_fin_bond = state.target_bond * surplus - state.hc_bond_component + state.exp_bond_component
        target_fin_cash = state.target_cash * surplus - state.hc_cash_component + state.exp_cash_component

        if self.allow_leverage:
            if fw > 1e-6:
                w_s = target_fin_stock / fw
                w_b = target_fin_bond / fw
                w_c = target_fin_cash / fw
            else:
                w_s, w_b, w_c = state.target_stock, state.target_bond, state.target_cash
        else:
            w_s, w_b, w_c = _normalize_weights_no_leverage(
                target_fin_stock, target_fin_bond, target_fin_cash, fw,
                state.target_stock, state.target_bond, state.target_cash
            )

        return StrategyActions(
            total_consumption=total_cons,
            subsistence_consumption=subsistence,
            variable_consumption=variable,
            stock_weight=w_s,
            bond_weight=w_b,
            cash_weight=w_c,
        )
