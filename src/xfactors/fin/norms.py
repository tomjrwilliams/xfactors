
from __future__ import annotations
from this import d

import typing
import collections
import functools
import itertools
import operator

import datetime

import numpy
import pandas

import jax
import jaxopt
import optax

import xtuples as xt

from .. import xfactors as xf

from . import stmts

# ---------------------------------------------------------------

if_none = xf.utils.funcs.if_none
if_none_lazy = xf.utils.funcs.if_none_lazy

# ---------------------------------------------------------------

V = stmts.V

# NOTE: no need for the independent vs dependent notion here
# also, no model type semantic links given we can't have
# type vars that refer to type vars
# and presumably there'll be different spec models
# with different fields

IS_Norm = typing.TypeVar("IS_Norm", None, V)
CF_Norm = typing.TypeVar("CF_Norm", None, V)
BS_Norm = typing.TypeVar("BS_Norm", None, V)
Other_Norm = typing.TypeVar("Other_Norm", None, V)

# ---------------------------------------------------------------

# TODO: optional kwarg specifying the _to key

def fields_income_statement():
    return

def fields_cash_flow():
    return

def fields_balance_sheet():
    return

def fields_other():
    return

# ---------------------------------------------------------------

def values_income_statement():
    return

def values_cash_flow():
    return

def values_balance_sheet():
    return

def values_other():
    return

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Norms(typing.NamedTuple):

    # as per stmts, not all are independent
    # eg. profit_gross_to_revenue is determined given rev and cogs_to_revenue, etc.

    prev: stmts.Statements
    curr: stmts.Statements

    # --

    def calc_k_to_k(self, k1, k2, curr = True, inv = False):
        v2 = getattr(self, k2)
        v1 = (
            getattr(self.curr, k1) if curr else getattr(self.prev, k1)
        )
        return v1 / v2 if not inv else v2 / v1

    def calc_to_revenue(self, k, **kwargs):
        return self.calc_k_to_k("revenue", k, **kwargs)

    def calc_to_assets(self, k, **kwargs):
        return self.calc_k_to_k("assets", k, **kwargs)

    def calc_to_liabilities(self, k, **kwargs):
        return self.calc_k_to_k("liabilities", k, **kwargs)

    def calc_to_equity(self, k, **kwargs):
        return self.calc_k_to_k("equity", k, **kwargs)

    def calc_to_working_cap_gross(self, k, **kwargs):
        return self.calc_k_to_k("working_cap_gross", k, **kwargs)

    # -- income statement --

    revenue_growth: IS_Norm = 0.
    cogs_to_revenue: IS_Norm = 0.

    profit_gross_to_revenue: IS_Norm = 0.
    profit_gross_res_to_revenue: IS_Norm = 0.

    def calc_revenue_growth(self):
        revenue_curr = self.curr.revenue
        revenue_prev = self.prev.revenue
        # curr = prev + (prev * g)
        return (revenue_curr - revenue_prev) / revenue_prev

    def calc_cogs_to_revenue(self):
        return self.calc_to_revenue("cogs")

    def calc_profit_gross_to_revenue(self):
        return self.calc_to_revenue("profit_gross")

    def calc_profit_gross_res_to_revenue(self):
        return self.calc_to_revenue("profit_gross_res")
        
    # --

    rd_to_revenue: IS_Norm = 0.
    sga_to_revenue: IS_Norm = 0.
    profit_operating_to_revenue: IS_Norm = 0.
    profit_operating_res_to_revenue: IS_Norm = 0.

    def calc_rd_to_revenue(self):
        return self.calc_to_revenue("rd")

    def calc_sga_to_revenue(self):
        return self.calc_to_revenue("sga")

    def calc_profit_operating_to_revenue(self):
        return self.calc_to_revenue("profit_operating")

    def calc_profit_operating_res_to_revenue(self):
        return self.calc_to_revenue("profit_operating_res")

    # --
    
    # to earnings?

    interest_in_to_revenue: IS_Norm = 0.
    interest_in_to_assets: IS_Norm = 0.
    interest_out_to_revenue: IS_Norm = 0.
    interest_out_to_liabilities: IS_Norm = 0.

    expense_other_to_revenue: IS_Norm = 0.

    profit_pretax_to_revenue: IS_Norm = 0.
    profit_pretax_res_to_revenue: IS_Norm = 0.

    def calc_interest_in_to_revenue(self):
        return self.calc_to_revenue("interest_in")

    def calc_interest_in_to_assets(self):
        return self.calc_to_assets("interest_in")

    def calc_interest_out_to_revenue(self):
        return self.calc_to_revenue("interest_out")

    def calc_interest_out_to_liabilities(self):
        return self.calc_to_liabilities("interest_out")

    def calc_expense_other_to_revenue(self):
        return self.calc_to_revenue("expense_other")

    def calc_profit_pretax_to_revenue(self):
        return self.calc_to_revenue("profit_pretax")

    def calc_profit_pretax_res_to_revenue(self):
        return self.calc_to_revenue("profit_pretax_res")

    # --

    tax_to_revenue: IS_Norm = 0.
    profit_net_to_revenue: IS_Norm = 0.
    profit_net_res_to_revenue: IS_Norm = 0.

    def calc_tax_to_revenue(self):
        return self.calc_to_revenue("tax")

    def calc_profit_net_to_revenue(self):
        return self.calc_to_revenue("profit_net")

    def calc_profit_net_res_to_revenue(self):
        return self.calc_to_revenue("profit_net_res")

    def calc_income_statement(self):
        return self._replace(
            revenue_growth=(
                self.calc_revenue_growth()
            ),
            cogs_to_revenue=(
                self.calc_cogs_to_revenue()
            ),
            profit_gross_to_revenue=(
                self.calc_profit_gross_to_revenue()
            ),
            profit_gross_res_to_revenue=(
                self.calc_profit_gross_res_to_revenue()
            ),
            rd_to_revenue=(
                self.calc_rd_to_revenue()
            ),
            sga_to_revenue=(
                self.calc_sga_to_revenue()
            ),
            profit_operating_to_revenue=(
                self.calc_profit_operating_to_revenue()
            ),
            profit_operating_res_to_revenue=(
                self.calc_profit_operating_res_to_revenue()
            ),
            interest_in_to_revenue=(
                self.calc_interest_in_to_revenue()
            ),
            interest_in_to_assets=(
                self.calc_interest_in_to_assets()
            ),
            interest_out_to_revenue=(
                self.calc_interest_out_to_revenue()
            ),
            interest_out_to_liabilities=(
                self.calc_interest_out_to_liabilities()
            ),
            expense_other_to_revenue=(
                self.calc_expense_other_to_revenue()
            ),
            profit_pretax_to_revenue=(
                self.calc_profit_pretax_to_revenue()
            ),
            profit_pretax_res_to_revenue=(
                self.calc_profit_pretax_res_to_revenue()
            ),
            tax_to_revenue=(
                self.calc_tax_to_revenue()
            ),
            profit_net_to_revenue=(
                self.calc_profit_net_to_revenue()
            ),
            profit_net_res_to_revenue=(
                self.calc_profit_net_res_to_revenue()
            ),
        )

    # -- cash flows --

    # to earnings?

    da_to_revenue: CF_Norm = 0.
    # stock based comp : CF_Norm = 0.
    working_cap_change_to_revenue: CF_Norm = 0.
    cash_other_to_revenue: CF_Norm = 0.
    cash_operating_to_revenue: CF_Norm = 0.
    cash_operating_res_to_revenue: CF_Norm = 0.

    def calc_da_to_revenue(self):
        return self.calc_to_revenue("da")

    def calc_working_cap_change_to_revenue(self):
        return self.calc_to_revenue("working_cap_change")

    def calc_cash_other_to_revenue(self):
        return self.calc_to_revenue("cash_other")

    def calc_cash_operating_to_revenue(self):
        return self.calc_to_revenue("cash_operating")

    def calc_cash_operating_res_to_revenue(self):
        return self.calc_to_revenue("cash_operating_res")

    # --

    # to earnings?

    capex_to_revenue: CF_Norm = 0.
    cash_intangibles_to_revenue: CF_Norm = 0.
    cash_investing_to_revenue: CF_Norm = 0.
    cash_investing_res_to_revenue: CF_Norm = 0.

    def calc_capex_to_revenue(self):
        return self.calc_to_revenue("capex")

    def calc_cash_intangibles_to_revenue(self):
        return self.calc_to_revenue("cash_intangibles")

    def calc_cash_investing_to_revenue(self):
        return self.calc_to_revenue("cash_investing")

    def calc_cash_investing_res_to_revenue(self):
        return self.calc_to_revenue("cash_investing_res")
        
    # --

    cash_debt_to_revenue: CF_Norm = 0.
    cash_equity_to_revenue: CF_Norm = 0.

    # - dividends
    # + change share issuance
    # - share repurchase

    cash_financing_to_revenue: CF_Norm = 0.
    cash_financing_res_to_revenue: CF_Norm = 0.

    def calc_cash_debt_to_revenue(self):
        return self.calc_to_revenue("cash_debt")

    def calc_cash_equity_to_revenue(self):
        return self.calc_to_revenue("cash_equity")

    def calc_cash_financing_to_revenue(self):
        return self.calc_to_revenue("cash_financing")

    def calc_cash_financing_res_to_revenue(self):
        return self.calc_to_revenue("cash_financing_res")
        
    # --
    
    cash_net_to_revenue: CF_Norm = 0.
    cash_net_res_to_revenue: CF_Norm = 0.

    def calc_cash_net_to_revenue(self):
        return self.calc_to_revenue("cash_net")

    def calc_cash_net_res_to_revenue(self):
        return self.calc_to_revenue("cash_net_res")

    def calc_cash_flow(self):
        return self._replace(
            da_to_revenue = (
                self.calc_da_to_revenue()
            ),
            working_cap_change_to_revenue = (
                self.calc_working_cap_change_to_revenue()
            ),
            cash_other_to_revenue = (
                self.calc_cash_other_to_revenue()
            ),
            cash_operating_to_revenue = (
                self.calc_cash_operating_to_revenue()
            ),
            cash_operating_res_to_revenue = (
                self.calc_cash_operating_res_to_revenue()
            ),
            capex_to_revenue = (
                self.calc_capex_to_revenue()
            ),
            cash_intangibles_to_revenue = (
                self.calc_cash_intangibles_to_revenue()
            ),
            cash_investing_to_revenue = (
                self.calc_cash_investing_to_revenue()
            ),
            cash_investing_res_to_revenue = (
                self.calc_cash_investing_res_to_revenue()
            ),
            cash_debt_to_revenue = (
                self.calc_cash_debt_to_revenue()
            ),
            cash_equity_to_revenue = (
                self.calc_cash_equity_to_revenue()
            ),
            cash_financing_to_revenue = (
                self.calc_cash_financing_to_revenue()
            ),
            cash_financing_res_to_revenue = (
                self.calc_cash_financing_res_to_revenue()
            ),
            cash_net_to_revenue = (
                self.calc_cash_net_to_revenue()
            ),
            cash_net_res_to_revenue = (
                self.calc_cash_net_res_to_revenue()
            ),
        )

    # --

    # -- balance sheet --

    # to non cash assets?

    # debt etc. vs cash
    # versus earnings

    cash_to_assets: BS_Norm = 0.
    securities_to_assets: BS_Norm = 0.

    acc_receivable_to_assets: BS_Norm = 0.
    acc_receivable_to_working_cap_gross: BS_Norm = 0.
    acc_receivable_to_revenue: BS_Norm = 0.

    inventory_to_assets: BS_Norm = 0.
    inventory_to_revenue: BS_Norm = 0.

    tax_deferred_to_assets: BS_Norm = 0.
    assets_other_to_assets: BS_Norm = 0.

    ppe_to_assets: BS_Norm = 0.
    ppe_to_revenue: BS_Norm = 0.

    intangibles_to_assets: BS_Norm = 0.
    intangibles_to_revenue: BS_Norm = 0.

    # 

    assets_to_revenue: BS_Norm = 0.
    assets_res_to_assets: BS_Norm = 0.

    def calc_cash_to_assets(self):
        return self.calc_to_assets("cash")

    def calc_securities_to_assets(self):
        return self.calc_to_assets("securities")

    def calc_acc_receivable_to_assets(self):
        return self.calc_to_assets("acc_receivable")

    def calc_acc_receivable_to_working_cap_gross(self):
        return self.calc_to_working_cap_gross("acc_receivable")

    def calc_acc_receivable_to_revenue(self):
        return self.calc_to_revenue("acc_receivable")

    def calc_inventory_to_assets(self):
        return self.calc_to_assets("inventory")

    def calc_inventory_to_revenue(self):
        return self.calc_to_revenue("inventory")

    def calc_tax_deferred_to_assets(self):
        return self.calc_to_assets("tax_deferred")

    def calc_assets_other_to_assets(self):
        return self.calc_to_assets("assets_other")

    def calc_ppe_to_assets(self):
        return self.calc_to_assets("ppe")

    def calc_ppe_to_revenue(self):
        return self.calc_to_revenue("ppe")

    def calc_intangibles_to_assets(self):
        return self.calc_to_assets("intangibles")
        
    def calc_intangibles_to_revenue(self):
        return self.calc_to_revenue("intangibles")

    # --

    def calc_assets_to_revenue(self):
        return self.calc_to_revenue("assets")

    def calc_assets_res_to_assets(self):
        return self.calc_to_assets("assets_res")

    # --
    
    working_cap_gross_to_revenue: BS_Norm = 0.

    def calc_working_cap_gross_to_revenue(self):
        return self.calc_to_revenue("working_cap_gross")

    working_cap_net_to_revenue: BS_Norm = 0.

    def calc_working_cap_net_to_revenue(self):
        return self.calc_to_revenue("working_cap_net")
        
    # --

    # to liabilities

    acc_payable_to_liabilities: BS_Norm = 0.
    acc_payable_to_working_cap_gross: BS_Norm = 0.
    acc_payable_to_revenue: BS_Norm = 0.

    accrued_expenses_to_liabilities: BS_Norm = 0.
    revenue_deferred_to_liabilities: BS_Norm = 0.
    
    debt_to_liabilities: BS_Norm = 0.
    debt_to_revenue: BS_Norm = 0.

    liabilities_other_to_liabilities: BS_Norm = 0.

    #

    liabilities_to_revenue: BS_Norm = 0.
    liabilities_res_to_liabilities: BS_Norm = 0.

    def calc_acc_payable_to_liabilities(self):
        return self.calc_to_liabilities("acc_payable")

    def calc_acc_payable_to_working_cap_gross(self):
        return self.calc_to_working_cap_gross("acc_payable")

    def calc_acc_payable_to_revenue(self):
        return self.calc_to_revenue("acc_payable")

    def calc_accrued_expenses_to_liabilities(self):
        return self.calc_to_liabilities("accrued_expenses")

    def calc_revenue_deferred_to_liabilities(self):
        return self.calc_to_liabilities("revenue_deferred")

    def calc_debt_to_liabilities(self):
        return self.calc_to_liabilities("debt")

    def calc_debt_to_revenue(self):
        return self.calc_to_revenue("debt")

    def calc_liabilities_other_to_liabilities(self):
        return self.calc_to_liabilities("liabilities_other")
    
    #

    def calc_liabilities_to_revenue(self):
        return self.calc_to_revenue("liabilities")

    def calc_liabilities_res_to_liabilities(self):
        return self.calc_to_liabilities("liabilities_res")

    # --

    # to equity

    equity_common_to_equity: BS_Norm = 0.
    equity_treasury_to_equity: BS_Norm = 0.
    retained_earnings_to_equity: BS_Norm = 0.

    #

    equity_to_revenue: BS_Norm = 0.
    equity_to_assets: BS_Norm = 0.

    equity_res_to_equity: BS_Norm = 0.

    def calc_equity_common_to_equity(self):
        return self.calc_to_equity("equity_common")

    def calc_equity_treasury_to_equity(self):
        return self.calc_to_equity("equity_treasury")

    def calc_retained_earnings_to_equity(self):
        return self.calc_to_equity("retained_earnings")

    #

    def calc_equity_to_revenue(self):
        return self.calc_to_revenue("equity")

    def calc_equity_to_assets(self):
        return self.calc_to_assets("equity")

    def calc_equity_res_to_equity(self):
        return self.calc_to_equity("equity_res")

    #  --

    def calc_balance_sheet(self):
        return self._replace(
            cash_to_assets=(
                self.calc_cash_to_assets()
            ),
            securities_to_assets=(
                self.calc_securities_to_assets()
            ),
            acc_receivable_to_assets=(
                self.calc_acc_receivable_to_assets()
            ),
            acc_receivable_to_working_cap_gross=(
                self.calc_acc_receivable_to_working_cap_gross()
            ),
            acc_receivable_to_revenue=(
                self.calc_acc_receivable_to_revenue()
            ),
            inventory_to_assets=(
                self.calc_inventory_to_assets()
            ),
            inventory_to_revenue=(
                self.calc_inventory_to_revenue()
            ),
            tax_deferred_to_assets=(
                self.calc_tax_deferred_to_assets()
            ),
            assets_other_to_assets=(
                self.calc_assets_other_to_assets()
            ),
            ppe_to_assets=(
                self.calc_ppe_to_assets()
            ),
            ppe_to_revenue=(
                self.calc_ppe_to_revenue()
            ),
            intangibles_to_assets=(
                self.calc_intangibles_to_assets()
            ),
            intangibles_to_revenue=(
                self.calc_intangibles_to_revenue()
            ),
            assets_to_revenue=(
                self.calc_assets_to_revenue()
            ),
            assets_res_to_assets=(
                self.calc_assets_res_to_assets()
            ),
            working_cap_gross_to_revenue=(
                self.calc_working_cap_gross_to_revenue()
            ),
            working_cap_net_to_revenue=(
                self.calc_working_cap_net_to_revenue()
            ),
            acc_payable_to_liabilities=(
                self.calc_acc_payable_to_liabilities()
            ),
            acc_payable_to_working_cap_gross=(
                self.calc_acc_payable_to_working_cap_gross()
            ),
            acc_payable_to_revenue=(
                self.calc_acc_payable_to_revenue()
            ),
            accrued_expenses_to_liabilities=(
                self.calc_accrued_expenses_to_liabilities()
            ),
            revenue_deferred_to_liabilities=(
                self.calc_revenue_deferred_to_liabilities()
            ),
            debt_to_liabilities=(
                self.calc_debt_to_liabilities()
            ),
            debt_to_revenue=(
                self.calc_debt_to_revenue()
            ),
            liabilities_other_to_liabilities=(
                self.calc_liabilities_other_to_liabilities()
            ),
            liabilities_to_revenue=(
                self.calc_liabilities_to_revenue()
            ),
            liabilities_res_to_liabilities=(
                self.calc_liabilities_res_to_liabilities()
            ),
            equity_common_to_equity=(
                self.calc_equity_common_to_equity()
            ),
            equity_treasury_to_equity=(
                self.calc_equity_treasury_to_equity()
            ),
            retained_earnings_to_equity=(
                self.calc_retained_earnings_to_equity()
            ),
            equity_to_revenue=(
                self.calc_equity_to_revenue()
            ),
            equity_to_assets=(
                self.calc_equity_to_assets()
            ),
            equity_res_to_equity=(
                self.calc_equity_res_to_equity()
            ),
        )

    # -- other -- 

    employee_to_revenue: Other_Norm = 0.

    # rev per employee
    def calc_employee_to_revenue(self):
        return self.calc_to_revenue("employees", inv=True)

    def calc_other(self):
        return self._replace(
            employee_to_revenue=(
                self.calc_employee_to_revenue()
            ),
        )

    #  -- 

    @classmethod
    def calc(cls, prev, curr):
        self = cls(prev, curr)
        return (
            self.calc_income_statement()
            .calc_cash_flow()
            .calc_balance_sheet()
            .calc_other()
        )

# ---------------------------------------------------------------

# given the above, then model as 
# random walk of revenue growth rate
# and walk of change in (independent) ratios

# with embeddings on eg. ^ as input data
# other relevant labels eg. industry
# and any market correlations

# ---------------------------------------------------------------
