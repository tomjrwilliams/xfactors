
from __future__ import annotations

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

# ---------------------------------------------------------------

# general process:

# - load data
# - map to financials raw
# - normalise (-> financials norm)
# - model
# - forecast (-> financials norm)
# - map back to financials raw

# ---------------------------------------------------------------

if_none = xf.utils.funcs.if_none
if_none_lazy = xf.utils.funcs.if_none_lazy

# ---------------------------------------------------------------

V = typing.Union[float, numpy.ndarray, jax.numpy.ndarray]

IS_Independent = typing.TypeVar("IS_Independent", None, V)
IS_Dependent = typing.TypeVar("IS_Dependent", None, V)

CF_Independent = typing.TypeVar("CF_Independent", None, V)
CF_Dependent = typing.TypeVar("CF_Dependent", None, V)

BS_Independent = typing.TypeVar("BS_Independent", None, V)
BS_Dependent = typing.TypeVar("BS_Dependent", None, V)

Other_Independent = typing.TypeVar("Other_Independent", None, V)
Other_Dependent = typing.TypeVar("Other_Dependent", None, V)

# ---------------------------------------------------------------

# TODO: and sub methods for each statement separately
# by type name inspection

# TODO: function to return tuple indices
# with an lru_cache (so only calc once)
# for optimal access time

# TODO: below take an optional stmt kwarg

def fields_independent(cls):
    return

def fields_dependent(cls):
    return

def fields_income_statement(cls):
    return

def fields_independent_income_statement(cls):
    return

def fields_dependent_income_statement(cls):
    return

def fields_cash_flow(cls):
    return

def fields_independent_cash_flow(cls):
    return

def fields_dependent_cash_flow(cls):
    return

def fields_balance_sheet(cls):
    return

def fields_independent_balance_sheet(cls):
    return

def fields_dependent_balance_sheet(cls):
    return

def fields_other(cls):
    return

def fields_independent_other(cls):
    return

def fields_dependent_other(cls):
    return

# ---------------------------------------------------------------

def values_independent(self):
    return

def values_dependent(self):
    return

def values_income_statement(cls):
    return

def values_independent_income_statement(cls):
    return

def values_dependent_income_statement(cls):
    return

def values_cash_flow(cls):
    return

def values_independent_cash_flow(cls):
    return

def values_dependent_cash_flow(cls):
    return

def values_balance_sheet(cls):
    return

def values_independent_balance_sheet(cls):
    return

def values_dependent_balance_sheet(cls):
    return

def values_other(cls):
    return

def values_independent_other(cls):
    return

def values_dependent_other(cls):
    return

# TODO: given a field name map for a given source
# can assert all fields per statement are mapped
# or explicitly specified that they shouldn't be

# ---------------------------------------------------------------

@xt.nTuple.decorate()
class Statements(typing.NamedTuple):

    # income statement

    revenue: IS_Independent = 0.
    cogs: IS_Independent = 0.
    profit_gross: IS_Dependent = None
    profit_gross_res: IS_Dependent = None

    def calc_profit_gross(self):
        return self.revenue - self.cogs

    def calc_profit_gross_res(self):
        return self.profit_gross - self.calc_profit_gross()

    def check_profit_gross(self, tolerance: float):
        return jax.numpy.abs(self.profit_gross_res) <= tolerance

    # --

    rd: IS_Independent = 0.
    sga: IS_Independent = 0.
    profit_operating: IS_Dependent = None # ebit
    profit_operating_res: IS_Dependent = None

    def calc_profit_operating(self, profit_gross = None):
        profit_gross = if_none(profit_gross, self.profit_gross)
        assert profit_gross is not None
        return profit_gross - (self.rd + self.sga)
    
    def calc_profit_operating_res(self):
        return self.profit_operating - self.calc_profit_operating()

    def check_profit_operating(self, tolerance: float):
        return jax.numpy.abs(self.profit_operating_res) <= tolerance

    # --

    interest_in: IS_Independent = 0.
    interest_out: IS_Independent = 0.

    expense_other: IS_Independent = 0.

    profit_pretax: IS_Dependent = None
    profit_pretax_res: IS_Dependent = None

    def calc_profit_pretax(self, profit_operating = None):
        profit_operating = if_none(profit_operating, self.profit_operating)
        assert profit_operating is not None
        return (
            profit_operating + (
                self.interest_in
                - self.interest_out
                - self.expense_other
            )
        )

    def calc_profit_pretax_res(self):
        return self.profit_prefx - self.calc_profit_pretax()

    def check_profit_pretax(self, tolerance: float):
        return jax.numpy.abs(self.profit_pretax_res) <= tolerance

    # --

    tax: IS_Independent = 0.
    profit_net: IS_Dependent = None
    profit_net_res: IS_Dependent = None

    def calc_profit_net(self, profit_pretax = None):
        profit_pretax = if_none(profit_pretax, self.profit_pretax)
        assert profit_pretax is not None
        return profit_pretax - self.tax
    
    def calc_profit_net_res(self):
        return self.profit_net - self.calc_profit_net()
        
    def check_profit_net(self, tolerance):
        return jax.numpy.abs(self.profit_net_res) <= tolerance

    @classmethod
    def income_statement(
        cls,
        revenue: IS_Independent = 0,
        cogs: IS_Independent = 0,
        profit_gross: IS_Dependent = None,
        # profit_gross_res
        rd: IS_Independent = 0,
        sga: IS_Independent = 0,
        profit_operating: IS_Dependent = None,
        # profit_operating_res
        interest_in: IS_Independent = 0,
        interest_out: IS_Independent = 0,
        expense_other: IS_Independent = 0,
        profit_pretax: IS_Dependent = None,
        # profit_pretax_res
        tax: IS_Independent = 0,
        profit_net: IS_Dependent = None,
        # profit_net_res
    ):
        self = cls(
            revenue=revenue,
            cogs=cogs,
            rd=rd,
            sga=sga,
            interest_in=interest_in,
            interest_out=interest_out,
            expense_other=expense_other,
            tax=tax,
        )
        # ^ minimum requirements to generate income statement
        profit_gross=if_none_lazy(
            profit_gross, self.calc_profit_gross
        )
        profit_operating=if_none_lazy(
            profit_operating, 
            lambda: self.calc_profit_operating(profit_gross)
        )
        profit_pretax = if_none_lazy(
            profit_pretax,
            lambda: self.calc_profit_pretax(profit_operating)
        )
        profit_net = if_none_lazy(
            profit_pretax,
            lambda: self.calc_profit_net(profit_pretax)
        )
        # ^ optionally provide if known
        self = self._replace(
            profit_gross=profit_gross,
            profit_operating=profit_operating,
            profit_pretax=profit_pretax,
            profit_net=profit_net,
        )
        self = self._replace(
            profit_gross_res = self.calc_profit_gross_res(),
            profit_operating_res = self.calc_profit_operating_res(),
            profit_pretax_res = self.calc_profit_pretax_res(),
            profit_net_res = self.calc_profit_net_res(),
        )
        # ^ will be zero if no profits directly provided
        return self

    # -- cash flow --

    da: CF_Independent = 0.
    # stock based comp : CF_Independent = 0.
    working_cap_change: CF_Independent = 0.
    cash_other: CF_Independent = 0.

    cash_operating: CF_Dependent = None
    cash_operating_res: CF_Dependent = None

    def calc_cash_operating(self, profit_net = None):
        profit_net = if_none(profit_net, self.profit_net)
        assert profit_net is not None, profit_net
        return profit_net + (
            self.da
            - self.working_cap_change
            # working cap = working assets - liab
            # so increase means more assets, so less cash
            + self.cash_other
        )
    
    def calc_cash_operating_res(self):
        return self.cash_operating - self.calc_cash_operating()

    def check_cash_operating(self, tolerance):
        return jax.numpy.abs(self.cash_operating_res) <= tolerance

    # --

    capex: CF_Independent = 0.
    cash_intangibles: CF_Independent = 0.
    
    cash_investing: CF_Dependent = None
    cash_investing_res: CF_Dependent = None

    def calc_cash_investing(self):
        return (
            - self.capex
            - self.cash_intangibles
        )

    def calc_cash_investing_res(self):
        return self.cash_investing - self.calc_cash_investing()

    def check_cash_investing(self, tolerance):
        return jax.numpy.abs(self.cash_investing_res) <= tolerance

    # --

    cash_debt: CF_Independent = 0.
    cash_equity: CF_Independent = 0.
    # - dividends
    # + change share issuance
    # - share repurchase

    # cash_equity_res: CF_Independent = 0. ?

    cash_financing: CF_Dependent = None#
    cash_financing_res: CF_Dependent = None

    def calc_cash_financing(self):
        return self.cash_debt + self.cash_equity

    def calc_cash_financing_res(self):
        return self.cash_financing - self.calc_cash_financing()

    def check_cash_financing(self, tolerance):
        return jax.numpy.abs(self.cash_financing_res) <= tolerance

    # --
    
    cash_net: CF_Dependent = None
    cash_net_res: CF_Dependent  = None

    def calc_cash_net(
        self,
        cash_operating = None,
        cash_investing = None,
        cash_financing = None,
    ):  
        cash_operating = if_none(cash_operating, self.cash_operating)
        assert cash_operating is not None, cash_operating
        
        cash_investing = if_none(cash_investing, self.cash_investing)
        assert cash_investing is not None, cash_investing
        
        cash_financing = if_none(cash_financing, self.cash_financing)
        assert cash_financing is not None, cash_financing

        return (
            cash_operating + cash_investing + cash_financing
        )

    def calc_cash_net_res(self):
        return self.cash_net - self.calc_cash_net()

    def check_cash_net(self, tolerance):
        return jax.numpy.abs(self.cash_net_res) <= tolerance

    @classmethod
    def cash_flow(
        cls,
        profit_net: CF_Dependent = None,
        da: CF_Independent = 0,
        working_cap_change: CF_Independent = 0,
        cash_other: CF_Independent = 0,
        cash_operating: CF_Dependent = None,
        # cash_operating_res
        capex: CF_Independent = 0,
        cash_intangibles: CF_Independent = 0,
        cash_investing: CF_Dependent = None,
        # cash_investing_res
        cash_debt: CF_Independent = 0,
        cash_equity: CF_Independent = 0,
        cash_financing: CF_Dependent = None,
        # cash_financing_res
        cash_net: CF_Dependent = None,
        # cash_net_res,
        self=None,
    ):
        if self is None:
            self = cls()
        self = self._replace(
            da=da,
            working_cap_change=working_cap_change,
            cash_other=cash_other,
            capex=capex,
            cash_intangibles=cash_intangibles,
            cash_debt=cash_debt,
            cash_equity=cash_equity,
        )
        cash_operating=if_none_lazy(
            cash_operating,
            lambda: self.calc_cash_operating(profit_net)
        )
        cash_investing=if_none_lazy(
            cash_investing, 
            self.calc_cash_investing,
        )
        cash_financing = if_none_lazy(
            cash_financing,
            self.calc_cash_financing,
        )
        cash_net = if_none_lazy(
            cash_net,
            lambda: self.calc_cash_net(
                cash_operating,
                cash_investing,
                cash_financing,
            )
        )
        self = self._replace(
            cash_operating=self.cash_operating,
            cash_investing=self.cash_investing,
            cash_financing=self.cash_financing,
            cash_net=self.cash_net,
        )
        self = self._replace(
            cash_operating_res=self.calc_cash_operating_res(),
            cash_investing_res=self.calc_cash_investing_res(),
            cash_financing_res=self.calc_cash_financing_res(),
            cash_net_res=self.calc_cash_net_res(),
        )
        return self

    # -- balance sheet --

    # --

    # keep separate working capital items
    # rather than just net
    # as gross is useful for industry embedding kernel

    cash: BS_Independent = 0.
    securities: BS_Independent = 0.

    acc_receivable: BS_Independent = 0.
    inventory: BS_Independent = 0.
    tax_deferred: BS_Independent = 0.
    assets_other: BS_Independent = 0.
    # current incl non trade receivable, also non current below?
    ppe: BS_Independent = 0.
    intangibles: BS_Independent = 0.
    
    assets: BS_Dependent = None
    assets_res: BS_Dependent = None

    def calc_assets(self):
        return (
            0
            + self.cash
            + self.securities
            + self.acc_receivable
            + self.inventory
            + self.tax_deferred
            + self.assets_other
            + self.ppe
            + self.intangibles
        )

    def calc_assets_res(self):
        return self.assets - self.calc_assets()
    
    def check_assets(self, tolerance):
        return jax.numpy.abs(self.assets_res) <= tolerance

    # --

    acc_payable: BS_Independent = 0.
    accrued_expenses: BS_Independent = 0.
    revenue_deferred: BS_Independent = 0.
    debt: BS_Independent = 0.
    liabilities_other: BS_Independent = 0.

    liabilities: BS_Dependent = None
    liabilities_res: BS_Dependent = None

    def calc_liabilities(self):
        return (
            0
            + self.acc_payable
            + self.accrued_expenses
            + self.revenue_deferred
            + self.debt
            + self.liabilities_other
        )

    def calc_liabilities_res(self):
        return self.liabilities - self.calc_liabilities()
    
    def check_liabilities(self, tolerance):
        return jax.numpy.abs(self.liabilities_res) <= tolerance
        
    # --

    working_cap_gross: BS_Independent = 0.

    working_cap_net: BS_Independent = 0.
    # change in -> cash flow

    # net change ppe = da - capex (?)

    # --

    equity_common: BS_Independent = 0.
    equity_treasury: BS_Independent = 0.
    retained_earnings: BS_Independent = 0.

    equity: BS_Dependent = None
    equity_res: BS_Dependent = None

    def calc_equity(self):
        return (
            0
            + self.equity_common
            + self.equity_treasury
            + self.retained_earnings
        )

    def calc_equity_res(self):
        return self.equity - self.calc_equity()
    
    def check_equity(self, tolerance):
        return jax.numpy.abs(self.equity_res) <= tolerance
        
    # --

    # versus the below, the balance sheet should be calc-able
    # from prev(balance_sheet) + current(income_stmt, cash_flow)

    @classmethod
    def balance_sheet(
        cls,
        cash: BS_Independent = 0.,
        securities: BS_Independent = 0.,
        acc_receivable: BS_Independent = 0.,
        inventory: BS_Independent = 0.,
        tax_deferred: BS_Independent = 0.,
        assets_other: BS_Independent = 0.,
        ppe: BS_Independent = 0.,
        intangibles: BS_Independent = 0.,
        assets: BS_Dependent = None,
        # assets_res: BS_Dependent = None,
        acc_payable: BS_Independent = 0.,
        accrued_expenses: BS_Independent = 0.,
        revenue_deferred: BS_Independent = 0.,
        debt: BS_Independent = 0.,
        liabilities_other: BS_Independent = 0.,
        liabilities: BS_Dependent = None,
        # liabilities_res: BS_Dependent = None,
        equity_common: BS_Independent = 0.,
        equity_treasury: BS_Independent = 0.,
        retained_earnings: BS_Independent = 0.,
        equity: BS_Dependent = None,
        # equity_res: BS_Dependent = None,
    ):
        self = cls(
            cash=cash,
            securities=securities,
            acc_receivable=acc_receivable,
            inventory=inventory,
            tax_deferred=tax_deferred,
            assets_other=assets_other,
            ppe=ppe,
            intangibles=intangibles,
            #
            acc_payable=acc_payable,
            accrued_expenses=accrued_expenses,
            revenue_deferred=revenue_deferred,
            debt=debt,
            liabilities_other=liabilities_other,
            #
            equity_common=equity_common,
            equity_treasury=equity_treasury,
            retained_earnings=retained_earnings,
        )
        assets = if_none_lazy(assets, self.calc_assets)
        liabilities = if_none_lazy(liabilities, self.calc_liabilities)
        equity = if_none_lazy(equity, self.calc_equity)
        self = self._replace(
            assets=assets,
            liabilities=liabilities,
            equity=equity,
        )
        self = self._replace(
            assets_res=self.calc_assets_res(),
            liabilities_res=self.calc_liabilities_res(),
            equity_res=self.calc_equity_res(),
        )
        # assets = liabilities + equity
        return self

    # -- other --
    
    employees: Other_Independent = None

    # --

# ---------------------------------------------------------------
