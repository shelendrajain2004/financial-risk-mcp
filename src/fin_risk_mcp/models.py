"""Data models for Financial Risk Quantitative Analytics MCP Server.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class AssetClass(str, Enum):
    INTEREST_RATE = "INTEREST_RATE"
    FOREIGN_EXCHANGE = "FOREIGN_EXCHANGE"
    CREDIT = "CREDIT"
    EQUITY = "EQUITY"
    COMMODITY = "COMMODITY"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    LINEAR = "LINEAR"  # Swaps, Forwards, Futures


@dataclass
class DerivativeTrade:
    trade_id: str
    asset_class: AssetClass
    notional: float
    market_value: float  # Current MTM (Mark-to-Market)
    maturity_years: float
    option_type: OptionType = OptionType.LINEAR
    strike: Optional[float] = None
    underlying_price: Optional[float] = None
    volatility: Optional[float] = None
    currency: str = "USD"
    supervisory_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "asset_class": self.asset_class.value,
            "notional": self.notional,
            "market_value": self.market_value,
            "maturity_years": self.maturity_years,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "underlying_price": self.underlying_price,
            "volatility": self.volatility,
            "currency": self.currency,
        }


@dataclass
class NettingSet:
    netting_set_id: str
    counterparty_id: str
    collateral_posted: float = 0.0  # Collateral posted by counterparty (C)
    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    is_margined: bool = False
    margin_period_of_risk_days: int = 10
    trades: List[DerivativeTrade] = field(default_factory=list)

    @property
    def net_market_value(self) -> float:
        return sum(t.market_value for t in self.trades)

    @property
    def gross_market_value(self) -> float:
        return sum(abs(t.market_value) for t in self.trades)


@dataclass
class SACCRResult:
    netting_set_id: str
    replacement_cost: float
    potential_future_exposure: float
    addon_aggregate: float
    multiplier: float
    ead: float  # Exposure at Default = 1.4 * (RC + PFE)
    asset_class_breakdown: Dict[str, float]
    alpha: float = 1.4


@dataclass
class PFEProfilePoint:
    tenor_years: float
    expected_exposure: float
    pfe_95: float
    pfe_97_5: float
    pfe_99: float


@dataclass
class MonteCarloPFEResult:
    portfolio_id: str
    num_simulations: int
    num_time_steps: int
    peak_pfe_97_5: float
    peak_tenor_years: float
    profile: List[PFEProfilePoint]
    computation_time_ms: float
    backend: str = "Python (NumPy Vectorized)"
