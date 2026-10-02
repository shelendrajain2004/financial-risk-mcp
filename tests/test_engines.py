"""Unit tests for quantitative risk engines."""

import unittest
from fin_risk_mcp.models import AssetClass, DerivativeTrade, NettingSet, OptionType
from fin_risk_mcp.engines import SACCREngine, MonteCarloPFEEngine, VaREngine, GreeksEngine


class TestRiskEngines(unittest.TestCase):

    def setUp(self):
        self.ir_trade = DerivativeTrade(
            trade_id="IR-SWAP-1",
            asset_class=AssetClass.INTEREST_RATE,
            notional=10_000_000.0,
            market_value=250_000.0,
            maturity_years=5.0,
        )
        self.fx_trade = DerivativeTrade(
            trade_id="FX-FWD-1",
            asset_class=AssetClass.FOREIGN_EXCHANGE,
            notional=5_000_000.0,
            market_value=100_000.0,
            maturity_years=1.0,
        )
        self.opt_trade = DerivativeTrade(
            trade_id="EQ-OPT-1",
            asset_class=AssetClass.EQUITY,
            notional=1_000_000.0,
            market_value=50_000.0,
            maturity_years=0.5,
            option_type=OptionType.CALL,
            strike=100.0,
            underlying_price=105.0,
            volatility=0.20,
        )

    def test_saccr_unmargined(self):
        ns = NettingSet(
            netting_set_id="NS-TEST-01",
            counterparty_id="CP-TEST",
            collateral_posted=50_000.0,
            is_margined=False,
            trades=[self.ir_trade, self.fx_trade],
        )
        res = SACCREngine.calculate(ns)

        # Net MTM = 350,000, C = 50,000 -> RC = 300,000
        self.assertAlmostEqual(res.replacement_cost, 300_000.0, places=1)
        self.assertGreater(res.addon_aggregate, 0.0)
        self.assertGreater(res.ead, res.replacement_cost)
        self.assertEqual(res.alpha, 1.4)
        self.assertIn(AssetClass.INTEREST_RATE.value, res.asset_class_breakdown)
        self.assertIn(AssetClass.FOREIGN_EXCHANGE.value, res.asset_class_breakdown)

    def test_saccr_margined(self):
        ns = NettingSet(
            netting_set_id="NS-TEST-02",
            counterparty_id="CP-TEST",
            collateral_posted=400_000.0,
            is_margined=True,
            threshold=50_000.0,
            minimum_transfer_amount=25_000.0,
            trades=[self.ir_trade],
        )
        res = SACCREngine.calculate(ns)
        # Margined RC should be bound by threshold + MTA
        self.assertGreaterEqual(res.replacement_cost, 75_000.0)

    def test_monte_carlo_pfe(self):
        res = MonteCarloPFEEngine.simulate(
            portfolio_id="PORT-TEST-01",
            trades=[self.ir_trade, self.fx_trade],
            num_simulations=5000,
        )
        self.assertEqual(res.portfolio_id, "PORT-TEST-01")
        self.assertEqual(res.num_simulations, 5000)
        self.assertGreater(len(res.profile), 0)
        self.assertGreater(res.peak_pfe_97_5, 0.0)
        self.assertGreater(res.peak_tenor_years, 0.0)

        # Check quantile ordering: EE <= PFE95 <= PFE97.5 <= PFE99
        for pt in res.profile:
            self.assertLessEqual(pt.expected_exposure, pt.pfe_95 + 1.0)
            self.assertLessEqual(pt.pfe_95, pt.pfe_97_5 + 1.0)
            self.assertLessEqual(pt.pfe_97_5, pt.pfe_99 + 1.0)

    def test_parametric_var(self):
        var_res = VaREngine.calculate_parametric_var(
            portfolio_value=10_000_000.0,
            daily_volatility=0.015,
            horizon_days=10,
            confidence_level=0.99,
        )
        # Expected: 10M * 2.32635 * 0.015 * sqrt(10) ≈ 1,103,485
        self.assertAlmostEqual(var_res["var"], 1_103_485.0, delta=2000.0)
        self.assertGreater(var_res["expected_shortfall_cvar"], var_res["var"])

    def test_greeks_calculation(self):
        greeks = GreeksEngine.calculate_position_greeks(self.opt_trade)
        # For call option with S=105, K=100, delta must be positive between 0 and Notional
        self.assertGreater(greeks["delta"], 0.0)
        self.assertLessEqual(greeks["delta"], self.opt_trade.notional)
        self.assertGreater(greeks["gamma"], 0.0)
        self.assertGreater(greeks["vega"], 0.0)


if __name__ == "__main__":
    unittest.main()
