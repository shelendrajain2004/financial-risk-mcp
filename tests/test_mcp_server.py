"""Unit tests for MCP Server protocol and handlers."""

import unittest
from fin_risk_mcp.server import FinancialRiskMCPServer


class TestFinancialRiskMCPServer(unittest.TestCase):

    def setUp(self):
        self.server = FinancialRiskMCPServer()

    def test_initialize(self):
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        }
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 1)
        self.assertEqual(res["result"]["serverInfo"]["name"], "financial-risk-mcp")
        self.assertIn("tools", res["result"]["capabilities"])

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        res = self.server.handle_request(req)
        tools = res["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("calculate_sacr_exposure", tool_names)
        self.assertIn("simulate_monte_carlo_pfe", tool_names)
        self.assertIn("compute_portfolio_var", tool_names)
        self.assertIn("calculate_portfolio_greeks", tool_names)

    def test_call_calculate_sacr_exposure(self):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "calculate_sacr_exposure",
                "arguments": {
                    "netting_set_id": "NS-001",
                    "counterparty_id": "CP-GOLDMAN",
                    "collateral_posted": 100_000.0,
                    "trades": [
                        {
                            "trade_id": "T1",
                            "asset_class": "INTEREST_RATE",
                            "notional": 20_000_000.0,
                            "market_value": 500_000.0,
                            "maturity_years": 5.0,
                        },
                        {
                            "trade_id": "T2",
                            "asset_class": "FOREIGN_EXCHANGE",
                            "notional": 10_000_000.0,
                            "market_value": -200_000.0,
                            "maturity_years": 1.0,
                        },
                    ],
                },
            },
        }
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 3)
        self.assertFalse(res["result"]["isError"])
        content_text = res["result"]["content"][0]["text"]
        self.assertIn("Exposure at Default (EAD)", content_text)
        self.assertIn("Replacement Cost (RC)", content_text)

    def test_call_simulate_monte_carlo_pfe(self):
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "simulate_monte_carlo_pfe",
                "arguments": {
                    "portfolio_id": "PORT-C++-TEST",
                    "num_simulations": 1000,
                    "trades": [
                        {
                            "trade_id": "T1",
                            "asset_class": "INTEREST_RATE",
                            "notional": 10_000_000.0,
                            "market_value": 200_000.0,
                            "maturity_years": 5.0,
                        }
                    ],
                },
            },
        }
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 4)
        self.assertFalse(res["result"]["isError"])
        content_text = res["result"]["content"][0]["text"]
        self.assertIn("Peak Forward Exposure (97.5%)", content_text)

    def test_resources_read(self):
        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": "financial://portfolio/citi-benchmark-01"},
        }
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 5)
        self.assertIn("trades", res["result"]["contents"][0]["text"])

    def test_prompts_get(self):
        req = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "prompts/get",
            "params": {
                "name": "audit_counterparty_risk",
                "arguments": {"netting_set_id": "NS-CITI-99"},
            },
        }
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 6)
        msg_text = res["result"]["messages"][0]["content"]["text"]
        self.assertIn("NS-CITI-99", msg_text)


if __name__ == "__main__":
    unittest.main()
