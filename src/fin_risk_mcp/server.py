"""Model Context Protocol (MCP) Server for Quantitative Financial Risk.
Implements MCP 2024-11-05 JSON-RPC 2.0 specification over standard I/O (stdio).
Zero external runtime dependencies beyond standard Python 3.10+ and NumPy.
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional

from .models import AssetClass, DerivativeTrade, NettingSet, OptionType
from .engines import SACCREngine, MonteCarloPFEEngine, VaREngine, GreeksEngine

# Configure logging to stderr so stdout remains strictly reserved for MCP JSON-RPC
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fin_risk_mcp")


# In-memory mock regulatory & portfolio benchmark data
BENCHMARK_PORTFOLIO = [
    DerivativeTrade(
        trade_id="IR-SWAP-001",
        asset_class=AssetClass.INTEREST_RATE,
        notional=50_000_000.0,
        market_value=1_250_000.0,
        maturity_years=5.0,
        option_type=OptionType.LINEAR,
    ),
    DerivativeTrade(
        trade_id="IR-SWAP-002",
        asset_class=AssetClass.INTEREST_RATE,
        notional=-30_000_000.0,
        market_value=-850_000.0,
        maturity_years=10.0,
        option_type=OptionType.LINEAR,
    ),
    DerivativeTrade(
        trade_id="FX-FWD-EURUSD",
        asset_class=AssetClass.FOREIGN_EXCHANGE,
        notional=25_000_000.0,
        market_value=420_000.0,
        maturity_years=1.5,
        option_type=OptionType.LINEAR,
    ),
    DerivativeTrade(
        trade_id="EQ-OPT-SPX-CALL",
        asset_class=AssetClass.EQUITY,
        notional=10_000_000.0,
        market_value=350_000.0,
        maturity_years=0.75,
        option_type=OptionType.CALL,
        strike=5000.0,
        underlying_price=5150.0,
        volatility=0.18,
    ),
]


class FinancialRiskMCPServer:
    """Production-grade MCP Server providing quantitative risk calculations to AI agents."""

    SERVER_NAME = "financial-risk-mcp"
    SERVER_VERSION = "1.0.0"
    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self):
        self.tools_schema = self._build_tools_schema()
        self.resources_schema = self._build_resources_schema()
        self.prompts_schema = self._build_prompts_schema()

    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculate_sacr_exposure",
                "description": (
                    "Calculates Basel III / BCBS 279 Standardized Approach for Counterparty "
                    "Credit Risk (SA-CCR) metrics: Replacement Cost (RC), Potential Future "
                    "Exposure (PFE), Supervisory Add-on, Multiplier, and Exposure at Default (EAD = 1.4 * (RC + PFE))."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "netting_set_id": {
                            "type": "string",
                            "description": "Unique netting agreement identifier (e.g., 'NS-CITI-001')",
                        },
                        "counterparty_id": {
                            "type": "string",
                            "description": "Counterparty legal entity identifier or name",
                        },
                        "trades": {
                            "type": "array",
                            "description": "List of derivative trades in the netting set",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "trade_id": {"type": "string"},
                                    "asset_class": {
                                        "type": "string",
                                        "enum": [ac.value for ac in AssetClass],
                                    },
                                    "notional": {"type": "number"},
                                    "market_value": {"type": "number"},
                                    "maturity_years": {"type": "number"},
                                    "option_type": {
                                        "type": "string",
                                        "enum": ["CALL", "PUT", "LINEAR"],
                                        "default": "LINEAR",
                                    },
                                    "strike": {"type": "number"},
                                    "underlying_price": {"type": "number"},
                                    "volatility": {"type": "number"},
                                },
                                "required": ["trade_id", "asset_class", "notional", "market_value", "maturity_years"],
                            },
                        },
                        "collateral_posted": {
                            "type": "number",
                            "description": "Total eligible collateral held (C) in USD",
                            "default": 0.0,
                        },
                        "is_margined": {
                            "type": "boolean",
                            "description": "Whether the netting set is subject to bilateral margin / CSA agreement",
                            "default": False,
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Threshold amount under CSA",
                            "default": 0.0,
                        },
                        "minimum_transfer_amount": {
                            "type": "number",
                            "description": "Minimum Transfer Amount (MTA) under CSA",
                            "default": 0.0,
                        },
                    },
                    "required": ["netting_set_id", "counterparty_id", "trades"],
                },
            },
            {
                "name": "simulate_monte_carlo_pfe",
                "description": (
                    "Executes a vectorized Monte Carlo simulation to project Potential Future "
                    "Exposure (PFE) profiles and Peak Forward Exposure across multi-year tenors (0.25y to 30y)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                        "trades": {
                            "type": "array",
                            "description": "List of trades to simulate exposure for",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "trade_id": {"type": "string"},
                                    "asset_class": {"type": "string"},
                                    "notional": {"type": "number"},
                                    "market_value": {"type": "number"},
                                    "maturity_years": {"type": "number"},
                                },
                                "required": ["trade_id", "notional", "market_value", "maturity_years"],
                            },
                        },
                        "num_simulations": {
                            "type": "integer",
                            "description": "Number of Monte Carlo paths (default 10,000)",
                            "default": 10000,
                        },
                        "volatility": {
                            "type": "number",
                            "description": "Annualized market volatility factor",
                            "default": 0.15,
                        },
                        "mean_reversion": {
                            "type": "number",
                            "description": "Mean reversion speed for rate/asset diffusion",
                            "default": 0.03,
                        },
                    },
                    "required": ["portfolio_id", "trades"],
                },
            },
            {
                "name": "compute_portfolio_var",
                "description": (
                    "Computes regulatory Value-at-Risk (VaR) and Expected Shortfall (CVaR) "
                    "for market risk across specified holding periods and confidence intervals."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "portfolio_value": {"type": "number", "description": "Total market value of the portfolio in USD"},
                        "daily_volatility": {"type": "number", "description": "1-day standard deviation of portfolio returns"},
                        "horizon_days": {
                            "type": "integer",
                            "description": "Holding period in days (e.g. 10 for Basel Market Risk)",
                            "default": 10,
                        },
                        "confidence_level": {
                            "type": "number",
                            "description": "Statistical confidence level (e.g. 0.99 for 99%)",
                            "default": 0.99,
                        },
                    },
                    "required": ["portfolio_value", "daily_volatility"],
                },
            },
            {
                "name": "calculate_portfolio_greeks",
                "description": (
                    "Aggregates first- and second-order derivatives sensitivities: Delta, "
                    "Gamma, Vega, Theta, and Rho across a book of positions."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "trades": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "trade_id": {"type": "string"},
                                    "notional": {"type": "number"},
                                    "option_type": {"type": "string", "enum": ["CALL", "PUT", "LINEAR"]},
                                    "strike": {"type": "number"},
                                    "underlying_price": {"type": "number"},
                                    "maturity_years": {"type": "number"},
                                    "volatility": {"type": "number"},
                                },
                                "required": ["trade_id", "notional", "maturity_years"],
                            },
                        }
                    },
                    "required": ["trades"],
                },
            },
        ]

    def _build_resources_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "uri": "financial://portfolio/citi-benchmark-01",
                "name": "Benchmark Multi-Asset Derivative Portfolio",
                "description": "Standard institutional portfolio containing Rates, FX, and Equity options for risk benchmarking",
                "mimeType": "application/json",
            },
            {
                "uri": "financial://regulatory/bcbs279-factors",
                "name": "BCBS 279 Supervisory Risk Factors Table",
                "description": "Basel III supervisory factors, duration formulas, and correlations across asset classes",
                "mimeType": "application/json",
            },
        ]

    def _build_prompts_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "audit_counterparty_risk",
                "description": "Prompts the LLM agent to conduct an audit of counterparty credit risk and SA-CCR capital adequacy.",
                "arguments": [
                    {
                        "name": "netting_set_id",
                        "description": "Netting set agreement identifier to audit",
                        "required": True,
                    }
                ],
            },
            {
                "name": "stress_test_scenario",
                "description": "Prompts the LLM agent to evaluate portfolio vulnerability under rate shocks and volatility spikes.",
                "arguments": [
                    {
                        "name": "volatility_shock_pct",
                        "description": "Relative percentage increase in market volatility (e.g. 50%)",
                        "required": False,
                    }
                ],
            },
        ]

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes incoming JSON-RPC 2.0 requests."""
        method = request.get("method")
        msg_id = request.get("id")
        params = request.get("params", {})

        logger.info(f"Received request: method={method}, id={msg_id}")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": self.SERVER_NAME,
                        "version": self.SERVER_VERSION,
                    },
                },
            }

        elif method == "notifications/initialized":
            logger.info("Client handshake complete.")
            return None

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.tools_schema},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            return self._handle_tool_call(msg_id, tool_name, arguments)

        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"resources": self.resources_schema},
            }

        elif method == "resources/read":
            uri = params.get("uri")
            return self._handle_resource_read(msg_id, uri)

        elif method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"prompts": self.prompts_schema},
            }

        elif method == "prompts/get":
            prompt_name = params.get("name")
            prompt_args = params.get("arguments", {})
            return self._handle_prompt_get(msg_id, prompt_name, prompt_args)

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found",
                },
            }

    def _handle_tool_call(self, msg_id: Any, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "calculate_sacr_exposure":
                trades_data = arguments.get("trades", [])
                trades = [
                    DerivativeTrade(
                        trade_id=t["trade_id"],
                        asset_class=AssetClass(t["asset_class"]),
                        notional=float(t["notional"]),
                        market_value=float(t["market_value"]),
                        maturity_years=float(t["maturity_years"]),
                        option_type=OptionType(t.get("option_type", "LINEAR")),
                        strike=float(t["strike"]) if t.get("strike") is not None else None,
                        underlying_price=float(t["underlying_price"]) if t.get("underlying_price") is not None else None,
                        volatility=float(t["volatility"]) if t.get("volatility") is not None else None,
                    )
                    for t in trades_data
                ]

                netting_set = NettingSet(
                    netting_set_id=arguments["netting_set_id"],
                    counterparty_id=arguments["counterparty_id"],
                    collateral_posted=float(arguments.get("collateral_posted", 0.0)),
                    threshold=float(arguments.get("threshold", 0.0)),
                    minimum_transfer_amount=float(arguments.get("minimum_transfer_amount", 0.0)),
                    is_margined=bool(arguments.get("is_margined", False)),
                    trades=trades,
                )

                result = SACCREngine.calculate(netting_set)
                payload = {
                    "netting_set_id": result.netting_set_id,
                    "replacement_cost_usd": result.replacement_cost,
                    "potential_future_exposure_usd": result.potential_future_exposure,
                    "addon_aggregate_usd": result.addon_aggregate,
                    "multiplier": result.multiplier,
                    "exposure_at_default_ead_usd": result.ead,
                    "alpha_regulatory_factor": result.alpha,
                    "asset_class_breakdown": result.asset_class_breakdown,
                }
                text_summary = (
                    f"### SA-CCR Exposure Report: {result.netting_set_id}\n\n"
                    f"* **Exposure at Default (EAD):** ${result.ead:,.2f} USD\n"
                    f"* **Replacement Cost (RC):** ${result.replacement_cost:,.2f} USD\n"
                    f"* **Potential Future Exposure (PFE):** ${result.potential_future_exposure:,.2f} USD\n"
                    f"* **Regulatory Multiplier:** {result.multiplier:.4f}\n"
                    f"* **Supervisory AddOn Aggregate:** ${result.addon_aggregate:,.2f} USD\n"
                    f"* **Alpha Factor:** {result.alpha}\n\n"
                    f"```json\n{json.dumps(payload, indent=2)}\n```"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": text_summary}],
                        "isError": False,
                    },
                }

            elif tool_name == "simulate_monte_carlo_pfe":
                trades_data = arguments.get("trades", [])
                trades = [
                    DerivativeTrade(
                        trade_id=t["trade_id"],
                        asset_class=AssetClass.INTEREST_RATE,
                        notional=float(t["notional"]),
                        market_value=float(t["market_value"]),
                        maturity_years=float(t["maturity_years"]),
                    )
                    for t in trades_data
                ]

                pfe_res = MonteCarloPFEEngine.simulate(
                    portfolio_id=arguments["portfolio_id"],
                    trades=trades,
                    num_simulations=int(arguments.get("num_simulations", 10000)),
                    volatility=float(arguments.get("volatility", 0.15)),
                    mean_reversion=float(arguments.get("mean_reversion", 0.03)),
                )

                profile_table = "| Tenor (Yrs) | Expected Exposure (EE) | PFE 95% | PFE 97.5% | PFE 99% |\n|---|---|---|---|---|\n"
                for pt in pfe_res.profile:
                    profile_table += f"| {pt.tenor_years:5.2f}y | ${pt.expected_exposure:,.2f} | ${pt.pfe_95:,.2f} | ${pt.pfe_97_5:,.2f} | ${pt.pfe_99:,.2f} |\n"

                summary = (
                    f"### Monte Carlo PFE Profile: {pfe_res.portfolio_id}\n\n"
                    f"* **Simulations:** {pfe_res.num_simulations:,} paths\n"
                    f"* **Peak Forward Exposure (97.5%):** ${pfe_res.peak_pfe_97_5:,.2f} USD at **{pfe_res.peak_tenor_years:.2f} years**\n"
                    f"* **Execution Runtime:** {pfe_res.computation_time_ms:.2f} ms\n\n"
                    f"{profile_table}"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": summary}],
                        "isError": False,
                    },
                }

            elif tool_name == "compute_portfolio_var":
                var_res = VaREngine.calculate_parametric_var(
                    portfolio_value=float(arguments["portfolio_value"]),
                    daily_volatility=float(arguments["daily_volatility"]),
                    horizon_days=int(arguments.get("horizon_days", 10)),
                    confidence_level=float(arguments.get("confidence_level", 0.99)),
                )
                summary = (
                    f"### Portfolio Value-at-Risk (VaR) Analysis\n\n"
                    f"* **Portfolio Market Value:** ${var_res['portfolio_value']:,.2f}\n"
                    f"* **Holding Horizon:** {var_res['horizon_days']} days\n"
                    f"* **Confidence Level:** {var_res['confidence_level']*100:.1f}%\n"
                    f"* **Value at Risk (VaR):** **${var_res['var']:,.2f} USD** ({var_res['var_pct']} %)\n"
                    f"* **Expected Shortfall (CVaR):** **${var_res['expected_shortfall_cvar']:,.2f} USD**"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": summary}],
                        "isError": False,
                    },
                }

            elif tool_name == "calculate_portfolio_greeks":
                trades_data = arguments.get("trades", [])
                greeks_totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

                breakdown_table = "| Trade ID | Delta | Gamma | Vega | Theta | Rho |\n|---|---|---|---|---|---|\n"
                for t in trades_data:
                    trade = DerivativeTrade(
                        trade_id=t["trade_id"],
                        asset_class=AssetClass.EQUITY,
                        notional=float(t["notional"]),
                        market_value=0.0,
                        maturity_years=float(t["maturity_years"]),
                        option_type=OptionType(t.get("option_type", "LINEAR")),
                        strike=float(t["strike"]) if t.get("strike") else None,
                        underlying_price=float(t["underlying_price"]) if t.get("underlying_price") else None,
                        volatility=float(t["volatility"]) if t.get("volatility") else None,
                    )
                    g = GreeksEngine.calculate_position_greeks(trade)
                    for k in greeks_totals:
                        greeks_totals[k] += g[k]
                    breakdown_table += f"| {t['trade_id']} | {g['delta']:,.2f} | {g['gamma']:.4f} | {g['vega']:,.2f} | {g['theta']:,.2f} | {g['rho']:,.2f} |\n"

                summary = (
                    f"### Portfolio Risk Sensitivities (Greeks)\n\n"
                    f"* **Aggregate Delta:** {greeks_totals['delta']:,.2f}\n"
                    f"* **Aggregate Gamma:** {greeks_totals['gamma']:,.4f}\n"
                    f"* **Aggregate Vega:** ${greeks_totals['vega']:,.2f}\n"
                    f"* **Aggregate Theta:** ${greeks_totals['theta']:,.2f}/day\n"
                    f"* **Aggregate Rho:** ${greeks_totals['rho']:,.2f}\n\n"
                    f"{breakdown_table}"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": summary}],
                        "isError": False,
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

        except Exception as e:
            logger.exception(f"Error handling tool call {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing {tool_name}: {str(e)}"}],
                    "isError": True,
                },
            }

    def _handle_resource_read(self, msg_id: Any, uri: str) -> Dict[str, Any]:
        if uri == "financial://portfolio/citi-benchmark-01":
            content = {
                "portfolio_id": "citi-benchmark-01",
                "description": "Multi-Asset Derivatives Benchmark",
                "trades": [t.to_dict() for t in BENCHMARK_PORTFOLIO],
            }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(content, indent=2),
                        }
                    ]
                },
            }
        elif uri == "financial://regulatory/bcbs279-factors":
            factors = {
                "standard": "BCBS 279 Basel III / Basel IV",
                "alpha_multiplier": 1.4,
                "supervisory_factors": {
                    "INTEREST_RATE": 0.005,
                    "FOREIGN_EXCHANGE": 0.040,
                    "CREDIT_IG": 0.005,
                    "CREDIT_HY": 0.014,
                    "EQUITY_LARGE_CAP": 0.320,
                    "COMMODITY": 0.180,
                },
                "supervisory_duration_formula": "[exp(-0.05 * S) - exp(-0.05 * E)] / 0.05",
            }
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(factors, indent=2),
                        }
                    ]
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": f"Resource not found: {uri}"},
            }

    def _handle_prompt_get(self, msg_id: Any, prompt_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if prompt_name == "audit_counterparty_risk":
            ns_id = args.get("netting_set_id", "NS-UNKNOWN")
            prompt_text = (
                f"You are an Institutional Senior Quantitative Risk Auditor. Conduct a comprehensive "
                f"SA-CCR regulatory audit for Netting Set '{ns_id}'. Use the 'calculate_sacr_exposure' "
                f"tool to compute EAD, Replacement Cost, and Potential Future Exposure. Evaluate "
                f"whether the counterparty collateral posted provides adequate capital relief under "
                f"Basel III guidelines, and identify which asset classes drive the largest supervisory add-ons."
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "description": "Counterparty credit risk audit prompt",
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": prompt_text},
                        }
                    ],
                },
            }
        elif prompt_name == "stress_test_scenario":
            vol_shock = args.get("volatility_shock_pct", "50%")
            prompt_text = (
                f"Perform a macro market stress test on the benchmark derivative portfolio. "
                f"Apply a {vol_shock} volatility shock. Run both 'simulate_monte_carlo_pfe' and "
                f"'compute_portfolio_var' to report the shift in Peak Forward Exposure and 10-day 99% VaR. "
                f"Provide actionable hedging recommendations."
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "description": "Macro volatility stress test prompt",
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": prompt_text},
                        }
                    ],
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32602, "message": f"Prompt not found: {prompt_name}"},
            }

    def run_stdio(self):
        """Standard I/O event loop for MCP JSON-RPC protocol."""
        logger.info(f"Starting {self.SERVER_NAME} v{self.SERVER_VERSION} on stdio...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"Malformed JSON: {e}")
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")


def main():
    server = FinancialRiskMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
