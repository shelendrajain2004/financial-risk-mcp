"""Sample client showing how an LLM agent communicates with the Financial Risk MCP Server over stdio.
Directly invokes both SA-CCR calculations and the high-performance native C++ Monte Carlo PFE simulation.
"""

import sys
import json
import subprocess


def main():
    print("Launching Financial Risk MCP Server subprocess...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "fin_risk_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def send_request(req):
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line)

    # 1. Initialize Handshake
    init_res = send_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "demo-agent", "version": "1.0"},
        },
    })
    print("\n[Handshake Completed]:", init_res["result"]["serverInfo"])

    # 2. List Tools
    tools_res = send_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    print(f"\n[Tools Available]: {len(tools_res['result']['tools'])} tools registered")
    for t in tools_res["result"]["tools"]:
        print(f"  * {t['name']}: {t['description'][:75]}...")

    # 3. Call SA-CCR Calculation
    print("\nExecuting SA-CCR Calculation on Institutional Netting Set...")
    saccr_res = send_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_sacr_exposure",
            "arguments": {
                "netting_set_id": "NS-MORGAN-STANLEY-001",
                "counterparty_id": "Morgan Stanley & Co.",
                "collateral_posted": 2_500_000.0,
                "is_margined": True,
                "threshold": 500_000.0,
                "minimum_transfer_amount": 100_000.0,
                "trades": [
                    {
                        "trade_id": "IRS-5Y-EURIBOR",
                        "asset_class": "INTEREST_RATE",
                        "notional": 100_000_000.0,
                        "market_value": 4_200_000.0,
                        "maturity_years": 5.0,
                    },
                    {
                        "trade_id": "FX-FWD-USDJPY",
                        "asset_class": "FOREIGN_EXCHANGE",
                        "notional": 50_000_000.0,
                        "market_value": -850_000.0,
                        "maturity_years": 2.0,
                    },
                    {
                        "trade_id": "EQ-CALL-NVDA",
                        "asset_class": "EQUITY",
                        "notional": 15_000_000.0,
                        "market_value": 1_100_000.0,
                        "maturity_years": 1.0,
                        "option_type": "CALL",
                        "strike": 120.0,
                        "underlying_price": 128.0,
                        "volatility": 0.35,
                    },
                ],
            },
        },
    })
    print("\n--- Agent Response 1: SA-CCR ---")
    print(saccr_res["result"]["content"][0]["text"])

    # 4. Call High-Performance Monte Carlo PFE Simulation (Invokes Native C++ Engine)
    print("\nExecuting Monte Carlo PFE Simulation (100,000 paths across 30y tenors via C++ Engine)...")
    pfe_res = send_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "simulate_monte_carlo_pfe",
            "arguments": {
                "portfolio_id": "PORT-DERIVATIVES-GLOBAL",
                "num_simulations": 100000,
                "trades": [
                    {
                        "trade_id": "IRS-5Y-EURIBOR",
                        "asset_class": "INTEREST_RATE",
                        "notional": 100_000_000.0,
                        "market_value": 4_200_000.0,
                        "maturity_years": 10.0,
                    }
                ],
            },
        },
    })
    print("\n--- Agent Response 2: Monte Carlo PFE (Native C++) ---")
    print(pfe_res["result"]["content"][0]["text"])

    proc.terminate()
    proc.wait()


if __name__ == "__main__":
    main()
