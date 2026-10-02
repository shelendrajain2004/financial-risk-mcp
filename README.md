# High-Performance Financial Risk & Quantitative Analytics MCP Server

[![CI Suite](https://github.com/shelendra/financial-risk-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/shelendra/financial-risk-mcp/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![C++17/20](https://img.shields.io/badge/C%2B%2B-17%2F20-green.svg)](https://en.cppreference.com/)
[![MCP Specification](https://img.shields.io/badge/MCP-2024--11--05-orange.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade **Model Context Protocol (MCP)** server bridging institutional quantitative risk analytics to autonomous AI agents (Claude, Cursor, Devin, Copilot). 

Designed and engineered by **Shelendra Jain** (Technical Architect & Hands-on Tech Lead, 19+ years experience in low-latency systems and investment banking trading platforms).

---

## 🏛️ Architectural Overview

Regulated financial institutions (Tier-1 banks, hedge funds, prime brokers) possess mission-critical quantitative engines written in C++ and distributed microservices. However, connecting these engines to modern Large Language Models (LLMs) requires strict adherence to security boundaries, deterministic arithmetic, and formal protocol standards.

This repository provides a production-hardened reference implementation of an **MCP Server** exposing quantitative risk functions over standard JSON-RPC 2.0:

```
┌────────────────────────────────────────────────────────┐
│           Autonomous AI Agents / Front-Office UI       │
│         (Claude Desktop, Cursor, Devin, Copilot)       │
└───────────────────────────┬────────────────────────────┘
                            │ Standard JSON-RPC 2.0 (stdio / TCP)
                            ▼
┌────────────────────────────────────────────────────────┐
│               Financial Risk MCP Server                │
│    • Tools Registry        • Dynamic Resource Provider │
│    • Prompt Templates      • Protocol Handshake (v1.0) │
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│   Python Analytics Core   │ │   High-Performance C++   │
│  • BCBS 279 SA-CCR Engine │ │  • Multithreaded Engine  │
│  • Parametric VaR / CVaR  │ │  • Monte Carlo PFE Sim   │
│  • Options Greeks (Delta, │ │  • 6.8M Valuations/Sec   │
│    Gamma, Vega, Theta)    │ │  • Zero-Allocation Pools │
└───────────────────────────┘ └──────────────────────────┘
```

---

## ⚡ Key Capabilities & Quantitative Formulations

### 1. Basel III / BCBS 279 SA-CCR Engine
Computes Exposure at Default (**EAD**) under the Standardized Approach for Counterparty Credit Risk:
$$\text{EAD} = \alpha \times (\text{RC} + \text{PFE})$$
where $\alpha = 1.4$.

* **Replacement Cost (RC):**
  * *Unmargined:* $\text{RC} = \max(V - C, 0)$
  * *Margined (CSA):* $\text{RC} = \max(V - C, \text{Threshold} + \text{MTA} - \text{NICA}, 0)$
* **Potential Future Exposure (PFE):**
  $$\text{PFE} = \text{Multiplier} \times \text{AddOn}^{\text{aggregate}}$$
  $$\text{Multiplier} = \min\left(1.0, 0.05 + 0.95 \cdot \exp\left(\frac{V - C}{2 \times 0.95 \times \text{AddOn}^{\text{aggregate}}}\right)\right)$$
* **Supervisory Duration:**
  $$\text{SD}_i = \frac{\exp(-0.05 \cdot S_i) - \exp(-0.05 \cdot E_i)}{0.05}$$

### 2. Multithreaded Monte Carlo Peak Forward Exposure (PFE)
* Simulates stochastic rate paths across tenors from **0.25 years to 30 years** using mean-reverting Ornstein-Uhlenbeck / Hull-White state diffusion.
* Aggregates distribution profiles to calculate **Expected Exposure (EE)**, **PFE 95%**, **PFE 97.5%**, and **PFE 99%** quantiles.
* Automatically identifies the **Peak Forward Exposure** point along the tenor curve.

### 3. Value-at-Risk (VaR) & Expected Shortfall (CVaR)
* Calculates regulatory holding-period VaR (10-day 99% confidence interval) and Conditional VaR (Expected Shortfall) using variance-covariance analytical scaling:
$$\text{VaR}_\alpha = \text{PV} \cdot Z_\alpha \cdot \sigma_{\text{daily}} \sqrt{T}$$

### 4. Derivative Sensitivities (Greeks)
* Real-time analytical first- and second-order Greeks:
  * **Delta ($\Delta$):** First-order directional sensitivity.
  * **Gamma ($\Gamma$):** Second-order underlying convexity.
  * **Vega ($\nu$):** Volatility surface exposure.
  * **Theta ($\Theta$):** Calendar time decay.
  * **Rho ($\rho$):** Risk-free interest rate sensitivity.

---

## 📊 Benchmark Performance (C++20 Engine)

Executed on an Intel x86_64 host (2 worker threads, 100,000 paths across 11 tenors):

| Metric | Measured Value |
| :--- | :--- |
| **Total Simulated Paths** | 100,000 |
| **Tenor Discretization** | 11 steps (0.25y – 30.0y) |
| **Total Valuation Events** | 1,100,000 |
| **Execution Time** | **161.8 ms** |
| **Simulation Throughput** | **6,797,363 valuations / second** |

---

## 🛠️ MCP Primitives Exposed

### Tools
1. `calculate_sacr_exposure`: Basel III counterparty credit risk capital calculation.
2. `simulate_monte_carlo_pfe`: Vectorized multi-path Monte Carlo PFE simulation across 30y tenors.
3. `compute_portfolio_var`: Parametric and regulatory Value-at-Risk and Expected Shortfall.
4. `calculate_portfolio_greeks`: Multi-asset portfolio Greeks aggregation (Delta, Gamma, Vega, Theta, Rho).

### Resources
* `financial://portfolio/citi-benchmark-01`: Standardized institutional benchmark portfolio with Rates, FX, and Equity options.
* `financial://regulatory/bcbs279-factors`: Regulatory reference lookup table for BCBS 279 supervisory parameters.

### Prompts
* `audit_counterparty_risk`: Guided LLM agent workflow for credit risk auditing, EAD verification, and margin adequacy analysis.
* `stress_test_scenario`: Guided agent workflow for applying macroeconomic rate shocks and volatility spikes.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10 or higher
* GCC / G++ with C++17 support (optional for C++ benchmark)
* `pip install numpy`

### 2. Verify Installation
```bash
git clone https://github.com/shelendra/financial-risk-mcp.git
cd financial-risk-mcp

# Run unit tests
make test

# Run C++ high-performance benchmark
make bench-cpp

# Run end-to-end sample client
make run-demo
```

---

## 🔌 Integration with AI Development Environments

### A. Claude Desktop
Add to your `claude_desktop_config.json`:
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "financial-risk-engine": {
      "command": "python3",
      "args": ["-m", "fin_risk_mcp.server"],
      "cwd": "/path/to/financial-risk-mcp",
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

### B. Cursor IDE
Create or update `.cursor/mcp.json` in your workspace:
```json
{
  "mcpServers": {
    "financial-risk-engine": {
      "command": "python3",
      "args": ["-m", "fin_risk_mcp.server"],
      "cwd": "/path/to/financial-risk-mcp",
      "env": {
        "PYTHONPATH": "src"
      }
    }
  }
}
```

---

## 🧪 Testing Suite
The repository includes comprehensive unit tests verifying math accuracy against standard quantitative tables and testing full JSON-RPC protocol compliance:

```bash
$ make test

test_saccr_unmargined (test_engines.TestRiskEngines) ... ok
test_saccr_margined (test_engines.TestRiskEngines) ... ok
test_monte_carlo_pfe (test_engines.TestRiskEngines) ... ok
test_parametric_var (test_engines.TestRiskEngines) ... ok
test_greeks_calculation (test_engines.TestRiskEngines) ... ok
test_initialize (test_mcp_server.TestFinancialRiskMCPServer) ... ok
test_tools_list (test_mcp_server.TestFinancialRiskMCPServer) ... ok
test_call_calculate_sacr_exposure (test_mcp_server.TestFinancialRiskMCPServer) ... ok
test_call_simulate_monte_carlo_pfe (test_mcp_server.TestFinancialRiskMCPServer) ... ok
test_resources_read (test_mcp_server.TestFinancialRiskMCPServer) ... ok
test_prompts_get (test_mcp_server.TestFinancialRiskMCPServer) ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.035s

OK
```

---

## 👨‍💻 Author

**Shelendra Jain**  
*Senior Vice President – Technical Architect & Tech Lead*  
* Pune, India  
* Email: [shelendra.jain2004@gmail.com](mailto:shelendra.jain2004@gmail.com)  
* LinkedIn: [linkedin.com/in/shelendra](https://linkedin.com/in/shelendra)  

## 📄 License
This project is open-sourced under the [MIT License](LICENSE).
