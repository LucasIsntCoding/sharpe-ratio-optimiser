# Sharpe Ratio Optimizer

A technical Python implementation of a **mean-variance portfolio optimizer** that maximizes the **Sharpe ratio** using historical asset returns. This project combines Monte Carlo portfolio simulation with constrained numerical optimization to identify the optimal long-only portfolio under realistic allocation constraints.

It includes:

- historical return estimation from price data
- annualized expected return and covariance estimation
- random portfolio generation for baseline search
- constrained Sharpe ratio maximization via `scipy.optimize`
- global minimum variance portfolio construction
- efficient frontier generation
- portfolio visualization and risk-return analysis

## Overview

This project implements the core logic of **modern portfolio theory** in a practical, reproducible way. Given a set of historical asset prices, the optimizer computes:

- expected annualized returns
- annualized covariance matrix
- portfolio return, volatility, and Sharpe ratio
- the long-only portfolio with the maximum Sharpe ratio
- the minimum-variance portfolio
- the efficient frontier across feasible target returns

The optimization is performed under the following constraints:

- portfolio weights sum to 1
- no short-selling (`0 <= w_i <= 1`)
- full capital allocation

## Mathematical Formulation

For a portfolio with weights `w`, expected return vector `mu`, covariance matrix `Sigma`, and risk-free rate `r_f`:

- Portfolio return: `mu_p = w^T mu`
- Portfolio variance: `sigma_p^2 = w^T Sigma w`
- Portfolio volatility: `sigma_p = sqrt(w^T Sigma w)`
- Sharpe ratio: `(mu_p - r_f) / sigma_p`

The optimization problem is:

```text
maximize    (w^T mu - r_f) / sqrt(w^T Sigma w)

subject to  sum(w_i) = 1
            0 <= w_i <= 1
```

The efficient frontier is constructed by solving a sequence of minimum-variance problems for fixed target returns.

## Features

- **Robust preprocessing**
  - cleans and validates price data
  - supports log returns and simple returns
  - drops invalid and constant-price series
  - applies covariance matrix ridge regularization for numerical stability

- **Monte Carlo baseline**
  - samples feasible long-only portfolios using a Dirichlet distribution
  - estimates return, volatility, and Sharpe ratio across thousands of portfolios

- **Constrained optimization**
  - uses **SLSQP** from `scipy.optimize`
  - solves for:
    - maximum Sharpe ratio portfolio
    - global minimum variance portfolio
    - efficient frontier portfolios at target returns

- **Visualization**
  - random portfolio cloud
  - Sharpe ratio color map
  - efficient frontier curve
  - maximum Sharpe portfolio marker
  - individual asset risk-return points

## Project Structure

```text
.
├── sharpe_ratio_optimizer.py   # main optimizer implementation
├── prices.csv                  # optional real price data input
└── README.md
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/sharpe-ratio-optimizer.git
cd sharpe-ratio-optimizer
pip install numpy pandas matplotlib scipy
```

## Requirements

- Python 3.9+
- NumPy
- pandas
- matplotlib
- SciPy

## Usage

### 1. Run with synthetic test data

The script includes a built-in synthetic price generator for testing and demonstration.

```bash
python sharpe_ratio_optimizer.py
```

This will:

- generate correlated synthetic asset prices
- estimate annualized return and covariance statistics
- run a large random portfolio search
- compute the maximum Sharpe portfolio
- compute the global minimum variance portfolio
- construct the efficient frontier
- display a risk-return plot

### 2. Run with your own historical price data

Replace the synthetic data section in `main()` with a CSV load:

```python
prices = load_prices_from_csv("prices.csv")
optimizer = SharpeRatioOptimizer(prices=prices)
```

Your CSV should contain:

- one date index column
- one column per asset
- strictly positive price levels (preferably adjusted close prices)

Example format:

```csv
Date,AAPL,MSFT,GOOGL,AMZN
2023-01-03,125.07,239.58,89.70,85.82
2023-01-04,126.36,229.10,88.71,85.14
2023-01-05,125.02,222.31,86.77,83.12
```

## Example Workflow

A typical workflow looks like this:

```python
from sharpe_ratio_optimizer import SharpeRatioOptimizer, OptimizerConfig, load_prices_from_csv

prices = load_prices_from_csv("prices.csv")

config = OptimizerConfig(
    risk_free_rate=0.03,
    periods_per_year=252,
    use_log_returns=True,
    allow_short=False,
    covariance_ridge=1e-10,
    random_seed=123
)

optimizer = SharpeRatioOptimizer(prices=prices, config=config)

random_portfolios = optimizer.random_search(n_portfolios=50000)
max_sharpe_solution = optimizer.optimize_max_sharpe()
min_var_solution = optimizer.optimize_min_variance()
frontier = optimizer.efficient_frontier(n_points=60)

print(max_sharpe_solution["weights"])
print(max_sharpe_solution["metrics"])

optimizer.plot(
    random_portfolios=random_portfolios,
    frontier=frontier,
    optimal_solution=max_sharpe_solution
)
```

## Output

The optimizer returns structured portfolio results including:

- optimized portfolio weights
- expected annualized return
- annualized volatility
- Sharpe ratio
- raw SciPy optimization output

Example:

```python
{
    "weights": pd.Series(...),
    "metrics": {
        "expected_return": 0.1482,
        "volatility": 0.1725,
        "sharpe_ratio": 0.6852
    },
    "scipy_result": ...
}
```

## Visualization

The generated chart includes:

- a scatter of random long-only portfolios
- color-coded Sharpe ratios
- the efficient frontier
- the optimal maximum Sharpe portfolio
- individual asset risk-return coordinates

This makes it easy to compare:

- brute-force simulated allocations
- numerically optimized allocations
- the frontier of efficient risk-return combinations

## Configuration

The optimizer is controlled through the `OptimizerConfig` dataclass:

```python
OptimizerConfig(
    risk_free_rate=0.02,
    periods_per_year=252,
    use_log_returns=True,
    allow_short=False,
    covariance_ridge=1e-10,
    random_seed=42
)
```

### Parameter Notes

- `risk_free_rate`: annualized risk-free rate used in Sharpe ratio calculation
- `periods_per_year`: trading periods used for annualization (`252` for daily data)
- `use_log_returns`: whether to compute log returns instead of simple returns
- `allow_short`: enables/disables short-selling bounds
- `covariance_ridge`: small diagonal regularization term for numerical stability
- `random_seed`: ensures reproducible Monte Carlo sampling

## Why This Project Matters

This project demonstrates the quantitative foundations of portfolio construction and risk-adjusted optimization. It is directly relevant to:

- quantitative finance
- portfolio analytics
- asset allocation
- risk modeling
- systematic trading research

It showcases practical use of:

- vectorized numerical computing
- covariance-based risk estimation
- constrained nonlinear optimization
- simulation-based portfolio search
- financial data engineering and visualization

## Possible Extensions

Future improvements could include:

- Ledoit-Wolf covariance shrinkage
- Black-Litterman return adjustment
- transaction cost and turnover penalties
- rolling-window walk-forward optimization
- out-of-sample performance backtesting
- leverage constraints
- sector or exposure constraints
- downside-risk objectives such as Sortino ratio or CVaR optimization

## Disclaimer

This project is for educational and research purposes only. It is not financial advice and should not be used as the sole basis for investment decisions.

## License

This project is released under the MIT License. You may use, modify, and distribute it freely.

---

### Author

Built as a quantitative finance project focused on portfolio optimization, numerical methods, and risk-adjusted performance analysis.
