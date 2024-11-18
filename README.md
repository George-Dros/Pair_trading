# Statistical Arbitrage Trading Strategy

## Description

This project implements a statistical arbitrage trading strategy using cointegrated stock pairs. By leveraging linear regression and Z-scores, the algorithm identifies mean-reverting opportunities and generates trading signals. Backtested on historical stock data, the portfolio grew from $24,050 to $25,489.50 over a 2-year period.

## Features

- Pair Selection:

    Identifies cointegrated pairs using the Engle-Granger method.
    Validates pairs with a 4-year in-sample and 2-year out-of-sample analysis.

- Trading Algorithm:

    Uses Z-scores of spreads for entry/exit signals.
    Implements long-short trades based on mean-reversion.

- Performance Metrics:

    Portfolio growth tracked with key metrics such as total return and Sharpe ratio.

  ## Installation

1. Clone the repository:
  ```
  bash
  git clone https://github.com/yourusername/statistical-arbitrage.git
  cd statistical-arbitrage
  ```
2. Install dependencies:
```
bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:

    Use Yahoo Finance or other APIs to download historical price data for stocks.

2. Run Pair Selection:

    Identify and validate cointegrated pairs using the provided scripts.

3. Backtesting:

    Simulate trades based on Z-score thresholds and track portfolio performance.

4. Visualization:

    View spread plots, Z-scores, and portfolio growth over time.

## Project Structure

```
bash
├── main.py               # Main script for pair selection and backtesting
├── functions.py          # Helper functions for cointegration, Z-scores, and trading logic
├── data/                 # Folder for historical stock data
├── screenshots/          # Output visuals for Z-scores and portfolio growth
│   ├── zscore_plot.jpg
│   ├── portfolio_growth.jpg
├── requirements.txt      # Project dependencies
├── LICENSE               # License information
├── README.md             # Project documentation (this file)
```

## Key Results

Initial Portfolio Value: $24,050
Final Portfolio Value: $25,000
Time Frame: 2 Years
Notable Metrics:
  Sharpe Ratio: 
  Win Rate: 

## Dependencies  

- Python 3.8+
- Libraries:
```
    pandas
    numpy
    matplotlib
    seaborn
    statsmodels
    yfinance
```
