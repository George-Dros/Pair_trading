import pandas as pd
import functions as f

start_date="2022-11-16"
end_date="2024-11-16"

regression_results = pd.read_csv("regression_results.csv")
pair_to_remove = (regression_results["Ticker 1"] == "PCG") & (regression_results["Ticker 2"] == "WMB")
regression_results_cleaned = regression_results[~pair_to_remove].reset_index(drop=True)
regression_results = regression_results_cleaned

n = len(regression_results)

capital = 24000
invest_per_pair = round(float(capital / n), 2)
portion_for_transactions = 50

portfolio_value = invest_per_pair * n + portion_for_transactions


final_value = 0

for i in range(len(regression_results)):
    test_pair = regression_results.iloc[i]

    pair_value = f.pair_trading_strategy_from_regression(
            test_pair=test_pair,
            invest_per_pair=invest_per_pair,
            start_date=start_date,
            end_date=end_date
        )

    final_value += pair_value
    print(f"Pair final value: {pair_value}")
print(f"\nTotal Final Portfolio Value Across All Pairs: {final_value:.2f}")
