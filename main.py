import numpy as np
import pandas as pd
import time
import yfinance as yf
from yahooquery import Screener
from yahooquery import Ticker  

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import functions as f


import matplotlib.pyplot as plt

#  Get companies and find initial cointegrated pairs
all_companies = f.get_companies()
filtered_symbols_df_2 = f.filter_all_companies(all_companies)
pairs_data_2 = f.find_cointegrated_pairs(
    filtered_symbols_df_2,
    p_value_threshold=0.02,
    min_data_points_threshold=100
)

# Validate pairs over a new time period
validated_pairs_sorted = f.validate_pairs(
    pairs_list=pairs_data_2,
    start_date="2022-11-16",
    end_date="2024-11-16",
    p_value_threshold=0.05
)

# Print the validated pairs
print("\nValidated Pairs in Ascending Order of p-value:")
for pair in validated_pairs_sorted:
    print(f"The pair: {pair[0]}, {pair[1]} with p-value = {pair[2]:.4f}")


# Perform linear regression on the validated pairs, with plotting enabled
results_list = f.perform_linear_regression(
    validated_pairs_list=validated_pairs_sorted,
    start_date="2018-11-16",
    end_date="2022-11-16",
    plot=True  # Enable plotting and saving of regression plots
)

# Save the regression results to a CSV file
results_df = pd.DataFrame(results_list, columns=['Ticker 1', 'Ticker 2', 'Slope (b)'])
results_df.to_csv('regression_results.csv', index=False)
print("All regression results have been saved to 'regression_results.csv'")

# Calculate and plot spreads from the CSV file, saving the plots
csv_file = 'regression_results.csv'
start_date = "2022-11-16"
end_date = "2024-11-16"

# Calculate and plot spreads, plots will be saved by the function
spreads_analysis_dict = f.calculate_and_plot_spreads_from_csv(csv_file, start_date, end_date)

# Compile and save the crossings count and evaluation for each pair
analysis_results_list = []
for pair, analysis in spreads_analysis_dict.items():
    ticker1, ticker2 = pair
    crossings_plus_1 = analysis['Crossings +1σ']
    crossings_plus_2 = analysis['Crossings +2σ']
    crossings_minus_1 = analysis['Crossings -1σ']
    crossings_minus_2 = analysis['Crossings -2σ']
    total_crossings = analysis['Total Crossings']
    evaluation = analysis['Evaluation']

    analysis_results_list.append({
        'Ticker 1': ticker1,
        'Ticker 2': ticker2,
        'Crossings +1σ': crossings_plus_1,
        'Crossings +2σ': crossings_plus_2,
        'Crossings -1σ': crossings_minus_1,
        'Crossings -2σ': crossings_minus_2,
        'Total Crossings': total_crossings,
        'Evaluation': evaluation
    })

# Save the analysis results to a CSV file
analysis_results_df = pd.DataFrame(analysis_results_list)
analysis_results_df.to_csv('spread_analysis_results.csv', index=False)
print("Spread analysis results have been saved to 'spread_analysis_results.csv'")
