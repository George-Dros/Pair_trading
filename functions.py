import numpy as np
import pandas as pd
import time
import yfinance as yf
from matplotlib import pyplot as plt
from yahooquery import Screener
from yahooquery import Ticker

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller



def get_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    sp500_df = table[0]
    sp500_symbols = sp500_df['Symbol'].tolist()
    tickers = Ticker(sp500_symbols)
    profiles = tickers.summary_profile
    profiles_df = pd.DataFrame(profiles).T.reset_index()
    profiles_df.rename(columns={'index': 'symbol'}, inplace=True)

    return profiles_df

def filter_all_companies(companies, marketCap_thresh=2_000_000_000, averageVolume_thresh=5_000_000, start_date="2018-11-16", end_date="2022-11-16"):
    """
    Filters companies by market cap and average volume.
    Fetches historical closing prices for each filtered symbol.

    Parameters:
    - companies: DataFrame containing company data, including 'sector' and 'symbol'.
    - marketCap_thresh: Minimum market capitalization (default is 2 billion).
    - averageVolume_thresh: Minimum average volume (default is 5 million).
    - start_date: Start date for historical data.
    - end_date: End date for historical data.
    """
    # Get the list of all company symbols
    all_symbols = companies['symbol'].tolist()

    filtered_symbols = []
    for symbol in all_symbols:
        try:
            # Fetch ticker info with error handling
            ticker_info = yf.Ticker(symbol).info

            # Check if data exists and if marketCap and averageVolume thresholds are met
            market_cap = ticker_info.get("marketCap")
            average_volume = ticker_info.get("averageVolume")
            if market_cap is not None and market_cap > marketCap_thresh and \
                    average_volume is not None and average_volume > averageVolume_thresh:
                filtered_symbols.append(symbol)
        except Exception as e:
            # Print or log error message for the symbol
            print(f"Error fetching data for {symbol}: {e}")
        # Add delay to avoid rate limiting
        time.sleep(0.05)

    # Initialize an empty DataFrame to store the historical closing prices
    filtered_symbols_df = pd.DataFrame()

    # Fetch historical data for each filtered symbol
    for symbol in filtered_symbols:
        try:
            ticker = yf.Ticker(symbol)
            historical_data = ticker.history(start=start_date, end=end_date)
            # Store the 'Close' prices in the DataFrame
            filtered_symbols_df[symbol] = historical_data['Close']
        except Exception as e:
            # Print or log error message for the symbol if historical data cannot be fetched
            print(f"Error fetching historical data for {symbol}: {e}")
        # Add delay to avoid rate limiting
        time.sleep(0.05)

    return filtered_symbols_df


def find_cointegrated_pairs(combined_df, p_value_threshold=0.07, min_data_points_threshold=100):
    """
    Finds and prints pairs of stock symbols that are cointegrated.

    Parameters:
    - combined_df: DataFrame containing historical daily prices with symbols as columns.
    - p_value_threshold: The significance level for cointegration (default is 0.02).
    - min_data_points_threshold: The minimum number of data points required to test cointegration.

    Returns:
    - pairs_sorted: List of tuples containing cointegrated pairs sorted by p-value.
    """

    symbols = combined_df.columns
    n = len(symbols)

    pairs = []

    # Loop through symbols and perform the cointegration test for each unique pair
    for i in range(n):
        for j in range(i + 1, n):
            symbol_1 = symbols[i]
            symbol_2 = symbols[j]

            series1 = combined_df[symbol_1]
            series2 = combined_df[symbol_2]

            # Combine series and drop NaN values
            combined_pair = pd.concat([series1, series2], axis=1).dropna()
            data_points = len(combined_pair)

            # Debug: Print the length of combined data
            print(f"Testing pair {symbol_1} and {symbol_2}, Number of data points: {data_points}")

            # Check if there are enough data points
            if data_points >= min_data_points_threshold:
                # Perform the cointegration test
                coint_t, p_value, _ = coint(combined_pair.iloc[:, 0], combined_pair.iloc[:, 1])

                # Check if the p-value is below the threshold
                if p_value < p_value_threshold:
                    print(f"{symbol_1} and {symbol_2} are likely cointegrated (p-value: {p_value:.4f})")
                    pairs.append((symbol_1, symbol_2, p_value))
                    # Debug: Confirm pairs being appended correctly
                    print(f"Current pairs in function after appending: {pairs}")

    # Debug: Confirm list of pairs before sorting
    print(f"Pairs before sorting: {pairs}")

    # Sort pairs by p-value
    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    # Debug: Print final sorted pairs before returning
    print(f"Final sorted pairs in function: {pairs_sorted}")

    return pairs_sorted


def validate_pairs(pairs_list, start_date, end_date, p_value_threshold=0.05):
    """
    Validates cointegrated pairs over a new time period.

    Parameters:
    - pairs_list: List of tuples containing cointegrated pairs and their p-values.
    - start_date: Start date for the validation period.
    - end_date: End date for the validation period.
    - p_value_threshold: p-value threshold for validation (default is 0.05).

    Returns:
    - validated_pairs_sorted: List of tuples containing validated pairs sorted by p-value.
    """
    validated_pairs = []

    for tup in pairs_list:
        try:
            ticker_1 = yf.Ticker(tup[0])
            ticker_2 = yf.Ticker(tup[1])
            data_1 = ticker_1.history(start=start_date, end=end_date)["Close"]
            data_2 = ticker_2.history(start=start_date, end=end_date)["Close"]

            # Perform the cointegration test
            coint_t, p_value, _ = coint(data_1, data_2)

            # If p-value is below the threshold, add to validated pairs list and print
            if p_value < p_value_threshold:
                print(f"The pair: {tup[0]}, {tup[1]} are validated with p-value = {p_value:.4f}")
                validated_pairs.append((tup[0], tup[1], p_value))

        except Exception as e:
            print(f"Error processing pair {tup[0]}, {tup[1]}: {e}")

    # Sort the validated pairs by p-value in ascending order
    validated_pairs_sorted = sorted(validated_pairs, key=lambda x: x[2])

    return validated_pairs_sorted


def perform_linear_regression(validated_pairs_list, start_date, end_date, plot=False):
    """
    Performs linear regression on validated pairs and returns the slope (b) for each pair.

    Parameters:
    - validated_pairs_list: List of tuples containing validated pairs.
    - start_date: Start date for the regression period.
    - end_date: End date for the regression period.
    - plot: Boolean indicating whether to plot the regression results (default is False).

    Returns:
    - results_list: List of tuples containing tickers and slope (b) values.
    """
    results_list = []

    for tup in validated_pairs_list:
        try:
            # Fetch historical closing prices for the validated pair
            ticker_1 = yf.Ticker(tup[0])
            ticker_2 = yf.Ticker(tup[1])
            x_1 = ticker_1.history(start=start_date, end=end_date)["Close"]
            x_2 = ticker_2.history(start=start_date, end=end_date)["Close"]

            # Drop NaN values to align both series
            combined_data = pd.concat([x_1, x_2], axis=1).dropna()
            x_1 = combined_data.iloc[:, 0]
            x_2 = combined_data.iloc[:, 1]

            # Ensure that x_1 and x_2 are numpy arrays
            x_1 = np.array(x_1)
            x_2 = np.array(x_2)

            # Add a constant to x_1 for the intercept term
            x_1_with_const = sm.add_constant(x_1)

            # Perform linear regression
            model = sm.OLS(x_2, x_1_with_const)
            results = model.fit()

            # Extract the slope (b) from the regression results
            b = results.params[1]

            # Save the pair of tickers and the value of b to the results list
            results_list.append((tup[0], tup[1], b))

            # Optionally, print the summary of the regression and the value of b
            print(f"Linear regression for pair: {tup[0]} and {tup[1]}")
            print(results.summary())
            print(f"Slope (b) of the regression: {b}")

            # Optionally, plot the results
            if plot:
                plt.scatter(x_1, x_2, color='blue', label=f'Data Points: {tup[0]} vs {tup[1]}')
                plt.plot(x_1, results.predict(x_1_with_const), color='red', label='Regression Line')
                plt.xlabel(f'{tup[0]} (Price)')
                plt.ylabel(f'{tup[1]} (Price)')
                plt.title(f'Linear Regression of Stock Prices: {tup[0]} vs {tup[1]}')
                plt.legend()

                # Save the plot to a file
                plot_filename = f"regression_plot_{tup[0]}_{tup[1]}.png"
                plt.savefig(plot_filename)
                print(f"Saved regression plot for {tup[0]} and {tup[1]} to {plot_filename}")

                plt.show()
                plt.clf()

        except Exception as e:
            print(f"Error processing pair {tup[0]}, {tup[1]}: {e}")

    return results_list


def calculate_and_plot_spreads_from_csv(csv_file, start_date, end_date):
    """
    Calculates and plots the Z-score of the spreads for each pair from the CSV file over the specified date range.
    Includes threshold lines at +1σ and ±2σ and counts the frequency of crossings.

    Parameters:
    - csv_file: Path to the CSV file containing 'Ticker 1', 'Ticker 2', and 'Slope (b)' columns.
    - start_date: Start date for fetching historical prices (format 'YYYY-MM-DD').
    - end_date: End date for fetching historical prices (format 'YYYY-MM-DD').

    Returns:
    - spreads_analysis_dict: A dictionary where keys are ticker pairs and values are dictionaries containing spread DataFrame and crossing counts.
    """

    from statsmodels.tsa.stattools import adfuller

    # Read the CSV file into a DataFrame
    pairs_df = pd.read_csv(csv_file)

    # Initialize a dictionary to store analysis results for each pair
    spreads_analysis_dict = {}

    # Iterate over each pair in the DataFrame
    for index, row in pairs_df.iterrows():
        ticker1 = row['Ticker 1']
        ticker2 = row['Ticker 2']
        b = row['Slope (b)']

        try:
            # Fetch historical prices for both tickers
            data1 = yf.Ticker(ticker1).history(start=start_date, end=end_date)['Close']
            data2 = yf.Ticker(ticker2).history(start=start_date, end=end_date)['Close']

            # Combine the data into a single DataFrame, aligning on dates
            combined_data = pd.concat([data1, data2], axis=1, keys=[ticker1, ticker2]).dropna()

            # Calculate the spread
            combined_data['Spread'] = combined_data[ticker1] - b * combined_data[ticker2]

            # Calculate mean and standard deviation of the spread
            spread_mean = combined_data['Spread'].mean()
            spread_std = combined_data['Spread'].std()

            # Calculate the Z-score of the spread
            combined_data['Z-score'] = (combined_data['Spread'] - spread_mean) / spread_std

            # Identify threshold crossings
            z_scores = combined_data['Z-score']

            # Crossings of +1σ and +2σ
            crossings_plus_1 = ((z_scores.shift(1) < 1) & (z_scores >= 1)).sum()
            crossings_plus_2 = ((z_scores.shift(1) < 2) & (z_scores >= 2)).sum()

            # Crossings of -1σ and -2σ
            crossings_minus_1 = ((z_scores.shift(1) > -1) & (z_scores <= -1)).sum()
            crossings_minus_2 = ((z_scores.shift(1) > -2) & (z_scores <= -2)).sum()

            # Total crossings
            total_crossings = crossings_plus_1 + crossings_plus_2 + crossings_minus_1 + crossings_minus_2

            # Evaluate the pair based on crossings (you can define your own criteria)
            if total_crossings >= 10:
                evaluation = 'Good for trading'
            elif total_crossings >= 5:
                evaluation = 'Average for trading'
            else:
                evaluation = 'Not ideal for trading'

            # Store the analysis results
            analysis_results = {
                'Data': combined_data,
                'Crossings +1σ': crossings_plus_1,
                'Crossings +2σ': crossings_plus_2,
                'Crossings -1σ': crossings_minus_1,
                'Crossings -2σ': crossings_minus_2,
                'Total Crossings': total_crossings,
                'Evaluation': evaluation
            }
            spreads_analysis_dict[(ticker1, ticker2)] = analysis_results

            # Plot the Z-score
            plt.figure(figsize=(12, 6))
            combined_data['Z-score'].plot(title=f"Z-score of Spread: {ticker1} - {b:.4f} * {ticker2}")
            plt.xlabel('Date')
            plt.ylabel('Z-score')
            plt.grid(True)

            # Add horizontal lines for thresholds
            plt.axhline(0, color='black', linestyle='--', label='Mean (Z=0)')
            plt.axhline(1, color='green', linestyle='--', label='+1σ Threshold')
            plt.axhline(-1, color='green', linestyle='--', label='-1σ Threshold')
            plt.axhline(2, color='red', linestyle='--', label='+2σ Threshold')
            plt.axhline(-2, color='red', linestyle='--', label='-2σ Threshold')

            plt.legend()

            # Annotate the plot with crossing counts and evaluation
            plt.text(0.02, 0.95, f'Crossings +1σ: {crossings_plus_1}\nCrossings +2σ: {crossings_plus_2}\n'
                                  f'Crossings -1σ: {crossings_minus_1}\nCrossings -2σ: {crossings_minus_2}\n'
                                  f'Total Crossings: {total_crossings}\nEvaluation: {evaluation}',
                     transform=plt.gca().transAxes, verticalalignment='top')

            # Save the plot to a file
            plot_filename = f"zscore_plot_{ticker1}_{ticker2}.png"
            plt.savefig(plot_filename)
            print(f"Saved Z-score plot for {ticker1} and {ticker2} to {plot_filename}")

            plt.show()
            plt.clf()  # Clear the figure

        except Exception as e:
            print(f"Error calculating spread for pair {ticker1}, {ticker2}: {e}")

    return spreads_analysis_dict


