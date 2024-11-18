import pandas as pd
import yfinance as yf

start_date="2022-11-16"
end_date="2024-11-16"

regression_results = pd.read_csv("regression_results.csv")

n = len(regression_results)

capital = 24000
invest_per_pair = round(float(capital / n), 2)
portion_for_transactions = 50

portfolio_value = invest_per_pair * n + portion_for_transactions


test_pair = regression_results.iloc[0]
print(test_pair)

ticker_1_data = yf.Ticker(test_pair["Ticker 1"]).history(start=start_date, end=end_date)["Close"]
ticker_2_data = yf.Ticker(test_pair["Ticker 2"]).history(start=start_date, end=end_date)["Close"]
b = test_pair["Slope (b)"]

ticker_1 = test_pair["Ticker 1"]
ticker_2 = test_pair["Ticker 2"]


pair_data = pd.concat([ticker_1_data, ticker_2_data], axis=1, keys=[ticker_1, ticker_2]).dropna()
pair_data["Spreads"] = pair_data[ticker_1] - b * pair_data[ticker_2]

spread_mean = pair_data["Spreads"].mean()
spread_std = pair_data["Spreads"].std()

pair_data["Z_Scores"] = (pair_data["Spreads"] - spread_mean) / spread_std

prev_row = None
prev_position = None  # Track if we are in a long or short position

shares_a = 0
shares_b = 0


cash = invest_per_pair

for row in pair_data.itertuples():
    price_a = getattr(row, ticker_1)
    price_b = getattr(row, ticker_2)

    # Update portfolio value based on current holdings
    portfolio_value = cash + (shares_a * price_a) + (shares_b * price_b)
    print(f"Date: {row.Index.date()}, Total Portfolio Value: {portfolio_value:.2f}")

    if prev_row is not None:
        # Buy signal (enter long position)
        if (row.Z_Scores < -2) and (prev_row.Z_Scores >= -2) and prev_position != "long":
            print(f"Buy on {row.Index.date()}, Z-Score dropped below -2 to: {row.Z_Scores:.2f}")
            prev_position = 'long'

            # Calculate shares for Stock A (long position)
            shares_a = cash / price_a

            # Calculate shares for Stock B (short position)
            shares_b = - b * shares_a

            # Update cash
            cash -= shares_a * price_a  # Cash used to buy Stock A
            cash += - shares_b * price_b  # Cash received from shorting Stock B

            print(f"Shares A (Long): {shares_a}, Shares B (Short): {shares_b}, Cash: {cash:.2f}")

        # Sell signal (enter short position)
        elif (row.Z_Scores > 2) and (prev_row.Z_Scores <= 2) and prev_position != "short":
            print(f"Sell on {row.Index.date()}, Z-Score increased above 2 to: {row.Z_Scores:.2f}")
            prev_position = 'short'

            # Calculate shares for Stock A (short position)
            shares_a = - (cash / price_a)

            # Calculate shares for Stock B (long position)
            shares_b = - (shares_a / b)

            # Update cash
            cash += - shares_a * price_a  # Cash received from shorting Stock A
            cash -= shares_b * price_b  # Cash used to buy Stock B

            print(f"Shares A (Short): {shares_a}, Shares B (Long): {shares_b}, Cash: {cash:.2f}")

        # Exit long position
        elif prev_position == 'long' and (row.Z_Scores > -1) and (row.Z_Scores < 0) and (prev_row.Z_Scores <= -1):
            print(f"Exit long position on {row.Index.date()}, Z-Score is: {row.Z_Scores:.2f}")
            # Update cash
            cash += shares_a * price_a  # Cash from selling Stock A
            cash -= - shares_b * price_b  # Cash used to cover Stock B

            # Reset positions
            shares_a = 0
            shares_b = 0
            prev_position = None
            print(f"Exited Long Position. Cash: {cash:.2f}")

        # Exit short position
        elif prev_position == 'short' and (row.Z_Scores < 1) and (row.Z_Scores > 0) and (prev_row.Z_Scores >= 1):
            print(f"Exit short position on {row.Index.date()}, Z-Score is: {row.Z_Scores:.2f}")

            # Update cash
            cash -= - shares_a * price_a  # Cash used to cover Stock A
            cash += shares_b * price_b  # Cash from selling Stock B

            # Reset positions
            shares_a = 0
            shares_b = 0
            prev_position = None
            print(f"Exited Short Position. Cash: {cash:.2f}")

    # Update previous row
    prev_row = row

# After the loop: Final cash out
if shares_a != 0 or shares_b != 0:
    # Calculate proceeds from selling long shares
    if shares_a > 0:
        cash += shares_a * price_a  # Sell all long shares of Stock A
        print(f"Sold all shares of {ticker_1}: {shares_a} at {price_a:.2f} each. Proceeds: {shares_a * price_a:.2f}")
    if shares_b > 0:
        cash += shares_b * price_b  # Sell all long shares of Stock B
        print(f"Sold all shares of {ticker_2}: {shares_b} at {price_b:.2f} each. Proceeds: {shares_b * price_b:.2f}")

    # Calculate cost of covering short positions
    if shares_a < 0:
        cash -= -shares_a * price_a  # Cover all short shares of Stock A
        print(f"Covered all short shares of {ticker_1}: {-shares_a} at {price_a:.2f} each. Cost: {-shares_a * price_a:.2f}")
    if shares_b < 0:
        cash -= -shares_b * price_b  # Cover all short shares of Stock B
        print(f"Covered all short shares of {ticker_2}: {-shares_b} at {price_b:.2f} each. Cost: {-shares_b * price_b:.2f}")

    # Reset positions
    shares_a = 0
    shares_b = 0

# Print final portfolio value
print(f"Final Portfolio Value: {cash:.2f}")



   # for row in data.itertuples():
   #     date = row.Index
   #     price1 = row.Ticker1
   #     price2 = row.Ticker2
   #     b = row.Slope

   # spread = price1 - b * price2
    #z_score = 1
