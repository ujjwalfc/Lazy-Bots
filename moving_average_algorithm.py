
import yfinance as yf
# import matplotlib.pyplot as plt


stock = yf.Ticker("TCS.NS")
stock_data = stock.history(period='max')
print(len(stock_data))
stock_data.drop(["Volume", "Dividends", "Stock Splits"], inplace=True, axis=1)
stock_data["7MA"] = stock_data[["Close"]].rolling(100).mean()
stock_data.dropna(inplace=True)

# plt.style.use("seaborn-whitegrid")
# plt.figure(figsize=(20, 10))
# plt.plot(stock_data["Close"], label="Close")
# plt.plot(stock_data["7MA"], label="-")
# plt.legend(loc='upper center', fontsize="x-large")
# plt.show()

# business logic


def trade(sd, money):
    account_money = money
    state = "sold"
    shares_bought = 0
    profit = 0
    count = 0
    total = money
    stop_loss = money * 0.5
    for ind in range(len(sd)):
        # condition for stopping the cycle
        if total <= stop_loss and count > 0:
            print("you are broke")
            break

        # conditions for buying the stocks
        if sd['Close'][ind] > sd['7MA'][ind] and state == "sold":
            state = "bought"
            bought_at = sd['Close'][ind]
            shares_bought = account_money//bought_at
            count += 1
            # shares bought

            account_money = account_money - (shares_bought * bought_at)
            print("you are buying shares")
            print(shares_bought)
            print("price => ")
            print(bought_at)

        # conditions for selling the stocks
        elif sd['Close'][ind] < sd['7MA'][ind] and state == "bought":
            state = "sold"
            sold_at = sd['Close'][ind]

            # shares sold

            account_money += shares_bought * sold_at
            print("you are selling bought shares")
            print(shares_bought)
            print("price => ")
            print(sold_at)
            shares_bought = 0

        else:
            continue

        price = sd['Close'][ind]
        total = account_money + (shares_bought * price)
        profit = total - money
        print("profit => ")
        print(profit)
        print("money in account => ")
        print(account_money)

    print("you are exiting the market in a profit of:")
    print(profit)
    print(shares_bought)
    print("API hits => ")
    print(count)


trade(stock_data, 4000)
