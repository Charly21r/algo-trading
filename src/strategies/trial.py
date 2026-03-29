from buy_strategies import PullbackStrategy, CoiledSpringStrategy, BlueSkyBreakoutStrategy, BullishDivergenceStrategy, BullishBaseBreakout
from sell_strategies import ReliefRallyShortStrategy, BearishDivergenceStrategy, GapDownShortStrategy, BlueSeaBreakdownStrategy, RisingWedgeBreakdownStrategy
import pandas as pd
import yfinance as yf
from utils import add_moving_average, add_macd, add_stochastic, add_obv, add_rsi, add_cci

ticker = "CD"
data = pd.DataFrame(yf.download(ticker, start='2024-01-01', end='2025-01-01'))

print(data.iloc[-1]["Close"].values[0]-data.iloc[0]["Close"].values[0])
data = add_moving_average(data, "Close", 20)
data = add_moving_average(data, "Close", 50)
data = add_moving_average(data, "Close", 200)
data = add_macd(data, "Close")
data = add_stochastic(data, "High", "Low", "Close", 5, 3, 3)
data = add_obv(data, "Close", "Volume")
data = add_rsi(data, "Close", 5)
data = add_cci(data, "High", "Low", "Close", 20)

print(data.columns)

Pullback = RisingWedgeBreakdownStrategy(data)

signals = Pullback.generate_signals()
trades, final_capital = Pullback.backtest()
profit_losses = Pullback.analyze_results(trades, final_capital)
Pullback.plot_results(trades)
print("Baseline:", (data["Close"].iloc[-1].values[0]-data.iloc[0]["Close"].values[0])/data.iloc[0]["Close"].values[0])
print(data["Close"].iloc[-1].values[0], data.iloc[0]["Close"].values[0])