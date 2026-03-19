from buy_strategies import PullbackStrategy, CoiledSpringStrategy, BlueSkyBreakoutStrategy, BullishDivergenceStrategy, BullishBaseBreakout
import pandas as pd
import yfinance as yf
from utils import add_moving_average, add_macd, add_stochastic, add_obv, add_rsi, add_cci

ticker = "^GSPC"
data = pd.DataFrame(yf.download(ticker, start='2024-01-01', end='2025-01-01'))

data = add_moving_average(data, "Close", 20)
data = add_moving_average(data, "Close", 50)
data = add_moving_average(data, "Close", 200)
data = add_macd(data, "Close")
data = add_stochastic(data, "High", "Low", "Close", 5, 3, 3)
data = add_obv(data, "Close", "Volume")
data = add_rsi(data, "Close", 5)
data = add_cci(data, "High", "Low", "Close", 20)

print(data.columns)

Pullback = BlueSkyBreakoutStrategy(data)

signals = Pullback.generate_signals()
trades = Pullback.backtest()
profit_losses = Pullback.analyze_results(trades)
Pullback.plot_results(trades)