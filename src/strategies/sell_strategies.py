import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaseStrategy:
    def __init__(self, df, initial_capital=10000):
        self.df = df
        self.initial_capital = initial_capital
        self.signals = df.copy()

    def backtest(self):
        in_trade = False
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0

        capital = self.initial_capital

        trades = []

        for i in range(len(self.df)):
            price = self.df['Close'].iloc[i]

            # ENTRY
            if not in_trade and self.signals['sell_signal'].iloc[i]:
                in_trade = True
                entry_price = float(price)
                stop_loss = float(self.signals['stop_loss'].iloc[i])
                take_profit = float(self.signals['take_profit'].iloc[i])

                # 🔥 Risk management (1% per trade)
                risk_per_trade = capital * 0.01
                risk_per_share = entry_price - stop_loss

                if risk_per_share == 0:
                    continue

                position_size = risk_per_trade / risk_per_share

            # EXIT
            elif in_trade:
                row = self.df.iloc[i]
                low = float(row["Low"])
                high = float(row["High"])

                # Stop loss
                if low <= take_profit:
                    pnl = (entry_price - take_profit) * position_size
                    capital += pnl

                    trades.append({
                        'entry': entry_price,
                        'exit': take_profit,
                        'pnl': pnl,
                        'result': 'loss',
                        'capital': capital
                    })

                    in_trade = False

                # Take profit
                elif high >= stop_loss:
                    pnl = (entry_price - stop_loss) * position_size
                    capital += pnl

                    trades.append({
                        'entry': entry_price,
                        'exit': stop_loss,
                        'pnl': pnl,
                        'result': 'win',
                        'capital': capital
                    })

                    in_trade = False

        return trades, capital
    
    def plot_results(self, trades):
        df = self.df

        plt.figure(figsize=(16, 8))

        # 📈 Price
        plt.plot(df.index, df['Close'], label='Close', linewidth=1)

        # 📊 Moving averages (if they exist)
        if 'Close_MA_20' in df.columns:
            plt.plot(df.index, df['Close_MA_20'], label='MA20', linestyle='--')

        if 'Close_MA_50' in df.columns:
            plt.plot(df.index, df['Close_MA_50'], label='MA50', linestyle='--')

        # 🟢 Buy signals
        if 'buy_signal' in self.signals.columns:
            buys = df[self.signals['buy_signal']]

            plt.scatter(
                buys.index,
                buys['Close'],
                marker='^',
                label='Buy Signal'
            )

        # 🔁 Trades (entries & exits)
        for t in trades:
            entry_i = t['entry_idx']
            exit_i = t['exit_idx']

            # Entry (green triangle up)
            plt.scatter(
                df.index[entry_i],
                t['entry'],
                marker='^',
                color='green',
                s=100,
                label='Entry'
            )

            # Exit (red triangle down)
            plt.scatter(
                df.index[exit_i],
                t['exit'],
                marker='v',
                color='red',
                s=100,
                label='Exit'
            )

            # Vertical lines connecting entry and exit
            plt.axvline(df.index[entry_i], linestyle='--', alpha=0.3, color='green')
            plt.axvline(df.index[exit_i], linestyle=':', alpha=0.3, color='red')

            # Horizontal lines showing entry and exit prices
            plt.hlines(
                y=t['entry'],
                xmin=df.index[entry_i],
                xmax=df.index[exit_i],
                linestyles='--',
                alpha=0.3,
                color='green'
            )
            plt.hlines(
                y=t['exit'],
                xmin=df.index[entry_i],
                xmax=df.index[exit_i],
                linestyles=':',
                alpha=0.3,
                color='red'
            )

        # 🏷 Labels
        plt.title("Backtest Results")
        plt.xlabel("Time")
        plt.ylabel("Price")

        plt.legend()
        plt.grid()

        plt.show()

    def analyze_results(self, trades, final_capital):
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['result'] == 'win')

        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = wins / total_trades if total_trades > 0 else 0

        return_pct = (final_capital - self.initial_capital) / self.initial_capital

        print(f"Trades: {total_trades}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Final Capital: {final_capital:.2f}")
        print(f"Return: {return_pct:.2%}")

        return {
            'trades': total_trades,
            'win_rate': win_rate,
            'pnl': total_pnl,
            'return_pct': return_pct
        }


class ReliefRallyShortStrategy(BaseStrategy):

    def __init__(self, df, ma_short=20, ma_long=50, trend_period=63 ,initial_capital=10000):
        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.trend_period = trend_period
        self.signals = pd.DataFrame(index=df.index)
        self.initial_capital = initial_capital

    def is_downtrend(self, idx):

        if idx < self.trend_period:
            return False

        ma20 = self.df[f"Close_MA_{self.ma_short}"].iloc[idx-self.trend_period:idx]
        ma50 = self.df[f"Close_MA_{self.ma_long}"].iloc[idx-self.trend_period:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        strong = (
            all(ma20_diff < 0)
            and all(ma50_diff < 0)
            and ma20.iloc[-1] < ma50.iloc[-1]
        )

        weak = (
            ma20_diff.mean() < 0
            and ma50_diff.mean() < 0
            and ma20.iloc[-1] < ma50.iloc[-1]
        )

        return strong or weak

    def rally_into_ma(self, idx):

        price = self.df["Close"].iloc[idx].values[0]
        ma20 = self.df[f"Close_MA_{self.ma_short}"].iloc[idx]
        ma50 = self.df[f"Close_MA_{self.ma_long}"].iloc[idx]

        strong_condition = abs(price - ma20) / ma20 < 0.05
        weak_condition = abs(price - ma50) / ma50 < 0.05

        return strong_condition or weak_condition

    def stochastic_overbought(self, idx):

        return self.df["stoch_k"].iloc[idx] >= 75

    def bearish_candle(self, idx):

        return self.df["Close"].iloc[idx].values[0] < self.df["Open"].iloc[idx].values[0]

    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["Close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)

        risk = stop_loss - entry

        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.is_downtrend(idx)
                and self.rally_into_ma(idx)
                and self.stochastic_overbought(idx)
                and self.bearish_candle(idx)
            ):

                stop_loss, take_profit = self.risk_management(idx)

                sell_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

            else:

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals
    

class BearishDivergenceStrategy(BaseStrategy):

    def __init__(self, df, ma_short=50, ma_long=200, initial_capital=10000):

        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.signals = pd.DataFrame(index=df.index)
        self.initial_capital = initial_capital

    def long_term_downtrend(self, idx, lookback=10):

        if idx < lookback:
            return False

        ma50 = self.df[f"Close_MA_{self.ma_short}"].iloc[idx]
        ma200 = self.df[f"Close_MA_{self.ma_long}"].iloc[idx]

        ma200_series = self.df[f"Close_MA_{self.ma_long}"].iloc[idx-lookback:idx]

        ma200_diff = ma200_series.diff().dropna()

        condition1 = ma50 < ma200
        condition2 = ma200_diff.mean() < 0

        return condition1 and condition2


    def price_above_ma50(self, idx):

        return self.df["Close"].iloc[idx].values[0] > self.df[f"Close_MA_{self.ma_short}"].iloc[idx]


    def find_two_highs(self, idx, lookback=60, min_gap=5):

        highs = []

        for i in range(max(1, idx-lookback), idx):

            if (
                self.df["High"].iloc[i].values[0] > self.df["High"].iloc[i-1].values[0]
                and self.df["High"].iloc[i].values[0] > self.df["High"].iloc[i+1].values[0]
            ):
                if len(highs) == 0 or i - highs[-1] >= min_gap:
                    highs.append(i)

        if len(highs) >= 2:
            return highs[-2:]

        return None


    def divergence(self, highs):

        h1, h2 = highs

        indicators = [
            "MACD",
            "MACD_hist",
            "stoch_k",
            "obv",
            "cci",
            "rsi",
        ]

        count = 0

        for ind in indicators:

            if ind in self.df.columns:

                if self.df[ind].iloc[h2] < self.df[ind].iloc[h1]:
                    count += 1

        return count >= 2


    def bearish_candle(self, idx):

        return self.df["Close"].iloc[idx].values[0] < self.df["Open"].iloc[idx].values[0]


    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["Close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)

        risk = stop_loss - entry

        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit


    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.long_term_downtrend(idx)
                and self.price_above_ma50(idx)
            ):

                highs = self.find_two_highs(idx)

                if highs and self.divergence(highs):

                    if self.bearish_candle(idx):

                        stop_loss, take_profit = self.risk_management(idx)

                        sell_signals.append(True)
                        stop_losses.append(stop_loss)
                        take_profits.append(take_profit)

                        continue

            sell_signals.append(False)
            stop_losses.append(np.nan)
            take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals


class GapDownShortStrategy(BaseStrategy):

    def __init__(self, df, ma=50, initial_capital=10000):

        self.df = df.copy()
        self.ma = ma
        self.signals = pd.DataFrame(index=df.index)
        self.initial_capital = initial_capital

    def steady_uptrend(self, idx, lookback=40):

        if idx < lookback:
            return False

        ma_series = self.df[f"Close_MA_{self.ma}"].iloc[idx-lookback:idx]

        ma_diff = ma_series.diff().dropna()

        price_series = self.df["Close"].iloc[idx-lookback:idx]

        ma_rising = ma_diff.mean() > 0
        #price_above_ma = all(price_series > ma_series)

        return ma_rising #and price_above_ma

    def gap_down(self, idx):

        if idx < 1:
            return False

        today_high = self.df["High"].iloc[idx].values[0]
        prev_low = self.df["Low"].iloc[idx-1].values[0]

        return today_high < prev_low

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        gap_low = None
        prev_low = None
        active_setup = False

        for idx in range(len(self.df)):

            if self.steady_uptrend(idx) and self.gap_down(idx):

                gap_low = self.df["Low"].iloc[idx].values[0]
                prev_low = self.df["Low"].iloc[idx-1].values[0]
                active_setup = True

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

                continue

            if active_setup:

                price = self.df["Close"].iloc[idx].values[0]

                # cancel setup if gap filled
                if price >= prev_low:
                    active_setup = False

                # entry trigger
                elif price < gap_low:

                    entry = price
                    stop_loss = entry * 1.07
                    risk = stop_loss - entry
                    take_profit = entry - risk * 1.75

                    sell_signals.append(True)
                    stop_losses.append(stop_loss)
                    take_profits.append(take_profit)

                    active_setup = False
                    continue

            sell_signals.append(False)
            stop_losses.append(np.nan)
            take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals


class BlueSeaBreakdownStrategy(BaseStrategy):

    def __init__(self, df, initial_capital=10000):

        self.df = df.copy()
        self.signals = pd.DataFrame(index=df.index)
        self.initial_capital = initial_capital

    def pivot_low_recent(self, idx, pivot_window=20):

        if idx < pivot_window:
            return False

        prev_low = self.df["Close"].iloc[idx-pivot_window:idx].min().values[0]

        return self.df["Close"].iloc[idx].values[0] < prev_low

    def three_month_low(self, idx, lookback=63):

        if idx < lookback:
            return False

        prev_lows = self.df["Close"].iloc[idx-lookback:idx]

        return self.df["Close"].iloc[idx].values[0] < prev_lows.min().values[0]

    def not_overextended(self, idx, lookback=252):

        if idx < lookback:
            return True  # allow if not enough data

        high_52w = self.df["High"].iloc[idx-lookback:idx].max().values[0]
        new_low = self.df["Close"].iloc[idx].values[0]

        ratio = high_52w / new_low

        return ratio <= 2.0

    def obv_confirmation(self, idx, lookback=63):

        if idx < lookback:
            return False

        prev_obv = self.df["obv"].iloc[idx-lookback:idx]

        return self.df["obv"].iloc[idx] <= prev_obv.min()

    def bearish_candle(self, idx):

        return self.df["Close"].iloc[idx].values[0] < self.df["Open"].iloc[idx].values[0]

    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["Close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)

        risk = stop_loss - entry

        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.pivot_low_recent(idx)
                and self.three_month_low(idx)
                and self.not_overextended(idx)
                and self.obv_confirmation(idx)
                and self.bearish_candle(idx)
            ):

                stop_loss, take_profit = self.risk_management(idx)

                sell_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

            else:

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals
    
class RisingWedgeBreakdownStrategy(BaseStrategy):

    def __init__(self, df, ma_short=20, ma_long=50, initial_capital=10000):

        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.signals = pd.DataFrame(index=df.index)
        self.initial_capital = initial_capital

    def uptrend(self, idx, lookback=30):

        if idx < lookback:
            return False

        ma20 = self.df[f"Close_MA_{self.ma_short}"].iloc[idx-lookback:idx]
        ma50 = self.df[f"Close_MA_{self.ma_long}"].iloc[idx-lookback:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        strong = all(ma20_diff > 0) and all(ma50_diff > 0) and ma20.iloc[-1] > ma50.iloc[-1]
        weak = ma20_diff.mean() > 0 and ma20.iloc[-1] > ma50.iloc[-1]

        return strong or weak

    def rising_wedge(self, idx, window=20):

        if idx < window:
            return False

        highs = self.df["High"].iloc[idx-window:idx]
        lows = self.df["Low"].iloc[idx-window:idx]

        higher_highs = highs.iloc[-1].values[0] > highs.iloc[0].values[0]
        higher_lows = lows.iloc[-1].values[0] > lows.iloc[0].values[0]

        ranges = highs.values[0] - lows.values[0]
        tightening = ranges[-1] < ranges[0]

        return higher_highs and higher_lows and tightening

    def macd_lower_highs(self, idx, window=20):

        if idx < window:
            return False

        macd = self.df["MACD"].iloc[idx-window:idx]

        peaks = macd.nlargest(3)

        if len(peaks) < 3:
            return False

        return peaks.iloc[0] > peaks.iloc[1] > peaks.iloc[2]

    def obv_breakdown(self, idx, window=20):

        if idx < window:
            return False

        obv_series = self.df["obv"].iloc[idx-window:idx]

        return self.df["obv"].iloc[idx] < obv_series.min()

    def bearish_candle(self, idx):

        return self.df["Close"].iloc[idx] < self.df["Open"].iloc[idx]

    def risk_management(self, idx, stop_distance=0.07, reward_ratio=1.75):

        entry = self.df["close"].iloc[idx]

        stop_loss = entry * (1 + stop_distance)
        risk = stop_loss - entry
        take_profit = entry - risk * reward_ratio

        return stop_loss, take_profit

    def generate_signals(self):

        sell_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):

            if (
                self.uptrend(idx)
                and self.rising_wedge(idx)
                and self.obv_breakdown(idx)
                and self.bearish_candle(idx)
            ):

                stop_loss, take_profit = self.risk_management(idx)

                sell_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)

            else:

                sell_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals["sell_signal"] = sell_signals
        self.signals["stop_loss"] = stop_losses
        self.signals["take_profit"] = take_profits

        return self.signals
