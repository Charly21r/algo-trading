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
            if not in_trade and self.signals['buy_signal'].iloc[i]:
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
                if low <= stop_loss:
                    pnl = (stop_loss - entry_price) * position_size
                    capital += pnl

                    trades.append({
                        'entry': entry_price,
                        'exit': stop_loss,
                        'pnl': pnl,
                        'result': 'loss',
                        'capital': capital
                    })

                    in_trade = False

                # Take profit
                elif high >= take_profit:
                    pnl = (take_profit - entry_price) * position_size
                    capital += pnl

                    trades.append({
                        'entry': entry_price,
                        'exit': take_profit,
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

class PullbackStrategy(BaseStrategy):
    def __init__(self, df, ma_short=20, ma_long=50, trend_lookback=63,initial_capital=10000):
        super().__init__(df, initial_capital)
        """
        df: DataFrame with columns: open, high, low, close, stoch_k
        ma_short: 20 MA
        ma_long: 50 MA
        trend_lookback: ~3 months (63 trading days)
        """
        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.trend_lookback = trend_lookback
        self.signals = pd.DataFrame(index=df.index)

    def is_uptrend(self, idx):
        """Check if stock is in uptrend for last 3 months"""
        if idx < self.trend_lookback:
            return False
        lows = self.df['Low'].iloc[idx-self.trend_lookback:idx]
        ma20 = self.df[f'Close_MA_{self.ma_short}'].iloc[idx-self.trend_lookback:idx]
        ma50 = self.df[f'Close_MA_{self.ma_long}'].iloc[idx-self.trend_lookback:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        strong = all(ma20_diff > 0) and all(ma50_diff > 0) and ma20.iloc[-1] > ma50.iloc[-1]
        weak = (ma20_diff.mean() > 0) and (ma50_diff.mean() > 0) and ma20.iloc[-1] > ma50.iloc[-1]
        #higher_lows = all(x < y for x, y in zip(lows[:-1], lows[1:]))

        return (strong or weak)

    def pullback_to_ma(self, idx):
        """Check price near 20 MA, between 20 & 50, or near 50 MA"""
        price = self.df['Close'].iloc[idx]
        ma20 = self.df[f'Close_MA_{self.ma_short}'].iloc[idx]
        ma50 = self.df[f'Close_MA_{self.ma_long}'].iloc[idx]

        return ma20 <= price.values[0] <= ma50 or abs(price.values[0] - ma20)/ma20 < 0.05 or abs(price.values[0] - ma50)/ma50 < 0.05

    def stochastic_oversold(self, idx):
        """Check if stochastic %K <= 20"""
        return self.df['stoch_k'].iloc[idx] <= 25

    def bullish_candle(self, idx):
        """Check for bullish reversal candle (simplified as close > open)"""
        return self.df['Close'].iloc[idx].values[0] > self.df['Open'].iloc[idx].values[0]

    def calculate_risk_reward(self, idx, stop_distance=0.07, reward_ratio=1.75):
        """Stop-loss 7% below entry, take-profit 1.75 x risk"""
        entry_price = self.df['Close'].iloc[idx]
        stop_loss = entry_price * (1 - stop_distance)
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * reward_ratio
        return stop_loss, take_profit

    def generate_signals(self):
        buy_signals = [False] * len(self.df)
        stop_losses = [np.nan] * len(self.df)
        take_profits = [np.nan] * len(self.df)

        for idx in range(len(self.df)):

            if (self.is_uptrend(idx) and
                self.pullback_to_ma(idx) and
                self.stochastic_oversold(idx) and
                self.bullish_candle(idx)):

                stop_loss, take_profit = self.calculate_risk_reward(idx)

                buy_signals[idx] = True
                stop_losses[idx] = stop_loss
                take_profits[idx] = take_profit

        self.signals['buy_signal'] = buy_signals
        self.signals['stop_loss'] = stop_losses
        self.signals['take_profit'] = take_profits

        return self.signals

class CoiledSpringStrategy(BaseStrategy):
    def __init__(self, df, ma_short=20, ma_long=50,initial_capital=10000):
        super().__init__(df, initial_capital)
        """
        df: DataFrame with columns: open, high, low, close
        ma_short: 20 MA
        ma_long: 50 MA
        """
        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.signals = pd.DataFrame(index=df.index)

    def check_uptrend(self, idx, lookback=20):
        """Check if stock is weakly or strongly uptrending"""
        if idx < lookback:
            return False
        ma20 = self.df[f'Close_MA_{self.ma_short}'].iloc[idx-lookback:idx]
        ma50 = self.df[f'Close_MA_{self.ma_long}'].iloc[idx-lookback:idx]
        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()
        # Strong: 20 above 50 and rising
        strong = (ma20.iloc[-1] > ma50.iloc[-1]) and (ma20_diff.mean() > 0) and (ma50_diff.mean() > 0)
        # Weak: 20 mostly above 50 but may fluctuate
        weak = (ma20.iloc[-1] > ma50.iloc[-1]) and (ma50_diff.mean() < 0)
        return strong or weak

    def recent_high(self, idx, high_lookback=20, three_month=63):
        """New high within last 20 days and 3-month high"""
        if idx < high_lookback or idx < three_month:
            return False
        recent_max = self.df['High'].iloc[idx-high_lookback:idx].max()
        three_month_max = self.df['High'].iloc[idx-three_month:idx].max()
        return (self.df['High'].iloc[idx].values[0] > recent_max.values[0]) and (self.df['High'].iloc[idx].values[0] > three_month_max.values[0])

    def coil_range_check(self, idx, min_days=7, max_days=20):
        """Check coiled spring narrowing range over last 7–20 days"""
        for days in range(min_days, max_days+1):
            if idx < days:
                continue
            window = self.df['High'].iloc[idx-days:idx], self.df['Low'].iloc[idx-days:idx]
            highs, lows = window
            # Range must be narrowing
            range_widths = highs.values - lows.values
            if all(x >= y for x, y in zip(range_widths[:-1], range_widths[1:])):
                # Must not touch 50 MA
                ma50_max = self.df[f'Close_MA_{self.ma_long}'].iloc[idx-days:idx].max()
                if highs.max() < ma50_max:
                    return True
        return False

    def generate_signals(self):
        buy_signals = []

        for idx in range(len(self.df)):
            if (self.check_uptrend(idx) and
                self.recent_high(idx) and
                self.coil_range_check(idx)):

                buy_signals.append(True)
            else:
                buy_signals.append(False)

        self.signals['buy_signal'] = buy_signals
        return self.signals

class BullishDivergenceStrategy(BaseStrategy):
    def __init__(self, df, ma_short=50, ma_long=200,initial_capital=10000):
        super().__init__(df, initial_capital)
        """
        df: DataFrame with indicators already calculated:
        close_MA_50, Close_MA_200, MACD, stoch_k, rsi, obv, cci, open, high, low, close
        """
        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.signals = pd.DataFrame(index=df.index)

    def trend_filter(self, idx, trend_period=10):
        """Check trend has held for last trend_period days"""
        if idx < trend_period:
            return False
        for i in range(idx-trend_period+1, idx+1):
            ma_50 = self.df[f'Close_MA_{self.ma_short}'].iloc[i]
            ma_200 = self.df[f'Close_MA_{self.ma_long}'].iloc[i]
            ma_200_slope = self.df[f'Close_MA_{self.ma_long}'].iloc[i] - self.df[f'Close_MA_{self.ma_long}'].iloc[i-1]
            price = self.df['Close'].iloc[i].values[0]
            if not (ma_50 > ma_200 and ma_200_slope > 0 and price < ma_50):
                return False
        return True

    def find_recent_lows(self, idx, lookback=60, min_gap=5):
        """Find at least two lows separated by min_gap days"""
        lows = []
        for i in range(max(0, idx-lookback), idx+1):
            if i == 0 or self.df['Low'].iloc[i].values[0] < self.df['Low'].iloc[i-1].values[0] and self.df['Low'].iloc[i].values[0] < self.df['Low'].iloc[min(i+1, idx)].values[0]:
                if len(lows) == 0 or i - lows[-1] >= min_gap:
                    lows.append(i)
        if len(lows) >= 2:
            return lows[-2:]  # last two lows
        return None

    def check_divergence(self, low_indices):
        """Check last low corresponds to higher low in at least 2 indicators"""
        idx1, idx2 = low_indices
        indicators = ['MACD', 'stoch_k', 'rsi', 'obv', 'cci']
        count = 0
        for ind in indicators:
            val1 = self.df[ind].iloc[idx1]
            val2 = self.df[ind].iloc[idx2]
            if val2 > val1:
                count += 1
        return count >= 2

    def bullish_candle(self, idx):
        """Check last candle bullish"""
        if idx < 1:
            return False
        prev = self.df.iloc[idx-1]
        curr = self.df.iloc[idx]
        body = abs(prev['Close'] - prev['Open'])
        candle_range = prev['High'] - prev['Low']
        lower_wick = prev['Open'] - prev['Low'] if prev['Close'].values[0] > prev['Open'].values[0] else prev['Close'] - prev['Low']
        hammer = lower_wick.values[0] > 2*body.values[0] and body.values[0]/candle_range.values[0] < 0.4 and prev['Close'].values[0] > prev['Open'].values[0]
        engulfing = curr['Close'].values[0] > curr['Open'].values[0] and curr['Open'].values[0] < prev['Close'].values[0] and curr['Close'].values[0] > prev['Open'].values[0]
        return hammer or engulfing

    def calculate_risk_reward(self, idx, last_low_idx):
        stop_loss = self.df['Low'].iloc[last_low_idx]
        entry_price = self.df['Close'].iloc[idx]
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * 1.75
        return stop_loss, take_profit

    def generate_signals(self):
        buy_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):
            if self.trend_filter(idx):
                lows = self.find_recent_lows(idx)
                if lows and self.check_divergence(lows):
                    if self.bullish_candle(idx):
                        stop_loss, take_profit = self.calculate_risk_reward(idx, lows[1])
                        buy_signals.append(True)
                        stop_losses.append(stop_loss)
                        take_profits.append(take_profit)
                        continue
            buy_signals.append(False)
            stop_losses.append(np.nan)
            take_profits.append(np.nan)

        self.signals['buy_signal'] = buy_signals
        self.signals['stop_loss'] = stop_losses
        self.signals['take_profit'] = take_profits
        return self.signals


class BlueSkyBreakoutStrategy(BaseStrategy):
    def __init__(self, df,initial_capital=10000):
        super().__init__(df, initial_capital)
        """
        df: DataFrame with columns:
        open, high, low, close, obv
        """
        self.df = df.copy()
        self.signals = pd.DataFrame(index=df.index)

    def is_breakout(self, idx, lookback=20):
        """Price exceeds the previous 20 highs"""
        if idx < lookback:
            return False
        prev_highs = self.df['High'].iloc[idx-lookback:idx]
        return (self.df['Close'].iloc[idx].values[0] > prev_highs.max().values[0]) or (self.df['High'].iloc[idx].values[0] > prev_highs.max().values[0])

    def is_three_month_high(self, idx, lookback=63):
        """High is the highest in last 3 months"""
        if idx < lookback:
            return False
        recent_high = self.df['High'].iloc[idx-lookback:idx].max().values[0]
        return self.df['High'].iloc[idx].values[0] > recent_high

    def not_too_far_from_low(self, idx, low_lookback=252):
        """Ensure new high is not too far from 52-week low"""
        if idx < low_lookback:
            low_window = self.df['Low'].iloc[:idx+1]
        else:
            low_window = self.df['Low'].iloc[idx-low_lookback:idx+1]
        low_52w = low_window.min().values[0]
        ratio = self.df['High'].iloc[idx].values[0] / low_52w
        return ratio < 3.0

    def obv_confirmation(self, idx, lookback=63):
        """OBV is highest in last 3 months"""
        if idx < lookback:
            return False
        recent_obv = self.df['obv'].iloc[idx-lookback:idx]
        return self.df['obv'].iloc[idx] > recent_obv.max()

    def bullish_candle(self, idx):
        """Current candle is green"""
        return self.df['Close'].iloc[idx].values[0] > self.df['Open'].iloc[idx].values[0]

    def calculate_risk_reward(self, idx, stop_distance=0.07, reward_ratio=1.75):
        """Stop-loss 7% below entry, take-profit 1.75× risk"""
        entry_price = self.df['Close'].iloc[idx]
        stop_loss = entry_price * (1 - stop_distance)
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * reward_ratio
        return stop_loss, take_profit

    def generate_signals(self):
        buy_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):
            if (self.is_breakout(idx) and
                self.is_three_month_high(idx) and
                self.not_too_far_from_low(idx) and
                self.obv_confirmation(idx) and
                self.bullish_candle(idx)):

                stop_loss, take_profit = self.calculate_risk_reward(idx)
                buy_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)
            else:
                buy_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals['buy_signal'] = buy_signals
        self.signals['stop_loss'] = stop_losses
        self.signals['take_profit'] = take_profits
        return self.signals
    

class BullishBaseBreakout(BaseStrategy):
    def __init__(self, df, ma_short=20, ma_long=50, base_lookback=30, downtrend_lookback=63, obv_ma_period=20,initial_capital=10000):
        super().__init__(df, initial_capital)
        """
        df: DataFrame with columns: open, high, low, close, MACD, MACD_signal, obv
        ma_short: period for 20 MA
        ma_long: period for 50 MA
        base_lookback: number of days for consolidation base (~30)
        downtrend_lookback: period to define downtrend (~63)
        obv_ma_period: period for OBV trendline
        """
        self.df = df.copy()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.base_lookback = base_lookback
        self.downtrend_lookback = downtrend_lookback
        self.obv_ma_period = obv_ma_period
        self.signals = pd.DataFrame(index=df.index)

    def is_downtrend(self, idx):
        """Check if stock is in uptrend for last 3 months"""
        if idx < self.downtrend_lookback:
            return False
        lows = self.df['Low'].iloc[idx-self.downtrend_lookback:idx]
        ma20 = self.df[f'Close_MA_{self.ma_short}'].iloc[idx-self.downtrend_lookback:idx]
        ma50 = self.df[f'Close_MA_{self.ma_long}'].iloc[idx-self.downtrend_lookback:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        strong = all(ma20_diff < 0) and all(ma50_diff < 0) and ma20.iloc[-1] < ma50.iloc[-1]
        weak = (ma20_diff.mean() < 0) and (ma50_diff.mean() < 0) and ma20.iloc[-1] < ma50.iloc[-1]
        #higher_lows = all(x < y for x, y in zip(lows[:-1], lows[1:]))

        return (strong or weak)

    def is_strong_or_weak_downtrend(self, idx):
        """Check 20 and 50 MA behavior for strong or weak downtrend"""
        if idx < self.ma_long + self.downtrend_lookback:
            return False
        ma20 = self.df[f'Close_MA_{self.ma_short}'].iloc[idx-self.downtrend_lookback:idx]
        ma50 = self.df[f'Close_MA_{self.ma_long}'].iloc[idx-self.downtrend_lookback:idx]

        ma20_diff = ma20.diff().dropna()
        ma50_diff = ma50.diff().dropna()

        # Strong downtrend: 20 MA consistently falling & below falling 50 MA
        strong = (ma20_diff.mean() < 0) and (ma20.iloc[-1] < ma50.iloc[-1]) and (ma50_diff.mean() < 0)
        # Weak downtrend: 20 MA may fluctuate but mostly below falling 50 MA
        weak = (ma20.iloc[-1] < ma50.iloc[-1]) and (ma50_diff.mean() < 0)
        return strong or weak

    def in_consolidation_base(self, idx):
        """Check price is in a base for last base_lookback days"""
        if idx < self.base_lookback:
            return False
        window = self.df['Close'].iloc[idx-self.base_lookback:idx].values[0]
        # Simple volatility check (flat or falling rectangle)
        range_pct = (window.max() - window.min()) / window.mean()
        return range_pct <= 0.08  # 8% threshold for base

    def macd_higher_lows(self, idx):
        """Check MACD making higher lows within base"""
        if idx < self.base_lookback:
            return False
        macd_window = self.df['MACD'].iloc[idx-self.base_lookback:idx]
        return all(x < y for x, y in zip(macd_window, macd_window[1:]))

    def obv_breakout(self, idx):
        """Check OBV breaks above trendline (moving average)"""
        if idx < self.obv_ma_period:
            return False
        obv_ma = self.df['obv'].iloc[idx-self.obv_ma_period:idx].mean()
        return self.df['obv'].iloc[idx] > obv_ma

    def bullish_candle(self, idx):
        """Green candle"""
        return self.df['Close'].iloc[idx].values[0] > self.df['Open'].iloc[idx].values[0]

    def calculate_risk_reward(self, idx):
        """Stop-loss below base low, take-profit 1.75 x risk"""
        base_low = self.df['Low'].iloc[idx-self.base_lookback:idx].min()
        entry_price = self.df['Close'].iloc[idx]
        risk = entry_price - base_low
        stop_loss = base_low
        take_profit = entry_price + risk * 1.75
        return stop_loss, take_profit

    def generate_signals(self):
        buy_signals = []
        stop_losses = []
        take_profits = []

        for idx in range(len(self.df)):
            if (self.is_downtrend(idx) and
                self.is_strong_or_weak_downtrend(idx) and
                self.in_consolidation_base(idx) and
                self.obv_breakout(idx) and
                self.bullish_candle(idx)):

                stop_loss, take_profit = self.calculate_risk_reward(idx)
                buy_signals.append(True)
                stop_losses.append(stop_loss)
                take_profits.append(take_profit)
            else:
                buy_signals.append(False)
                stop_losses.append(np.nan)
                take_profits.append(np.nan)

        self.signals['buy_signal'] = buy_signals
        self.signals['stop_loss'] = stop_losses
        self.signals['take_profit'] = take_profits
        return self.signals