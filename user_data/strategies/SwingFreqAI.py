"""
crytrd swing signal strategy (Phase 1).

The model answers one question per pair and candle:
    P(price touches >= LABEL_MOVE above the close within the next 72h)
Entries fire only when that probability clears a strict threshold. The bot runs in
dry-run only; entries are paper trades whose Telegram messages act as the signals.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.enums import RunMode
from freqtrade.persistence import Order, Trade
from freqtrade.strategy import DecimalParameter, IStrategy, stoploss_from_absolute


logger = logging.getLogger(__name__)

# Trade geometry. TARGET_NET is the paper profit after fees; the label asks for the
# gross move needed to get there.
TARGET_NET = 0.05
ROUND_TRIP_COST = 0.006  # 2 x config "fee" (0.3% placeholder for CoinDCX fee + slippage)
LABEL_MOVE = TARGET_NET + ROUND_TRIP_COST
MAX_HOLD_HOURS = 72
ATR_STOP_MULT = 2.0
MIN_STOP_PCT = 0.02
MAX_STOP_PCT = 0.10

REFERENCE_PAIR = "BTC/USDT"


def _fmt_price(x: float) -> str:
    return np.format_float_positional(x, precision=5, unique=False, fractional=False, trim="-")


def _stop_distance(close: pd.Series | float, atr: pd.Series | float):
    return np.clip(ATR_STOP_MULT * atr / close, MIN_STOP_PCT, MAX_STOP_PCT)


class SwingFreqAI(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 200

    # Take profit via ROI so backtests fill on the candle high; time and ATR stops below.
    minimal_roi = {"0": TARGET_NET}
    stoploss = -MAX_STOP_PCT
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = False

    entry_threshold = DecimalParameter(0.50, 0.85, default=0.65, decimals=2, space="buy")

    plot_config = {
        "main_plot": {},
        "subplots": {
            "P(swing)": {"swing": {"color": "#2ca02c"}},
            "do_predict": {"do_predict": {"color": "#7f7f7f"}},
        },
    }

    # ---------------------------------------------------------------- features

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: dict, **kwargs) -> DataFrame:
        close = dataframe["close"]
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.0)
        width = bb["upper"] - bb["lower"]
        dataframe["%-bb_pos-period"] = (close - bb["lower"]) / width
        dataframe["%-bb_width-period"] = width / bb["mid"]

        vol = dataframe["volume"]
        dataframe["%-vol_z-period"] = (vol - vol.rolling(period).mean()) / vol.rolling(period).std()

        dataframe["%-dist_high-period"] = close / dataframe["high"].rolling(period).max() - 1
        dataframe["%-dist_low-period"] = close / dataframe["low"].rolling(period).min() - 1
        return dataframe.replace([np.inf, -np.inf], np.nan)

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict,
                                         **kwargs) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-range"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe["%-body"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict,
                                     **kwargs) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"] / dataframe["close"]
        dataframe["%-macdhist"] = macd["macdhist"] / dataframe["close"]

        # Strength relative to BTC over 1, 3 and 7 days.
        btc = self.dp.get_pair_dataframe(REFERENCE_PAIR, self.timeframe)
        if not btc.empty:
            btc_close = dataframe["date"].map(btc.set_index("date")["close"])
            for n in (24, 72, 168):
                own = dataframe["close"].pct_change(n)
                ref = btc_close.pct_change(n, fill_method=None)
                dataframe[f"%-rel_btc_{n}"] = own - ref
        return dataframe

    # ------------------------------------------------------------------ target

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        self.freqai.class_names = ["no_swing", "swing"]
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&s-swing"] = self._label(dataframe, horizon)
        return dataframe

    @staticmethod
    def _label(dataframe: DataFrame, horizon: int) -> pd.Series:
        """'swing' if the max high over the next `horizon` candles is >= LABEL_MOVE above
        the close. NaN where the future is not yet known, so FreqAI drops those rows."""
        future_max = dataframe["high"].rolling(horizon).max().shift(-horizon)
        move = future_max / dataframe["close"] - 1
        label = pd.Series(np.where(move >= LABEL_MOVE, "swing", "no_swing"),
                          index=dataframe.index, dtype=object)
        label[move.isna()] = np.nan
        return label

    # -------------------------------------------------------------- indicators

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        if self.dp.runmode == RunMode.BACKTEST:
            self._dump_predictions(dataframe, metadata["pair"])
        return dataframe

    def _dump_predictions(self, dataframe: DataFrame, pair: str) -> None:
        """Save walk-forward probabilities plus the realised label, for
        user_data/scripts/evaluate_signals.py (precision vs base rate)."""
        if "swing" not in dataframe.columns:
            return
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        out = dataframe[["date", "close", "swing", "do_predict"]].copy()
        out["label"] = self._label(dataframe, horizon)
        out["pair"] = pair
        folder = Path(self.config["user_data_dir"]) / "backtest_predictions"
        folder.mkdir(parents=True, exist_ok=True)
        out.reset_index(drop=True).to_feather(folder / f"{pair.replace('/', '_')}.feather")

    # ------------------------------------------------------------------ signals

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if "swing" not in dataframe.columns:
            return dataframe

        signal = (
            (dataframe["do_predict"] == 1)
            & (dataframe["swing"] >= self.entry_threshold.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[signal, "enter_long"] = 1
        if signal.any():
            dataframe.loc[signal, "enter_tag"] = dataframe.loc[signal].apply(self._setup_tag, axis=1)
        return dataframe

    @staticmethod
    def _setup_tag(row: pd.Series) -> str:
        close, atr = row["close"], row["atr"]
        zone_low = close - 0.5 * atr
        stop = close * (1 - _stop_distance(close, atr))
        target = close * (1 + LABEL_MOVE)
        return (f"p={row['swing']:.2f} | zone {_fmt_price(zone_low)}-{_fmt_price(close)}"
                f" | SL {_fmt_price(stop)} | TP {_fmt_price(target)} | hold<={MAX_HOLD_HOURS}h")

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    # ------------------------------------------------------------ trade mgmt

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime,
                     **kwargs) -> None:
        # Fix the ATR stop at entry; the analyzed frame's last row is the signal candle.
        if order.ft_order_side != trade.entry_side or trade.nr_of_successful_entries != 1:
            return
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or pd.isna(dataframe["atr"].iloc[-1]):
            return
        dist = float(_stop_distance(trade.open_rate, dataframe["atr"].iloc[-1]))
        trade.set_custom_data("stop_price", trade.open_rate * (1 - dist))

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:
        stop_price = trade.get_custom_data("stop_price")
        if stop_price is None:
            return None
        return stoploss_from_absolute(stop_price, current_rate, is_short=trade.is_short,
                                      leverage=trade.leverage)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> str | None:
        if current_time - trade.open_date_utc >= timedelta(hours=MAX_HOLD_HOURS):
            return "time_exit"
        return None
