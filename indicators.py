from __future__ import annotations

from typing import Any


def sma(values: list[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: list[float], period: int) -> float | None:
    if period <= 0 or len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema_value = sum(values[:period]) / period
    for value in values[period:]:
        ema_value = (value - ema_value) * multiplier + ema_value
    return ema_value


def calculate_rsi(closes: list[float], period: int) -> float:
    if period <= 0:
        raise ValueError("Period must be positive.")
    if len(closes) < period + 1:
        raise ValueError("Not enough data to calculate RSI.")

    gains = []
    losses = []
    for index in range(1, period + 1):
        delta = closes[index] - closes[index - 1]
        gains.append(delta if delta > 0 else 0.0)
        losses.append(abs(delta) if delta < 0 else 0.0)

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    for index in range(period + 1, len(closes)):
        delta = closes[index] - closes[index - 1]
        gain = delta if delta > 0 else 0.0
        loss = abs(delta) if delta < 0 else 0.0
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period

    if average_loss == 0:
        return 100.0

    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def macd(closes: list[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[float, float, float] | None:
    if len(closes) < slow_period + signal_period:
        return None
    macd_series: list[float] = []
    for idx in range(slow_period, len(closes) + 1):
        chunk = closes[:idx]
        fast = ema(chunk, fast_period)
        slow = ema(chunk, slow_period)
        if fast is None or slow is None:
            continue
        macd_series.append(fast - slow)
    signal = ema(macd_series, signal_period)
    if signal is None:
        return None
    macd_line = macd_series[-1]
    hist = macd_line - signal
    return macd_line, signal, hist


def stochastic_kd(highs: list[float], lows: list[float], closes: list[float], k_period: int = 14, d_period: int = 3, smooth: int = 3) -> tuple[float, float] | None:
    needed = k_period + smooth + d_period
    if len(closes) < needed or len(highs) < needed or len(lows) < needed:
        return None

    raw_k: list[float] = []
    for idx in range(k_period - 1, len(closes)):
        window_high = max(highs[idx - k_period + 1 : idx + 1])
        window_low = min(lows[idx - k_period + 1 : idx + 1])
        den = window_high - window_low
        if den == 0:
            raw_k.append(50.0)
        else:
            raw_k.append((closes[idx] - window_low) / den * 100)

    if len(raw_k) < smooth:
        return None
    smooth_k: list[float] = []
    for idx in range(smooth - 1, len(raw_k)):
        smooth_k.append(sum(raw_k[idx - smooth + 1 : idx + 1]) / smooth)
    if len(smooth_k) < d_period:
        return None
    d_val = sum(smooth_k[-d_period:]) / d_period
    return smooth_k[-1], d_val


def bollinger_percent_b(closes: list[float], period: int = 20, stddev: float = 2.0) -> float | None:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    sd = variance ** 0.5
    upper = mid + stddev * sd
    lower = mid - stddev * sd
    if upper == lower:
        return None
    return (closes[-1] - lower) / (upper - lower)


def adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for idx in range(1, len(closes)):
        up_move = highs[idx] - highs[idx - 1]
        down_move = lows[idx - 1] - lows[idx]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr = max(highs[idx] - lows[idx], abs(highs[idx] - closes[idx - 1]), abs(lows[idx] - closes[idx - 1]))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    plus = sum(plus_dm[:period]) / period
    minus = sum(minus_dm[:period]) / period
    dx_values: list[float] = []
    for idx in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[idx]) / period
        plus = (plus * (period - 1) + plus_dm[idx]) / period
        minus = (minus * (period - 1) + minus_dm[idx]) / period
        if atr == 0:
            continue
        plus_di = 100 * (plus / atr)
        minus_di = 100 * (minus / atr)
        den = plus_di + minus_di
        dx = 0.0 if den == 0 else 100 * abs(plus_di - minus_di) / den
        dx_values.append(dx)
    if not dx_values:
        return None
    return sum(dx_values[-period:]) / min(period, len(dx_values))


def ultimate_oscillator(highs: list[float], lows: list[float], closes: list[float], p1: int = 7, p2: int = 14, p3: int = 28) -> float | None:
    if len(closes) < p3 + 1:
        return None
    bp: list[float] = []
    tr: list[float] = []
    for idx in range(1, len(closes)):
        low = min(lows[idx], closes[idx - 1])
        high = max(highs[idx], closes[idx - 1])
        bp.append(closes[idx] - low)
        tr.append(high - low)

    def avg(period: int) -> float | None:
        if len(bp) < period:
            return None
        tr_sum = sum(tr[-period:])
        if tr_sum == 0:
            return None
        return sum(bp[-period:]) / tr_sum

    a1 = avg(p1)
    a2 = avg(p2)
    a3 = avg(p3)
    if a1 is None or a2 is None or a3 is None:
        return None
    return 100 * (4 * a1 + 2 * a2 + a3) / 7


def parabolic_sar(highs: list[float], lows: list[float], step: float = 0.02, max_step: float = 0.2) -> float | None:
    if len(highs) < 2 or len(lows) < 2:
        return None
    bull = True
    sar = lows[0]
    ep = highs[0]
    af = step
    for idx in range(1, len(highs)):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, lows[idx - 1], lows[idx])
            if lows[idx] < sar:
                bull = False
                sar = ep
                ep = lows[idx]
                af = step
            else:
                if highs[idx] > ep:
                    ep = highs[idx]
                    af = min(max_step, af + step)
        else:
            sar = max(sar, highs[idx - 1], highs[idx])
            if highs[idx] > sar:
                bull = True
                sar = ep
                ep = highs[idx]
                af = step
            else:
                if lows[idx] < ep:
                    ep = lows[idx]
                    af = min(max_step, af + step)
    return sar


def mfi(highs: list[float], lows: list[float], closes: list[float], volumes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1 or len(volumes) < period + 1:
        return None
    positive = 0.0
    negative = 0.0
    for idx in range(len(closes) - period, len(closes)):
        tp = (highs[idx] + lows[idx] + closes[idx]) / 3
        prev_tp = (highs[idx - 1] + lows[idx - 1] + closes[idx - 1]) / 3
        flow = tp * volumes[idx]
        if tp > prev_tp:
            positive += flow
        elif tp < prev_tp:
            negative += flow
    if negative == 0:
        return 100.0
    money_ratio = positive / negative
    return 100 - (100 / (1 + money_ratio))


def cci(highs: list[float], lows: list[float], closes: list[float], period: int = 20) -> float | None:
    if len(closes) < period:
        return None
    tps = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes) - period, len(closes))]
    sma_tp = sum(tps) / period
    mean_dev = sum(abs(tp - sma_tp) for tp in tps) / period
    if mean_dev == 0:
        return None
    return (tps[-1] - sma_tp) / (0.015 * mean_dev)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def fetch_ohlcv_closes(exchange, symbol: str, timeframe: str, limit: int) -> list[float]:
    ohlcv = fetch_ohlcv(exchange, symbol, timeframe, limit)
    return [float(candle[4]) for candle in ohlcv]


def _passes_condition(value: float | None, mode: str, first: float, second: float | None = None) -> bool:
    if value is None:
        return False
    if mode == "<":
        return value < first
    if mode == ">":
        return value > first
    if mode == "between" and second is not None:
        low = min(first, second)
        high = max(first, second)
        return low <= value <= high
    return False


def _to_float(data: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _to_int(data: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate_entry_filters_with_reason(exchange, symbol: str, filters: dict[str, Any]) -> tuple[bool, str]:
    if not filters:
        return True, "no filters"
    timeframe = str(filters.get("timeframe", "15m"))
    ohlcv = fetch_ohlcv(exchange, symbol, timeframe, 250)
    if len(ohlcv) < 50:
        return False, "not enough candles"

    highs = [float(x[2]) for x in ohlcv]
    lows = [float(x[3]) for x in ohlcv]
    closes = [float(x[4]) for x in ohlcv]
    volumes = [float(x[5]) for x in ohlcv]
    price = closes[-1]

    for name, cfg in filters.items():
        if name == "timeframe":
            continue
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            continue
        mode = str(cfg.get("mode", "<"))
        v1 = _to_float(cfg, "value1", 0.0)
        v2 = _to_float(cfg, "value2", 0.0)

        if name == "rsi":
            indicator = calculate_rsi(closes, _to_int(cfg, "period", 14))
            if not _passes_condition(indicator, mode, v1, v2):
                return False, "rsi condition failed"
        elif name == "macd":
            vals = macd(closes)
            if vals is None:
                return False, "macd not ready"
            macd_line, signal_line, hist = vals
            macd_mode = str(cfg.get("macd_mode", "hist"))
            if macd_mode == "macd>signal" and not (macd_line > signal_line):
                return False, "macd>signal failed"
            elif macd_mode == "macd<signal" and not (macd_line < signal_line):
                return False, "macd<signal failed"
            elif macd_mode == "hist" and not _passes_condition(hist, mode, v1, v2):
                return False, "macd hist condition failed"
        elif name == "ma":
            period = _to_int(cfg, "period", 20)
            ma_type = str(cfg.get("ma_type", "SMA"))
            ma_value = ema(closes, period) if ma_type == "EMA" else sma(closes, period)
            if ma_value is None:
                return False, "ma not ready"
            relation = str(cfg.get("price_relation", "price>ma"))
            if relation == "price>ma" and not (price > ma_value):
                return False, "price>ma failed"
            if relation == "price<ma" and not (price < ma_value):
                return False, "price<ma failed"
        elif name == "stochastic":
            vals = stochastic_kd(highs, lows, closes, _to_int(cfg, "k_period", 14), _to_int(cfg, "d_period", 3), _to_int(cfg, "smooth", 3))
            if vals is None:
                return False, "stochastic not ready"
            k_val, d_val = vals
            if not _passes_condition(k_val, mode, v1, v2):
                return False, "stochastic K condition failed"
            if bool(cfg.get("cross_required", False)):
                cross_mode = str(cfg.get("cross_mode", "k>d"))
                if cross_mode == "k>d" and not (k_val > d_val):
                    return False, "stochastic cross k>d failed"
                if cross_mode == "k<d" and not (k_val < d_val):
                    return False, "stochastic cross k<d failed"
        elif name == "bollinger":
            pb = bollinger_percent_b(closes, _to_int(cfg, "period", 20), _to_float(cfg, "stddev", 2.0))
            if not _passes_condition(pb, mode, v1, v2):
                return False, "bollinger condition failed"
        elif name == "adx":
            value = adx(highs, lows, closes, _to_int(cfg, "period", 14))
            if not _passes_condition(value, mode, v1, v2):
                return False, "adx condition failed"
        elif name == "ultimate":
            value = ultimate_oscillator(highs, lows, closes, _to_int(cfg, "p1", 7), _to_int(cfg, "p2", 14), _to_int(cfg, "p3", 28))
            if not _passes_condition(value, mode, v1, v2):
                return False, "ultimate condition failed"
        elif name == "sar":
            value = parabolic_sar(highs, lows, _to_float(cfg, "step", 0.02), _to_float(cfg, "max_step", 0.2))
            if value is None:
                return False, "sar not ready"
            rel = str(cfg.get("price_relation", "price>sar"))
            if rel == "price>sar" and not (price > value):
                return False, "price>sar failed"
            if rel == "price<sar" and not (price < value):
                return False, "price<sar failed"
        elif name == "mfi":
            value = mfi(highs, lows, closes, volumes, _to_int(cfg, "period", 14))
            if not _passes_condition(value, mode, v1, v2):
                return False, "mfi condition failed"
        elif name == "cci":
            value = cci(highs, lows, closes, _to_int(cfg, "period", 20))
            if not _passes_condition(value, mode, v1, v2):
                return False, "cci condition failed"
    return True, "all filters passed"


def evaluate_entry_filters(exchange, symbol: str, filters: dict[str, Any]) -> bool:
    ok, _ = evaluate_entry_filters_with_reason(exchange, symbol, filters)
    return ok


def get_rsi(exchange, symbol: str, timeframe: str, period: int) -> float:
    limit = max(100, period * 3)
    closes = fetch_ohlcv_closes(exchange, symbol, timeframe, limit)
    return calculate_rsi(closes, period)


def simple_self_check() -> bool:
    closes = [float(i) for i in range(1, 260)]
    highs = [v + 1 for v in closes]
    lows = [v - 1 for v in closes]
    volumes = [1000 + i for i in range(len(closes))]
    return all(
        [
            sma(closes, 20) is not None,
            ema(closes, 20) is not None,
            macd(closes) is not None,
            stochastic_kd(highs, lows, closes) is not None,
            bollinger_percent_b(closes) is not None,
            adx(highs, lows, closes) is not None,
            ultimate_oscillator(highs, lows, closes) is not None,
            parabolic_sar(highs, lows) is not None,
            mfi(highs, lows, closes, volumes) is not None,
            cci(highs, lows, closes) is not None,
            calculate_rsi(closes, 14) > 0,
        ]
    )


if __name__ == "__main__":
    assert simple_self_check()
    assert sma([1.0, 2.0], 5) is None
    assert ema([1.0, 2.0], 5) is None
    assert macd([1.0] * 10) is None
    assert stochastic_kd([1.0] * 5, [1.0] * 5, [1.0] * 5) is None
    assert bollinger_percent_b([1.0] * 5) is None
    assert adx([1.0] * 5, [1.0] * 5, [1.0] * 5) is None
    assert ultimate_oscillator([1.0] * 5, [1.0] * 5, [1.0] * 5) is None
    assert parabolic_sar([1.0], [1.0]) is None
    assert mfi([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5) is None
    assert cci([1.0] * 5, [1.0] * 5, [1.0] * 5) is None
    print("indicators self-test: OK")
