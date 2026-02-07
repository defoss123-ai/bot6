from __future__ import annotations


def calculate_rsi(closes: list[float], period: int) -> float:
    if period <= 0:
        raise ValueError("Period must be positive.")
    if len(closes) < period + 1:
        raise ValueError("Not enough data to calculate RSI.")

    gains = []
    losses = []
    for index in range(1, period + 1):
        delta = closes[index] - closes[index - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))

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


def fetch_ohlcv_closes(exchange, symbol: str, timeframe: str, limit: int) -> list[float]:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return [float(candle[4]) for candle in ohlcv]


def get_rsi(exchange, symbol: str, timeframe: str, period: int) -> float:
    limit = max(100, period * 3)
    closes = fetch_ohlcv_closes(exchange, symbol, timeframe, limit)
    return calculate_rsi(closes, period)
