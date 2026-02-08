from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import ccxt

from indicators import get_rsi


class PairWorker:
    def __init__(
        self,
        pair: str,
        event_queue: queue.Queue,
        logger: logging.Logger,
        settings: dict[str, Any],
        initial_stats: dict[str, float | int],
        api_key: str,
        api_secret: str,
    ) -> None:
        self.pair = pair
        self.event_queue = event_queue
        self.logger = logger
        self.settings = settings
        self.api_key = api_key
        self.api_secret = api_secret
        self.status = "STOPPED"
        self.cycle_count = int(initial_stats.get("cycle", 0))
        self.pnl_usdt = float(initial_stats.get("pnl_usdt", 0.0))
        self.pnl_pct = float(initial_stats.get("pnl_pct", 0.0))
        self.closed_trades = int(initial_stats.get("closed_trades", 0))
        self.invested_usdt = float(initial_stats.get("invested_usdt", 0.0))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exchange = ccxt.mexc(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        self._state = "IDLE"
        self._entry_price = 0.0
        self._avg_price = 0.0
        self._position_qty = 0.0
        self._total_cost = 0.0
        self._next_safety_index = 0
        self._last_rsi_check = 0.0
        self.entry_order_id: str | None = None
        self.tp_order_id: str | None = None
        self.safety_order_ids: list[str] = []
        self._accounted_safety_ids: set[str] = set()
        self._last_entry_log = 0.0
        self._last_tp_log = 0.0
        self._last_tp_attempt = 0.0
        self._last_heartbeat = 0.0
        self._last_ticker_time = 0.0
        self._order_poll_times: dict[str, float] = {}
        self._error_sleep = 1
        self.entry_active = False
        self.tp_active = False
        self._safety_placed = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.logger.info("Worker already running for %s", self.pair)
            return
        if not self.api_key or not self.api_secret:
            self.logger.error("Missing API credentials for %s", self.pair)
            return
        self._stop_event.clear()
        self.status = "RUNNING"
        self._thread = threading.Thread(target=self._run, name=f"PairWorker-{self.pair}", daemon=True)
        self._thread.start()
        self.logger.info("Worker started for %s", self.pair)

    def stop(self) -> None:
        self._stop_event.set()
        self._cancel_open_orders()
        self.status = "STOPPED"
        self.logger.info("Worker stop requested for %s", self.pair)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(1)
            self.cycle_count += 1
            now = time.time()
            if now - self._last_ticker_time < 1:
                continue
            ticker_ok, ticker = self._safe_ccxt_call(self._exchange.fetch_ticker, self.pair)
            if not ticker_ok:
                continue
            self._last_ticker_time = now
            try:
                current_price = float(ticker["last"])
            except (TypeError, ValueError, KeyError):
                self.logger.warning("Invalid ticker data for %s: %s", self.pair, ticker)
                continue

            strategy = self.settings.get("strategy", {})
            rsi_settings = self.settings.get("rsi", {})
            take_profit_pct = float(strategy.get("take_profit_pct", 1.0))
            base_order_usdt = float(strategy.get("base_order_usdt", 10.0))
            safety_orders_count = int(strategy.get("safety_orders_count", 0))
            safety_step_pct = float(strategy.get("safety_step_pct", 0.0))
            volume_multiplier = float(strategy.get("volume_multiplier", 1.0))
            fee_pct = float(strategy.get("fee_pct", 0.1))
            use_safety = bool(strategy.get("use_safety_orders", True))
            use_market_entry = bool(strategy.get("use_market_entry", False))

            if self._state == "IDLE":
                can_enter = True
                if bool(rsi_settings.get("enabled", False)):
                    now = time.time()
                    if now - self._last_rsi_check >= 10:
                        try:
                            rsi_value = get_rsi(
                                self._exchange,
                                self.pair,
                                str(rsi_settings.get("timeframe", "15m")),
                                int(rsi_settings.get("period", 14)),
                            )
                            threshold = float(rsi_settings.get("threshold", 30))
                            can_enter = rsi_value < threshold
                            self._last_rsi_check = now
                            self.logger.info(
                                "RSI check for %s: value=%.2f threshold=%.2f",
                                self.pair,
                                rsi_value,
                                threshold,
                            )
                        except Exception as exc:
                            self.logger.warning("RSI check error for %s: %s", self.pair, exc)
                            can_enter = False
                    else:
                        can_enter = False

                if can_enter:
                    entry_limit_price = current_price * 0.999
                    amount_base = base_order_usdt / entry_limit_price
                    if not self.entry_active:
                        if use_market_entry:
                            order_ok, order = self._safe_ccxt_call(
                                self._exchange.create_market_buy_order,
                                self.pair,
                                amount_base,
                            )
                        else:
                            order_ok, order = self._safe_ccxt_call(
                                self._exchange.create_limit_buy_order,
                                self.pair,
                                amount_base,
                                entry_limit_price,
                            )
                        if order_ok:
                            self.entry_order_id = str(order.get("id") or "")
                            self.entry_active = bool(self.entry_order_id)
                            self._entry_price = entry_limit_price
                            self._state = "ENTRY_PLACED"
                            self.logger.info(
                                "Entry order placed for %s (%s) (id=%s)",
                                self.pair,
                                "market" if use_market_entry else f"limit {entry_limit_price:.6f}",
                                self.entry_order_id,
                            )

            elif self._state == "ENTRY_PLACED":
                if not self.entry_order_id:
                    self._state = "IDLE"
                else:
                    order = self._safe_fetch_order(self.entry_order_id)
                    if order is None:
                        continue
                    status = order.get("status")
                    if status == "closed":
                        self.entry_active = False
                        filled = float(order.get("filled") or 0)
                        cost = float(order.get("cost") or 0)
                        if filled <= 0:
                            filled = base_order_usdt / self._entry_price
                        if cost <= 0:
                            cost = filled * self._entry_price
                        self._total_cost = cost
                        self._position_qty = filled
                        self._avg_price = self._total_cost / self._position_qty
                        self._entry_price = self._avg_price
                        self._next_safety_index = 0
                        self.safety_order_ids = []
                        self._accounted_safety_ids.clear()
                        self._safety_placed = False

                        if use_safety and safety_orders_count > 0 and not self._safety_placed:
                            for index in range(safety_orders_count):
                                safety_price = self._entry_price * (
                                    1 - (safety_step_pct / 100) * (index + 1)
                                )
                                safety_cost = base_order_usdt * (volume_multiplier ** (index + 1))
                                safety_amount = safety_cost / safety_price
                                safety_ok, safety_order = self._safe_ccxt_call(
                                    self._exchange.create_limit_buy_order,
                                    self.pair,
                                    safety_amount,
                                    safety_price,
                                )
                                if safety_ok:
                                    safety_id = safety_order.get("id")
                                    if safety_id:
                                        self.safety_order_ids.append(str(safety_id))
                            self._safety_placed = True

                        if not self.tp_active:
                            self._place_take_profit_order(take_profit_pct, reason="entry")
                        self._state = "TP_PLACED"
                        self.logger.info("Entry filled for %s; TP placed.", self.pair)
                    elif status == "partial":
                        now = time.time()
                        if now - self._last_entry_log >= 10:
                            self.logger.warning("Entry order partial for %s", self.pair)
                            self._last_entry_log = now

            elif self._state == "TP_PLACED":
                now = time.time()
                if not self.tp_active and now - self._last_tp_attempt >= 5:
                    self._last_tp_attempt = now
                    self._place_take_profit_order(take_profit_pct, reason="retry")
                    self._place_take_profit_order(take_profit_pct)
                if self.tp_order_id and self.tp_active:
                    tp_order = self._safe_fetch_order(self.tp_order_id)
                    if tp_order and tp_order.get("status") == "closed":
                        self.tp_active = False
                        self.tp_order_id = None
                        sell_return = float(tp_order.get("cost") or 0)
                        filled_qty = float(tp_order.get("filled") or 0)
                        avg_sell_price = float(tp_order.get("average") or tp_order.get("price") or 0)
                        if sell_return <= 0 and filled_qty > 0:
                            sell_return = filled_qty * avg_sell_price
                        cost_basis = filled_qty * self._avg_price
                        fee_rate = fee_pct / 100
                        fees = (cost_basis * fee_rate) + (sell_return * fee_rate)
                        profit = sell_return - cost_basis - fees
                        profit_pct = (profit / cost_basis) * 100 if cost_basis else 0.0
                        self.pnl_usdt += profit
                        self.invested_usdt += cost_basis
                        if self.invested_usdt > 0:
                            self.pnl_pct = (self.pnl_usdt / self.invested_usdt) * 100
                        else:
                            self.pnl_pct = 0.0
                        self.closed_trades += 1
                        epsilon = 1e-12
                        self.logger.info(
                            "TP filled for %s: filled_qty=%.8f sell_return=%.8f cost_basis=%.8f "
                            "profit=%.8f new_state=%s",
                            self.pair,
                            filled_qty,
                            sell_return,
                            cost_basis,
                            profit,
                            "IDLE" if filled_qty >= self._position_qty - epsilon else "IN_POSITION",
                        )
                        if filled_qty >= self._position_qty - epsilon:
                            self._cancel_open_orders(exclude_tp=True)
                            self.entry_order_id = None
                            self.tp_order_id = None
                            self.safety_order_ids = []
                            self._accounted_safety_ids.clear()
                            self._safety_placed = False
                            self._position_qty = 0.0
                            self._total_cost = 0.0
                            self._avg_price = 0.0
                            self._state = "IDLE"
                        else:
                            self._position_qty = max(0.0, self._position_qty - filled_qty)
                            self._total_cost = self._avg_price * self._position_qty
                            self.tp_active = False
                            self._place_take_profit_order(take_profit_pct, reason="partial_tp")
                        self.logger.info(
                            "TP HIT for %s: profit=%.4f (%.2f%%)",
                            self.pair,
                            profit,
                            profit_pct,
                        )
                        self.event_queue.put(
                            {
                                "type": "pair_update",
                                "pair": self.pair,
                                "cycle": self.cycle_count,
                                "pnl_usdt": self.pnl_usdt,
                                "pnl_pct": self.pnl_pct,
                                "closed_trades": self.closed_trades,
                                "invested_usdt": self.invested_usdt,
                                "status": self.status,
                                "save": True,
                            }
                        )

                for safety_id in list(self.safety_order_ids):
                    if safety_id in self._accounted_safety_ids:
                        continue
                    safety_order = self._safe_fetch_order(safety_id)
                    if safety_order is None:
                        continue
                    status = safety_order.get("status")
                    if status == "closed":
                        cost = float(safety_order.get("cost") or 0)
                        filled = float(safety_order.get("filled") or 0)
                        if filled <= 0:
                            continue
                        if cost <= 0:
                            cost = filled * float(safety_order.get("average") or self._entry_price)
                        self._total_cost += cost
                        self._position_qty += filled
                        self._avg_price = self._total_cost / self._position_qty
                        self._accounted_safety_ids.add(safety_id)
                        self.logger.info("Safety filled for %s (id=%s)", self.pair, safety_id)
                        if self.tp_order_id and self.tp_active:
                            cancel_ok, _ = self._safe_ccxt_call(
                                self._exchange.cancel_order, self.tp_order_id, self.pair
                            )
                            if cancel_ok:
                                self.tp_active = False
                                self.tp_order_id = None
                        if not self.tp_active:
                            self._place_take_profit_order(take_profit_pct, reason="safety_update")
                    elif status == "partial":
                        self.logger.warning("Safety order partial for %s (id=%s)", self.pair, safety_id)

                if now - self._last_tp_log >= 20:
                    self._last_tp_log = now
                    self.logger.info("Monitoring TP for %s", self.pair)

            if now - self._last_heartbeat >= 30:
                self._last_heartbeat = now
                self.logger.info("Worker heartbeat for %s (state=%s)", self.pair, self._state)

            self.event_queue.put(
                {
                    "type": "pair_update",
                    "pair": self.pair,
                    "cycle": self.cycle_count,
                    "pnl_usdt": self.pnl_usdt,
                    "pnl_pct": self.pnl_pct,
                    "closed_trades": self.closed_trades,
                    "invested_usdt": self.invested_usdt,
                    "status": self.status,
                }
            )
        self.logger.info("Worker stopped for %s", self.pair)

    def _safe_fetch_order(self, order_id: str) -> dict[str, Any] | None:
        now = time.time()
        last_poll = self._order_poll_times.get(order_id, 0.0)
        if now - last_poll < 2:
            return None
        self._order_poll_times[order_id] = now
        ok, result = self._safe_ccxt_call(self._exchange.fetch_order, order_id, self.pair)
        if not ok:
            return None
        return result

    def _place_take_profit_order(self, take_profit_pct: float, reason: str) -> bool:
        if self.tp_active and self.tp_order_id:
            return True

        market_ok, _ = self._safe_ccxt_call(self._exchange.load_markets)
        if not market_ok:
            self.logger.warning("TP placement skipped for %s: failed to load markets", self.pair)
            return False

        market = self._exchange.market(self.pair) or {}
        base_asset = market.get("base")
        if not base_asset and "/" in self.pair:
            base_asset = self.pair.split("/")[0]

        if not base_asset:
            self.logger.warning("TP placement skipped for %s: cannot определить base-актив", self.pair)
            return False

        min_qty = float((market.get("limits", {}) or {}).get("amount", {}).get("min") or 0.0)
        min_cost = float((market.get("limits", {}) or {}).get("cost", {}).get("min") or 0.0)
        amount_precision = (market.get("precision", {}) or {}).get("amount")
        step = 0.0
        if isinstance(amount_precision, int):
            step = 10 ** (-amount_precision)
        safety_margin = max(step * 2, 0.0)

        tp_price_raw = self._avg_price * (1 + take_profit_pct / 100)
        price_precise = float(self._exchange.price_to_precision(self.pair, tp_price_raw))

        for attempt in range(1, 4):
            bal_ok, balance = self._safe_ccxt_call(self._exchange.fetch_balance)
            if not bal_ok:
                self.logger.warning("TP placement attempt %s failed: balance not доступен", attempt)
                time.sleep(1.5)
                continue

            free_qty = float((balance.get("free", {}) or {}).get(base_asset, 0.0) or 0.0)
            sell_qty_raw = min(self._position_qty, free_qty)
            sell_qty_raw = max(0.0, sell_qty_raw - safety_margin)
            sell_qty_precise = float(self._exchange.amount_to_precision(self.pair, sell_qty_raw))

            self.logger.info(
                "TP calc for %s: reason=%s avg_price=%.6f tp_price=%.6f "
                "position_qty=%.8f free_qty=%.8f sell_qty_raw=%.8f sell_qty_precise=%.8f",
                self.pair,
                reason,
                self._avg_price,
                tp_price_raw,
                self._position_qty,
                free_qty,
                sell_qty_raw,
                sell_qty_precise,
            )

            if sell_qty_precise <= 0 or sell_qty_precise < min_qty:
                self.logger.warning(
                    "TP placement skipped for %s: insufficient free qty (%.8f)",
                    self.pair,
                    free_qty,
                )
                time.sleep(1.5)
                continue

            if min_cost and sell_qty_precise * price_precise < min_cost:
                self.logger.warning(
                    "TP placement skipped for %s: min cost not met (qty=%.8f price=%.8f min_cost=%.8f)",
                    self.pair,
                    sell_qty_precise,
                    price_precise,
                    min_cost,
                )
                time.sleep(1.5)
                continue

            order_ok, order = self._safe_ccxt_call(
                self._exchange.create_limit_sell_order,
                self.pair,
                sell_qty_precise,
                price_precise,
            )
            if order_ok:
                self.tp_order_id = str(order.get("id") or "")
                self.tp_active = bool(self.tp_order_id)
                self.logger.info("TP order placed for %s (id=%s)", self.pair, self.tp_order_id)
                return True

            error_text = str(order).lower()
            if "insufficient position" in error_text:
                self.logger.warning(
                    "TP insufficient position for %s: free_qty=%.8f position_qty=%.8f tp_qty=%.8f "
                    "min_qty=%.8f step=%.8f",
                    self.pair,
                    free_qty,
                    self._position_qty,
                    sell_qty_precise,
                    min_qty,
                    step,
                )
            self.logger.warning("TP placement attempt %s failed for %s", attempt, self.pair)
            time.sleep(1.5)

        return False

    def _cancel_open_orders(self, exclude_tp: bool = False) -> None:
        order_ids = []
        if self.entry_order_id:
            order_ids.append(self.entry_order_id)
        if self.tp_order_id and not exclude_tp:
            order_ids.append(self.tp_order_id)
        order_ids.extend(self.safety_order_ids)
        for order_id in order_ids:
            ok, order = self._safe_ccxt_call(self._exchange.fetch_order, order_id, self.pair)
            if not ok:
                continue
            try:
                status = order.get("status")
            except AttributeError:
                continue
            if status == "open":
                cancel_ok, cancel_result = self._safe_ccxt_call(
                    self._exchange.cancel_order, order_id, self.pair
                )
                if cancel_ok:
                    if order_id == self.tp_order_id:
                        self.tp_active = False
                    if order_id == self.entry_order_id:
                        self.entry_active = False
                else:
                    error_text = str(cancel_result).lower()
                    if "order not found" in error_text or "not found" in error_text:
                        self.logger.info("Order already closed for %s (id=%s)", self.pair, order_id)

    def _safe_ccxt_call(self, func, *args):
        try:
            result = func(*args)
        except Exception as exc:
            self.logger.warning("CCXT call failed for %s: %s", self.pair, exc)
            self._error_sleep = min(30, max(2, self._error_sleep * 2))
            time.sleep(self._error_sleep)
            return False, exc
        self._error_sleep = 1
        return True, result
