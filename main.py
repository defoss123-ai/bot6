"""Quick start:
- pip install ccxt
- python main.py
- steps: enter API, check, add pair, configure strategy/RSI, start
"""

import logging
import os
import queue
import threading
import tkinter as tk
from collections import deque
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import ttk

import ccxt

from bot_worker import PairWorker
from indicators import get_rsi
from storage import load_json
from storage import save_json


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.log_queue.put(message)


class App:
    def __init__(
        self,
        root: tk.Tk,
        logger: logging.Logger,
        log_queue: queue.Queue,
        worker_event_queue: queue.Queue,
    ) -> None:
        self.root = root
        self.logger = logger
        self.log_queue = log_queue
        self.worker_event_queue = worker_event_queue
        self.settings_data: dict[str, str] = {}
        self.rsi_data: dict[str, str] = {}
        self.state_path = "data/state.json"
        self.state = self._default_state()
        self.workers: dict[str, PairWorker] = {}
        self._pair_stats: dict[str, dict[str, float]] = {}
        self._rsi_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

        self.root.title("DCA Bot")
        self._build_ui()
        self._load_state()
        self._schedule_log_updates()
        self._schedule_worker_event_processing()
        self._schedule_stats_update()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.api_frame = ttk.Frame(notebook)
        self.pairs_frame = ttk.Frame(notebook)
        self.settings_frame = ttk.Frame(notebook)
        self.rsi_frame = ttk.Frame(notebook)
        self.stats_frame = ttk.Frame(notebook)
        self.logs_frame = ttk.Frame(notebook)

        notebook.add(self.api_frame, text="API")
        notebook.add(self.pairs_frame, text="Пары")
        notebook.add(self.settings_frame, text="Настройки стратегии")
        notebook.add(self.rsi_frame, text="RSI-фильтр")
        notebook.add(self.stats_frame, text="Статистика")
        notebook.add(self.logs_frame, text="Логи")

        self._build_api_tab()
        self._build_pairs_tab()
        self._build_settings_tab()
        self._build_rsi_tab()
        self._build_stats_tab()
        self._build_logs_tab()

    def _build_api_tab(self) -> None:
        ttk.Label(self.api_frame, text="API Key").grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        self.api_key_entry = ttk.Entry(self.api_frame, width=40)
        self.api_key_entry.grid(row=0, column=1, padx=8, pady=6, sticky=tk.W)

        ttk.Label(self.api_frame, text="API Secret").grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        self.api_secret_entry = ttk.Entry(self.api_frame, width=40, show="*")
        self.api_secret_entry.grid(row=1, column=1, padx=8, pady=6, sticky=tk.W)

        ttk.Label(self.api_frame, text="Биржа").grid(row=2, column=0, sticky=tk.W, padx=8, pady=6)
        self.exchange_var = tk.StringVar(value="MEXC")
        self.exchange_combo = ttk.Combobox(
            self.api_frame,
            textvariable=self.exchange_var,
            values=list(self._exchange_labels.keys()),
            width=15,
            state="readonly",
        )
        self.exchange_combo.grid(row=2, column=1, sticky=tk.W, padx=8, pady=6)

        self.api_status_label = ttk.Label(self.api_frame, text="—")
        self.api_status_label.grid(row=3, column=1, sticky=tk.W, padx=8, pady=6)

        self.api_details_label = ttk.Label(self.api_frame, text="Детали: —")
        self.api_details_label.grid(row=4, column=1, sticky=tk.W, padx=8, pady=4)

        self.save_credentials_var = tk.BooleanVar(value=False)
        save_credentials_check = ttk.Checkbutton(
            self.api_frame,
            text="Сохранять API данные",
            variable=self.save_credentials_var,
            command=self._save_state,
        )
        save_credentials_check.grid(row=5, column=0, columnspan=2, padx=8, pady=4, sticky=tk.W)

        check_button = ttk.Button(
            self.api_frame,
            text="Проверить подключение",
            command=self._handle_api_check,
        )
        check_button.grid(row=3, column=0, padx=8, pady=6, sticky=tk.W)

    def _handle_api_check(self) -> None:
        api_key = self.api_key_entry.get().strip()
        api_secret = self.api_secret_entry.get().strip()
        exchange_label = self.exchange_var.get() or "MEXC"
        exchange_id = self._exchange_labels.get(exchange_label, "mexc")
        if not api_key or not api_secret:
            self.api_status_label.config(text="ERROR")
            self.api_details_label.config(text="Детали: Пустые ключи")
            messagebox.showwarning("API", "Введите API Key и API Secret.")
            self.logger.info("API check ERROR: empty credentials")
            self._save_state()
            return

        self.api_status_label.config(text="—")
        self.api_details_label.config(text="Детали: Проверка...")
        self.logger.info("API check started (%s spot)", exchange_id)
        thread = threading.Thread(
            target=self._run_api_check,
            args=(exchange_id, api_key, api_secret),
            daemon=True,
        )
        thread.start()
        self._save_state()

    def _build_pairs_tab(self) -> None:
        columns = ("pair", "status", "cycle", "pnl_usdt", "pnl_pct", "closed_trades")
        self.pairs_tree = ttk.Treeview(self.pairs_frame, columns=columns, show="headings", height=10)
        self.pairs_tree.heading("pair", text="Pair")
        self.pairs_tree.heading("status", text="Status")
        self.pairs_tree.heading("cycle", text="Cycle")
        self.pairs_tree.heading("pnl_usdt", text="PnL_USDT")
        self.pairs_tree.heading("pnl_pct", text="PnL_%")
        self.pairs_tree.heading("closed_trades", text="ClosedTrades")

        self.pairs_tree.column("pair", width=120)
        self.pairs_tree.column("status", width=100)
        self.pairs_tree.column("cycle", width=80, anchor=tk.CENTER)
        self.pairs_tree.column("pnl_usdt", width=100, anchor=tk.CENTER)
        self.pairs_tree.column("pnl_pct", width=80, anchor=tk.CENTER)
        self.pairs_tree.column("closed_trades", width=110, anchor=tk.CENTER)

        self.pairs_tree.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=8, pady=8)

        self.pairs_frame.grid_rowconfigure(0, weight=1)
        self.pairs_frame.grid_columnconfigure(0, weight=1)

        add_button = ttk.Button(self.pairs_frame, text="Добавить пару", command=self._open_add_pair_dialog)
        delete_button = ttk.Button(self.pairs_frame, text="Удалить", command=self._delete_selected_pair)
        start_button = ttk.Button(self.pairs_frame, text="Запустить", command=self._start_selected_pair)
        stop_button = ttk.Button(self.pairs_frame, text="Остановить", command=self._stop_selected_pair)
        stop_all_button = ttk.Button(self.pairs_frame, text="Остановить все", command=self._stop_all_pairs)

        add_button.grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        delete_button.grid(row=1, column=1, sticky=tk.W, padx=8, pady=6)
        start_button.grid(row=1, column=2, sticky=tk.W, padx=8, pady=6)
        stop_button.grid(row=1, column=3, sticky=tk.W, padx=8, pady=6)
        stop_all_button.grid(row=1, column=4, sticky=tk.W, padx=8, pady=6)

    def _open_add_pair_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавить пару")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Пара").grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)
        pair_entry = ttk.Entry(dialog, width=30)
        pair_entry.grid(row=0, column=1, padx=8, pady=8)

        def confirm() -> None:
            pair_value = normalize_symbol(pair_entry.get().strip())
            if pair_value:
                self.pairs_tree.insert(
                    "",
                    tk.END,
                    values=(pair_value, "STOPPED", 0, 0, 0, 0),
                )
                self.logger.info("Pair added: %s", pair_value)
                self._save_state()
            dialog.destroy()

        add_button = ttk.Button(dialog, text="Добавить", command=confirm)
        add_button.grid(row=1, column=0, columnspan=2, pady=8)

    def _get_selected_pair(self) -> str | None:
        selected = self.pairs_tree.selection()
        if not selected:
            return None
        return selected[0]

    def _delete_selected_pair(self) -> None:
        selected = self._get_selected_pair()
        if selected:
            values = self.pairs_tree.item(selected, "values")
            pair = str(values[0])
            worker = self.workers.pop(pair, None)
            if worker:
                worker.stop()
            self.pairs_tree.delete(selected)
            self.logger.info("Pair removed: %s", pair)
            self._save_state()

    def _start_selected_pair(self) -> None:
        selected = self._get_selected_pair()
        if selected:
            values = list(self.pairs_tree.item(selected, "values"))
            pair = normalize_symbol(str(values[0]))
            api_key = self.api_key_entry.get().strip()
            api_secret = self.api_secret_entry.get().strip()
            if not api_key or not api_secret:
                messagebox.showwarning("API", "Сначала настройте API.")
                self.logger.warning("Cannot start worker for %s: missing API credentials.", pair)
                return
            if not self._validate_strategy_inputs(for_save=False):
                messagebox.showwarning("Настройки", "Сначала сохраните корректные настройки стратегии.")
                self.logger.warning("Cannot start worker for %s: invalid strategy settings.", pair)
                return
            exchange_label = self.exchange_var.get() or "MEXC"
            exchange_id = self._exchange_labels.get(exchange_label, "mexc")
            try:
                exchange = build_exchange(exchange_id, api_key, api_secret)
            except Exception as exc:
                messagebox.showerror("Биржа", f"Не удалось загрузить рынки: {exc}")
                self.logger.error("Failed to load markets for %s: %s", exchange_id, exc)
                return
            if pair not in exchange.markets:
                messagebox.showwarning("Пара", f"Пара {pair} не найдена на бирже {exchange_label}.")
                self.logger.warning("Symbol %s not found on %s", pair, exchange_id)
                return
            worker = self.workers.get(pair)
            if worker is None:
                state = self._collect_state()
                initial_stats = {
                    "cycle": int(values[2]),
                    "pnl_usdt": float(values[3]),
                    "pnl_pct": float(values[4]),
                    "closed_trades": int(values[5]),
                    "invested_usdt": float(self._pair_stats.get(pair, {}).get("invested_usdt", 0.0)),
                }
                settings = {
                    "strategy": state.get("strategy", {}),
                    "rsi": state.get("rsi", {}),
                }
                worker = PairWorker(
                    pair,
                    self.worker_event_queue,
                    self.logger,
                    settings,
                    initial_stats,
                    api_key,
                    api_secret,
                )
                self.workers[pair] = worker
            worker.start()
            values[1] = worker.status
            self.pairs_tree.item(selected, values=values)
            self.logger.info("Pair started: %s", pair)
            self._save_state()

    def _stop_selected_pair(self) -> None:
        selected = self._get_selected_pair()
        if selected:
            values = list(self.pairs_tree.item(selected, "values"))
            pair = str(values[0])
            worker = self.workers.get(pair)
            if worker:
                worker.stop()
            values[1] = "STOPPED"
            self.pairs_tree.item(selected, values=values)
            self.logger.info("Pair stopped: %s", pair)
            self._save_state()

    def _stop_all_pairs(self, save_state: bool = True) -> None:
        for worker in self.workers.values():
            worker.stop()
        for item in self.pairs_tree.get_children():
            values = list(self.pairs_tree.item(item, "values"))
            values[1] = "STOPPED"
            self.pairs_tree.item(item, values=values)
        self.logger.info("All pairs stopped")
        if save_state:
            self._save_state()

    def _build_settings_tab(self) -> None:
        fields = [
            ("Take Profit %", "take_profit"),
            ("Базовый ордер (USDT)", "base_order"),
            ("Кол-во страховочных ордеров", "safety_orders"),
            ("Шаг страховочных ордеров %", "safety_step"),
            ("Мультипликатор объёма", "volume_multiplier"),
            ("Диапазон цен % (опционально)", "price_range"),
            ("Комиссия %", "fee"),
        ]

        self.settings_entries: dict[str, ttk.Entry] = {}
        for row, (label, key) in enumerate(fields):
            ttk.Label(self.settings_frame, text=label).grid(row=row, column=0, padx=8, pady=4, sticky=tk.W)
            entry = ttk.Entry(self.settings_frame, width=30)
            entry.grid(row=row, column=1, padx=8, pady=4, sticky=tk.W)
            self.settings_entries[key] = entry

        self.settings_entries["fee"].insert(0, "0.1")

        self.use_safety_var = tk.BooleanVar(value=True)
        safety_check = ttk.Checkbutton(
            self.settings_frame,
            text="Использовать страховочные ордера",
            variable=self.use_safety_var,
        )
        safety_check.grid(row=len(fields), column=0, columnspan=2, padx=8, pady=6, sticky=tk.W)

        self.use_market_entry_var = tk.BooleanVar(value=False)
        market_entry_check = ttk.Checkbutton(
            self.settings_frame,
            text="Первый ордер — Market",
            variable=self.use_market_entry_var,
        )
        market_entry_check.grid(row=len(fields) + 1, column=0, columnspan=2, padx=8, pady=4, sticky=tk.W)

        save_button = ttk.Button(
            self.settings_frame,
            text="Сохранить настройки",
            command=self._save_settings,
        )
        save_button.grid(row=len(fields) + 2, column=0, columnspan=2, pady=8)

    def _save_settings(self) -> None:
        if not self._validate_strategy_inputs(for_save=True):
            return
        self.settings_data = {key: entry.get().strip() for key, entry in self.settings_entries.items()}
        self.settings_data["use_safety_orders"] = str(self.use_safety_var.get())
        self.settings_data["use_market_entry"] = str(self.use_market_entry_var.get())
        self.logger.info("Strategy settings saved: %s", self.settings_data)
        self._save_state()

    def _build_rsi_tab(self) -> None:
        self.use_rsi_var = tk.BooleanVar(value=False)
        rsi_check = ttk.Checkbutton(
            self.rsi_frame,
            text="Включить RSI-фильтр",
            variable=self.use_rsi_var,
        )
        rsi_check.grid(row=0, column=0, columnspan=2, padx=8, pady=6, sticky=tk.W)

        ttk.Label(self.rsi_frame, text="RSI < значение").grid(row=1, column=0, padx=8, pady=4, sticky=tk.W)
        self.rsi_value_entry = ttk.Entry(self.rsi_frame, width=20)
        self.rsi_value_entry.insert(0, "30")
        self.rsi_value_entry.grid(row=1, column=1, padx=8, pady=4, sticky=tk.W)

        ttk.Label(self.rsi_frame, text="Таймфрейм").grid(row=2, column=0, padx=8, pady=4, sticky=tk.W)
        self.rsi_timeframe = ttk.Combobox(
            self.rsi_frame,
            values=self._rsi_timeframes,
            width=18,
        )
        self.rsi_timeframe.set("15m")
        self.rsi_timeframe.grid(row=2, column=1, padx=8, pady=4, sticky=tk.W)

        ttk.Label(self.rsi_frame, text="Период RSI").grid(row=3, column=0, padx=8, pady=4, sticky=tk.W)
        self.rsi_period_entry = ttk.Entry(self.rsi_frame, width=20)
        self.rsi_period_entry.insert(0, "14")
        self.rsi_period_entry.grid(row=3, column=1, padx=8, pady=4, sticky=tk.W)

        save_button = ttk.Button(self.rsi_frame, text="Сохранить RSI", command=self._save_rsi)
        save_button.grid(row=4, column=0, columnspan=2, pady=8)

        self.rsi_current_label = ttk.Label(self.rsi_frame, text="Текущий RSI: —")
        self.rsi_current_label.grid(row=5, column=0, columnspan=2, padx=8, pady=4, sticky=tk.W)

        self.rsi_condition_label = ttk.Label(self.rsi_frame, text="Условие входа (RSI<threshold): —")
        self.rsi_condition_label.grid(row=6, column=0, columnspan=2, padx=8, pady=4, sticky=tk.W)

        check_button = ttk.Button(
            self.rsi_frame,
            text="Проверить RSI по выбранной паре",
            command=self._check_rsi_for_selected_pair,
        )
        check_button.grid(row=7, column=0, columnspan=2, pady=8)

    def _save_rsi(self) -> None:
        if not self._validate_rsi_inputs():
            return
        self.rsi_data = {
            "enabled": str(self.use_rsi_var.get()),
            "rsi_value": self.rsi_value_entry.get().strip(),
            "timeframe": self.rsi_timeframe.get().strip(),
            "period": self.rsi_period_entry.get().strip(),
        }
        self.logger.info("RSI settings saved: %s", self.rsi_data)
        self._save_state()

    def _check_rsi_for_selected_pair(self) -> None:
        selected = self._get_selected_pair()
        if not selected:
            messagebox.showwarning("RSI", "Выберите пару в таблице.")
            self.logger.warning("RSI check requested without selected pair.")
            return

        values = self.pairs_tree.item(selected, "values")
        symbol = normalize_symbol(str(values[0]))
        if symbol != values[0]:
            values = list(values)
            values[0] = symbol
            self.pairs_tree.item(selected, values=values)
            self._save_state()

        try:
            threshold = float(self.rsi_value_entry.get().strip())
            period = int(self.rsi_period_entry.get().strip())
            timeframe = self.rsi_timeframe.get().strip() or "15m"
            exchange_label = self.exchange_var.get() or "MEXC"
            exchange_id = self._exchange_labels.get(exchange_label, "mexc")
            exchange = build_exchange(exchange_id, "", "")
            rsi_value = get_rsi(exchange, symbol, timeframe, period)
            condition_met = rsi_value < threshold
            self.rsi_current_label.config(text=f"Текущий RSI: {rsi_value:.2f}")
            self.rsi_condition_label.config(
                text=f"Условие входа (RSI<threshold): {'ДА' if condition_met else 'НЕТ'}",
            )
            self.logger.info(
                "RSI check for %s: value=%.2f threshold=%.2f condition=%s",
                symbol,
                rsi_value,
                threshold,
                "YES" if condition_met else "NO",
            )
        except Exception as exc:
            messagebox.showerror("RSI", "Не удалось проверить RSI. Проверьте настройки и пару.")
            self.logger.exception("RSI check failed for %s: %s", symbol, exc)

    def _build_stats_tab(self) -> None:
        labels = [
            ("Общий PnL (USDT)", "total_pnl_usdt"),
            ("Общий PnL (%)", "total_pnl_pct"),
            ("Всего закрытых сделок", "total_trades"),
            ("Лучшая пара", "best_pair"),
            ("Худшая пара", "worst_pair"),
        ]
        self.stats_labels: dict[str, ttk.Label] = {}
        for row, (label, key) in enumerate(labels):
            ttk.Label(self.stats_frame, text=label).grid(row=row, column=0, padx=8, pady=4, sticky=tk.W)
            value_label = ttk.Label(self.stats_frame, text="—")
            value_label.grid(row=row, column=1, padx=8, pady=4, sticky=tk.W)
            self.stats_labels[key] = value_label

        refresh_button = ttk.Button(self.stats_frame, text="Обновить", command=self._update_stats)
        refresh_button.grid(row=len(labels), column=0, columnspan=2, pady=8)

    def _calculate_stats(self) -> dict[str, str]:
        total_pnl_usdt = 0.0
        total_invested_usdt = 0.0
        total_trades = 0
        best_pair = "—"
        worst_pair = "—"
        best_pnl = None
        worst_pnl = None

        for item in self.pairs_tree.get_children():
            values = self.pairs_tree.item(item, "values")
            pnl_usdt = float(values[3])
            closed_trades = int(values[5])
            total_pnl_usdt += pnl_usdt
            total_trades += closed_trades
            pair_key = str(values[0])
            invested_usdt = float(self._pair_stats.get(pair_key, {}).get("invested_usdt", 0.0))
            total_invested_usdt += invested_usdt

            if best_pnl is None or pnl_usdt > best_pnl:
                best_pnl = pnl_usdt
                best_pair = values[0]
            if worst_pnl is None or pnl_usdt < worst_pnl:
                worst_pnl = pnl_usdt
                worst_pair = values[0]

        if total_invested_usdt > 0:
            total_pnl_pct = f"{(total_pnl_usdt / total_invested_usdt) * 100:.2f}"
        else:
            total_pnl_pct = "0"
        stats = {
            "total_pnl_usdt": f"{total_pnl_usdt:.2f}",
            "total_pnl_pct": total_pnl_pct,
            "total_trades": str(total_trades),
            "best_pair": best_pair,
            "worst_pair": worst_pair,
        }
        return stats

    def _update_stats(self) -> None:
        stats = self._calculate_stats()
        for key, value in stats.items():
            self.stats_labels[key].config(text=value)

    def _schedule_stats_update(self) -> None:
        self._update_stats()
        self.root.after(1000, self._schedule_stats_update)

    def _build_logs_tab(self) -> None:
        self.log_buffer = deque(maxlen=3000)
        self.log_pause_var = tk.BooleanVar(value=False)
        self.log_filter_var = tk.StringVar(value="ALL")

        controls_frame = ttk.Frame(self.logs_frame)
        controls_frame.pack(fill=tk.X, padx=8, pady=6)

        pause_check = ttk.Checkbutton(controls_frame, text="Пауза", variable=self.log_pause_var)
        pause_check.pack(side=tk.LEFT)

        ttk.Label(controls_frame, text="Фильтр:").pack(side=tk.LEFT, padx=(12, 4))
        filter_box = ttk.Combobox(
            controls_frame,
            textvariable=self.log_filter_var,
            values=["ALL", "INFO", "WARNING", "ERROR"],
            width=12,
            state="readonly",
        )
        filter_box.pack(side=tk.LEFT)
        filter_box.bind("<<ComboboxSelected>>", lambda _: self._render_log_buffer())

        clear_button = ttk.Button(controls_frame, text="Очистить", command=self._clear_logs)
        clear_button.pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(self.logs_frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _clear_logs(self) -> None:
        self.log_buffer.clear()
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log_line_matches_filter(self, line: str) -> bool:
        level_filter = self.log_filter_var.get()
        if level_filter == "ALL":
            return True
        upper_line = line.upper()
        if level_filter == "ERROR":
            return "ERROR" in upper_line
        if level_filter == "WARNING":
            return "WARNING" in upper_line or "ERROR" in upper_line
        if "DEBUG" in upper_line:
            return False
        return True

    def _render_log_buffer(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for line in self.log_buffer:
            if self._log_line_matches_filter(line):
                self.log_text.insert(tk.END, line + "\n")
        if not self.log_pause_var.get():
            self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _schedule_log_updates(self) -> None:
        self._drain_log_queue()
        self.root.after(200, self._schedule_log_updates)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if self.log_pause_var.get():
                continue
            was_full = len(self.log_buffer) == self.log_buffer.maxlen
            self.log_buffer.append(message)
            if was_full:
                self._render_log_buffer()
                continue
            if self._log_line_matches_filter(message):
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._stop_all_pairs(save_state=False)
        for worker in self.workers.values():
            worker.join(timeout=2)
        self._save_state()
        self.root.destroy()

    def _validate_strategy_inputs(self, for_save: bool) -> bool:
        take_profit_text = self.settings_entries["take_profit"].get().strip()
        base_order_text = self.settings_entries["base_order"].get().strip()
        safety_orders_text = self.settings_entries["safety_orders"].get().strip()
        safety_step_text = self.settings_entries["safety_step"].get().strip()
        volume_text = self.settings_entries["volume_multiplier"].get().strip()
        price_range_text = self.settings_entries["price_range"].get().strip()
        fee_text = self.settings_entries["fee"].get().strip()

        try:
            take_profit = float(take_profit_text)
        except ValueError:
            messagebox.showwarning("Настройки", "Take Profit % должен быть числом.")
            self.logger.warning("Invalid take_profit_pct: %s", take_profit_text)
            return False
        if take_profit <= 0 or take_profit >= 100:
            messagebox.showwarning("Настройки", "Take Profit % должен быть между 0 и 100.")
            self.logger.warning("Invalid take_profit_pct range: %s", take_profit_text)
            return False

        try:
            base_order = float(base_order_text)
        except ValueError:
            messagebox.showwarning("Настройки", "Базовый ордер (USDT) должен быть числом.")
            self.logger.warning("Invalid base_order_usdt: %s", base_order_text)
            return False
        if base_order <= 0:
            messagebox.showwarning("Настройки", "Базовый ордер (USDT) должен быть больше 0.")
            self.logger.warning("Invalid base_order_usdt range: %s", base_order_text)
            return False

        try:
            safety_orders = int(safety_orders_text)
        except ValueError:
            messagebox.showwarning("Настройки", "Кол-во страховочных ордеров должно быть целым числом.")
            self.logger.warning("Invalid safety_orders_count: %s", safety_orders_text)
            return False
        if safety_orders < 0:
            messagebox.showwarning("Настройки", "Кол-во страховочных ордеров не может быть отрицательным.")
            self.logger.warning("Invalid safety_orders_count range: %s", safety_orders_text)
            return False

        try:
            safety_step = float(safety_step_text)
        except ValueError:
            messagebox.showwarning("Настройки", "Шаг страховочных ордеров % должен быть числом.")
            self.logger.warning("Invalid safety_step_pct: %s", safety_step_text)
            return False
        if safety_orders > 0 and safety_step <= 0:
            messagebox.showwarning("Настройки", "Шаг страховочных ордеров % должен быть больше 0.")
            self.logger.warning("Invalid safety_step_pct range: %s", safety_step_text)
            return False

        try:
            volume_multiplier = float(volume_text)
        except ValueError:
            messagebox.showwarning("Настройки", "Мультипликатор объёма должен быть числом.")
            self.logger.warning("Invalid volume_multiplier: %s", volume_text)
            return False
        if volume_multiplier < 1.0:
            if for_save:
                messagebox.showwarning(
                    "Настройки",
                    "Мультипликатор объёма не может быть меньше 1.0. Установлено 1.0.",
                )
                self.logger.warning("volume_multiplier < 1.0 corrected to 1.0")
                self.settings_entries["volume_multiplier"].delete(0, tk.END)
                self.settings_entries["volume_multiplier"].insert(0, "1.0")
            else:
                self.logger.warning("Invalid volume_multiplier range: %s", volume_text)
                return False

        if fee_text == "":
            if for_save:
                self.settings_entries["fee"].delete(0, tk.END)
                self.settings_entries["fee"].insert(0, "0.1")
                self.logger.warning("Empty fee_pct corrected to default 0.1")
            else:
                messagebox.showwarning("Настройки", "Комиссия % обязательна.")
                self.logger.warning("Missing fee_pct")
                return False
            fee_text = "0.1"
        try:
            fee_pct = float(fee_text)
        except ValueError:
            messagebox.showwarning("Настройки", "Комиссия % должна быть числом.")
            self.logger.warning("Invalid fee_pct: %s", fee_text)
            return False
        if fee_pct < 0 or fee_pct >= 5:
            messagebox.showwarning("Настройки", "Комиссия % должна быть между 0 и 5.")
            self.logger.warning("Invalid fee_pct range: %s", fee_text)
            return False

        if price_range_text:
            try:
                price_range = float(price_range_text)
            except ValueError:
                messagebox.showwarning("Настройки", "Диапазон цен % должен быть числом.")
                self.logger.warning("Invalid price_range_pct: %s", price_range_text)
                return False
            if price_range <= 0 or price_range >= 100:
                messagebox.showwarning("Настройки", "Диапазон цен % должен быть между 0 и 100.")
                self.logger.warning("Invalid price_range_pct range: %s", price_range_text)
                return False

        return True

    def _validate_rsi_inputs(self) -> bool:
        threshold_text = self.rsi_value_entry.get().strip()
        period_text = self.rsi_period_entry.get().strip()
        timeframe = self.rsi_timeframe.get().strip()

        try:
            threshold = float(threshold_text)
        except ValueError:
            messagebox.showwarning("RSI", "RSI порог должен быть числом.")
            self.logger.warning("Invalid RSI threshold: %s", threshold_text)
            return False
        if threshold <= 0 or threshold >= 100:
            messagebox.showwarning("RSI", "RSI порог должен быть между 1 и 99.")
            self.logger.warning("Invalid RSI threshold range: %s", threshold_text)
            return False

        try:
            period = int(period_text)
        except ValueError:
            messagebox.showwarning("RSI", "Период RSI должен быть целым числом.")
            self.logger.warning("Invalid RSI period: %s", period_text)
            return False
        if period < 2 or period > 100:
            messagebox.showwarning("RSI", "Период RSI должен быть между 2 и 100.")
            self.logger.warning("Invalid RSI period range: %s", period_text)
            return False

        if timeframe not in self._rsi_timeframes:
            messagebox.showwarning("RSI", "Неверный таймфрейм RSI.")
            self.logger.warning("Invalid RSI timeframe: %s", timeframe)
            return False

        return True

    def _schedule_worker_event_processing(self) -> None:
        self._process_worker_events()
        self.root.after(200, self._schedule_worker_event_processing)

    def _process_worker_events(self) -> None:
        while True:
            try:
                event = self.worker_event_queue.get_nowait()
            except queue.Empty:
                break
            if event.get("type") == "pair_update":
                self._apply_pair_update(event)
            elif event.get("type") == "api_check_result":
                self._handle_api_check_result(event)

    def _run_api_check(self, exchange_id: str, api_key: str, api_secret: str) -> None:
        try:
            exchange = build_exchange(exchange_id, api_key, api_secret)
            exchange.fetch_balance()
            self.worker_event_queue.put(
                {"type": "api_check_result", "ok": True, "message": "OK"}
            )
        except Exception as exc:
            self.worker_event_queue.put(
                {"type": "api_check_result", "ok": False, "message": "ERROR", "error": repr(exc)}
            )

    def _handle_api_check_result(self, event: dict) -> None:
        ok = bool(event.get("ok"))
        if ok:
            self.api_status_label.config(text="OK")
            self.api_details_label.config(text="Детали: OK")
            self.logger.info("API check OK")
            return
        error_detail = event.get("error", "Unknown error")
        self.api_status_label.config(text="ERROR")
        self.api_details_label.config(text=f"Детали: {error_detail}")
        messagebox.showerror("API", "Не удалось проверить ключи API.")
        self.logger.error("API check ERROR: %s", error_detail)

    def _apply_pair_update(self, event: dict) -> None:
        pair = event.get("pair")
        if not pair:
            return
        for item in self.pairs_tree.get_children():
            values = list(self.pairs_tree.item(item, "values"))
            if str(values[0]) == str(pair):
                if event.get("cycle") is not None:
                    values[2] = event["cycle"]
                if event.get("pnl_usdt") is not None:
                    values[3] = round(float(event["pnl_usdt"]), 4)
                if event.get("pnl_pct") is not None:
                    values[4] = round(float(event["pnl_pct"]), 4)
                if event.get("closed_trades") is not None:
                    values[5] = int(event["closed_trades"])
                if event.get("status"):
                    values[1] = event["status"]
                self.pairs_tree.item(item, values=values)
                break
        if event.get("invested_usdt") is not None:
            self._pair_stats.setdefault(str(pair), {})["invested_usdt"] = float(event["invested_usdt"])
        if event.get("save"):
            self._save_state()

    def _default_state(self) -> dict:
        return {
            "api": {"exchange_id": "mexc", "key": "", "secret": "", "save_credentials": False},
            "strategy": {
                "take_profit_pct": 1.0,
                "base_order_usdt": 10.0,
                "safety_orders_count": 5,
                "safety_step_pct": 2.0,
                "volume_multiplier": 1.2,
                "price_range_pct": "",
                "fee_pct": 0.1,
                "use_safety_orders": True,
                "use_market_entry": False,
            },
            "rsi": {
                "enabled": False,
                "threshold": 30,
                "timeframe": "15m",
                "period": 14,
            },
            "pairs": [],
        }

    def _load_state(self) -> None:
        try:
            self.state = load_json(self.state_path, self._default_state())
            self.logger.info("State loaded from %s", self.state_path)
        except (OSError, ValueError, TypeError) as exc:
            self.logger.error("Failed to load state: %s", exc)
            self.state = self._default_state()

        self._pair_stats: dict[str, dict[str, float]] = {}
        api_state = self.state.get("api", {})
        exchange_id = str(api_state.get("exchange_id", "mexc")).lower()
        exchange_label = next(
            (label for label, ex_id in self._exchange_labels.items() if ex_id == exchange_id),
            "MEXC",
        )
        self.exchange_var.set(exchange_label)
        self.api_key_entry.insert(0, api_state.get("key", ""))
        self.api_secret_entry.insert(0, api_state.get("secret", ""))
        self.save_credentials_var.set(bool(api_state.get("save_credentials", False)))

        strategy = self.state.get("strategy", {})
        self.settings_entries["take_profit"].insert(0, str(strategy.get("take_profit_pct", "")))
        self.settings_entries["base_order"].insert(0, str(strategy.get("base_order_usdt", "")))
        self.settings_entries["safety_orders"].insert(0, str(strategy.get("safety_orders_count", "")))
        self.settings_entries["safety_step"].insert(0, str(strategy.get("safety_step_pct", "")))
        self.settings_entries["volume_multiplier"].insert(0, str(strategy.get("volume_multiplier", "")))
        self.settings_entries["price_range"].insert(0, str(strategy.get("price_range_pct", "")))
        self.settings_entries["fee"].delete(0, tk.END)
        self.settings_entries["fee"].insert(0, str(strategy.get("fee_pct", 0.1)))
        self.use_safety_var.set(bool(strategy.get("use_safety_orders", True)))
        self.use_market_entry_var.set(bool(strategy.get("use_market_entry", False)))

        rsi = self.state.get("rsi", {})
        self.use_rsi_var.set(bool(rsi.get("enabled", False)))
        self.rsi_value_entry.delete(0, tk.END)
        self.rsi_value_entry.insert(0, str(rsi.get("threshold", 30)))
        self.rsi_timeframe.set(str(rsi.get("timeframe", "15m")))
        self.rsi_period_entry.delete(0, tk.END)
        self.rsi_period_entry.insert(0, str(rsi.get("period", 14)))

        for item in self.pairs_tree.get_children():
            self.pairs_tree.delete(item)
        for pair in self.state.get("pairs", []):
            pair_name = pair.get("pair", "")
            self._pair_stats[str(pair_name)] = {
                "invested_usdt": float(pair.get("invested_usdt", 0.0)),
            }
            self.pairs_tree.insert(
                "",
                tk.END,
                values=(
                    pair_name,
                    pair.get("status", "STOPPED"),
                    pair.get("cycle", 0),
                    pair.get("pnl_usdt", 0.0),
                    pair.get("pnl_pct", 0.0),
                    pair.get("closed_trades", 0),
                ),
            )

    def _collect_pairs(self) -> list[dict]:
        pairs = []
        for item in self.pairs_tree.get_children():
            values = self.pairs_tree.item(item, "values")
            pair_key = str(values[0])
            pair_data = {
                "pair": pair_key,
                "status": str(values[1]),
                "cycle": int(values[2]),
                "pnl_usdt": float(values[3]),
                "pnl_pct": float(values[4]),
                "closed_trades": int(values[5]),
                "invested_usdt": float(self._pair_stats.get(pair_key, {}).get("invested_usdt", 0.0)),
            }
            pairs.append(pair_data)
        return pairs

    def _collect_state(self) -> dict:
        api_key = self.api_key_entry.get().strip()
        api_secret = self.api_secret_entry.get().strip()
        save_credentials = bool(self.save_credentials_var.get())
        exchange_label = self.exchange_var.get() or "MEXC"
        exchange_id = self._exchange_labels.get(exchange_label, "mexc")
        if not save_credentials:
            api_key = ""
            api_secret = ""

        strategy = {
            "take_profit_pct": self.settings_entries["take_profit"].get().strip(),
            "base_order_usdt": self.settings_entries["base_order"].get().strip(),
            "safety_orders_count": self.settings_entries["safety_orders"].get().strip(),
            "safety_step_pct": self.settings_entries["safety_step"].get().strip(),
            "volume_multiplier": self.settings_entries["volume_multiplier"].get().strip(),
            "price_range_pct": self.settings_entries["price_range"].get().strip(),
            "fee_pct": self.settings_entries["fee"].get().strip(),
            "use_safety_orders": bool(self.use_safety_var.get()),
            "use_market_entry": bool(self.use_market_entry_var.get()),
        }

        def to_float(value: str, default: float) -> float | str:
            if value == "":
                return ""
            try:
                return float(value)
            except ValueError:
                return default

        def to_int(value: str, default: int) -> int:
            try:
                return int(value)
            except ValueError:
                return default

        strategy_state = {
            "take_profit_pct": to_float(strategy["take_profit_pct"], 1.0),
            "base_order_usdt": to_float(strategy["base_order_usdt"], 10.0),
            "safety_orders_count": to_int(strategy["safety_orders_count"], 5),
            "safety_step_pct": to_float(strategy["safety_step_pct"], 2.0),
            "volume_multiplier": to_float(strategy["volume_multiplier"], 1.2),
            "price_range_pct": to_float(strategy["price_range_pct"], ""),
            "fee_pct": to_float(strategy["fee_pct"], 0.1),
            "use_safety_orders": strategy["use_safety_orders"],
            "use_market_entry": strategy["use_market_entry"],
        }

        rsi_threshold = self.rsi_value_entry.get().strip()
        rsi_period = self.rsi_period_entry.get().strip()
        rsi_state = {
            "enabled": bool(self.use_rsi_var.get()),
            "threshold": to_float(rsi_threshold, 30),
            "timeframe": self.rsi_timeframe.get().strip() or "15m",
            "period": to_int(rsi_period, 14),
        }

        return {
            "api": {
                "exchange_id": exchange_id,
                "key": api_key,
                "secret": api_secret,
                "save_credentials": save_credentials,
            },
            "strategy": strategy_state,
            "rsi": rsi_state,
            "pairs": self._collect_pairs(),
        }

    def _save_state(self) -> None:
        self.state = self._collect_state()
        try:
            save_json(self.state_path, self.state)
            self.logger.info("State saved to %s", self.state_path)
        except (OSError, ValueError, TypeError) as exc:
            self.logger.error("Failed to save state: %s", exc)


def setup_logger(log_queue: queue.Queue) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("dca_bot")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler("logs/bot.log")
    file_handler.setFormatter(formatter)

    queue_handler = QueueHandler(log_queue)
    queue_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(queue_handler)

    return logger


def main() -> None:
    log_queue = queue.Queue()
    worker_event_queue = queue.Queue()
    logger = setup_logger(log_queue)
    logger.info("Application started")

    root = tk.Tk()
    app = App(root, logger, log_queue, worker_event_queue)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
