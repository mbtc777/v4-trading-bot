#!/usr/bin/env python3
"""
Backtest Engine v4 — trailing pyramiding + asset ΔΔ + per-asset optimization.

Ключевые изменения против v3:
  - Новая trailing-догрузка: после первой догрузки сделка переходит
    в trailing TP режим (шаги, фиксация частей позиции, TP level).
  - ΔΔ фильтр самого актива (не только BTC).
  - Пре-бэктест для подбора параметров под актив.
  - Global breakeven stop после догрузки (защита PnL).
  - Проверка: догрузка не должна ухудшать PnL.

Конфигурируется через BacktestParams — любые комбинации.
Результаты разбиты по фазам BTC для адаптивной оптимизации.

Использование:
    from engine_v4 import BacktestRunner, BacktestParams, DEFAULT_PARAMS_V4
    runner = BacktestRunner(params=DEFAULT_PARAMS_V4)
    runner.load_btc_data()
    runner.run(max_assets=50)
    summary = runner.summary()
"""
import sys
import os
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.downloader import download_symbol_range, load_cached, get_top_usdt_pairs
import market_phase as mp

# ─── Constants ─────────────────────────────────────────────────────────────
BTC_SYMBOL = "BTCUSDT"
START_TS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
END_TS   = int(datetime(2026, 6, 4, tzinfo=timezone.utc).timestamp())


# ─── Params ────────────────────────────────────────────────────────────────

@dataclass
class BacktestParams:
    """Все настраиваемые параметры бэктеста v3. JSON-сериализуемо."""

    # ── SL ──
    sl_atr_mult: float = 2.0           # SL = ATR × this (запасной метод, пока не используем)
    sl_fallback_pct: float = 2.0       # если ATR нет
    max_sl_pct: float = 2.4            # ограничение SL сверху (%)

    # ── TP ──
    tp_count: int = 5                  # 3, 5 или 10 уровней
    tp_first_rr: float = 0.8           # первый TP = SL × this

    # ── Импульс (Option 3) ──
    impulse_enabled: bool = False
    impulse_max_body_ratio: float = 1.2
    impulse_max_deviation: float = 0.5
    impulse_max_slope: float = 0.15
    impulse_lookback: int = 4

    # ── RSI gates ──
    rsi_long_min: float = 40.0
    rsi_short_max: float = 60.0

    # ── BTC фильтры ──
    use_btc_phase: bool = True
    use_dd_filter: bool = True
    dd_threshold: float = 20_000_000

    # ── Объём ──
    min_volume_24h: float = 10_000_000

    # ── Депо ──
    deposit: float = 10_000

    # ── Таймаут ──
    timeout_bars: int = 48

    # ════════════════════════════════════════════════════
    #  НОВОЕ v3
    # ════════════════════════════════════════════════════

    # ── Риск-менеджмент ──
    max_deposit_risk_pct: float = 0.8      # макс риск на позицию (% от депо)
    global_stop_loss_pct: float = 2.4      # при -2.4% всех открытых -> стоп
    max_drawdown_pct: float = 50.0         # отключено для тестов (было 3%)

    # ── Комиссии Binance Futures ──
    commission_maker: float = 0.0002       # 0.02%
    commission_taker: float = 0.0004       # 0.04%

    # ── Усреднение (микро-инкременты) ──
    avg_enabled: bool = False              # вкл/выкл усреднение
    avg_risk_per_level: float = 1.5        # макс риск на все усреднения (% от депо)
    avg_incremental_pct: float = 2.0       # каждая добавка = X% от тек. позиции
    avg_max_increments: int = 50           # макс микро-усреднений
    avg_min_move_pct: float = 0.5          # мин движение против для 1-го триггера
    avg_use_delta_delta: bool = True       # подтверждение ΔΔ на экстремумах
    avg_only_if_profit_others: bool = True # только если другие сделки в плюсе

    # ── Догрузка (pyramiding) ──
    pyramiding_enabled: bool = False
    pyramiding_trigger_pct: float = 2.0    # +2% от входа
    pyramiding_size_pct: float = 0.15      # +15% от начальной позиции
    pyramiding_max_count: int = 5          # макс догрузок
    pyramiding_only_trailing_profit: bool = True  # только если трейлинг в +

    # ── Догрузка: ΔΔ подтверждение ──
    pyramiding_use_dd: bool = True            # вкл/выкл ΔΔ фильтр для догрузки
    pyramiding_dd_lookback: int = 5           # период для среднего ΔΔ
    pyramiding_dd_min_ratio: float = 1.5      # мин отношение ΔД/средний ΔΔ для догрузки
    pyramiding_volume_ratio: float = 1.2      # мин отношение объёма к среднему

    # ════════════════════════════════════════════════════
    #  НОВОЕ v4 — trailing pyramiding + asset DD + opt
    # ════════════════════════════════════════════════════

    # ── Новая догрузка: trailing TP ──
    pyramiding_trailing_mode: str = "step"       # "step" | "off" — режим трейлинга
    pyramiding_trailing_step_pct: float = 1.0    # шаг трейлинга (%)
    pyramiding_trailing_alloc_pct: float = 0.25  # % оставшейся позиции на каждом шаге
    pyramiding_trailing_buffer_steps: int = 1    # сколько шагов отстаёт TP

    # ── Критический стоп для позиции в целом ──
    pyramiding_global_breakeven: bool = True     # стоп всей позиции на breakeven после догрузки

    # ── ΔΔ самого актива ──
    use_asset_dd_filter: bool = True             # фильтр ΔΔ самого актива
    asset_dd_threshold: float = 500000           # порог ΔΔ для актива

    # ── Пре-бэктест для подбора параметров ──
    enable_per_asset_optimization: bool = False  # вкл/выкл (по умолч. выкл для скорости)

    # ── Пре-трейд бэктест-фильтр ──
    pre_trade_backtest_filter: bool = False      # фильтр: бэктест актива перед сигналом
    pre_trade_min_pnl: float = 0                  # мин PnL для прохода ($)
    pre_trade_min_winrate: float = 50.0          # мин винрейт для прохода (%)

    # ── Фильтр аномальной волатильности (анти-стопхантинг) ──
    anti_stophunting_filter: bool = False         # вкл/выкл
    anti_stophunting_tail_threshold: float = 2.5  # макс. отношение тени к ATR (свеча считается аномальной если tail > atr * this)
    anti_stophunting_lookback: int = 20           # сколько свечей назад проверять
    anti_stophunting_max_anomalies: int = 2       # сколько аномальных свечей допустимо за lookback

    # ── Фильтр ATR актива ──
    asset_atr_filter: bool = False                # вкл/выкл
    asset_atr_multiplier: float = 3.0             # макс ATR% относительно медианы (если > this — блокируем)

    # ── Зонное усреднение (AVG по проторговке + разворот) ──
    avg_zone_enabled: bool = False                 # вкл/выкл зонный режим
    avg_zone_start_move_pct: float = 1.5           # мин движение против до начала зоны (%)
    avg_zone_min_bars: int = 3                     # мин свечей в зоне проторговки
    avg_zone_max_spread_pct: float = 1.0           # макс разброс цены в зоне (%)
    avg_zone_confirmation_bars: int = 1            # свечей в нужную сторону после зоны

    # ── Настройки фаз ──
    skip_accumulation: bool = True         # не торговать в accumulation
    volatility_filter_enabled: bool = True # определять подтип фазы


DEFAULT_PARAMS_V4 = BacktestParams()


# ─── Param grid for optimizer ─────────────────────────────────────────────

PARAM_GRID = {
    "sl_atr_mult":         [1.0, 1.5, 2.0, 2.5, 3.0],
    "sl_fallback_pct":     [1.0, 1.5, 2.0, 2.5, 3.0],
    "max_sl_pct":          [1.6, 2.0, 2.4, 3.2, 4.0, 8.0],
    "tp_count":            [3, 5, 10],
    "tp_first_rr":         [0.3, 0.4, 0.5, 0.6, 0.8],
    "impulse_enabled":     [True, False],
    "impulse_max_body_ratio": [0.8, 1.0, 1.2, 1.5, 2.0],
    "impulse_max_deviation":  [0.3, 0.5, 0.7, 1.0],
    "impulse_max_slope":      [0.10, 0.12, 0.15, 0.20, 0.25],
    "impulse_lookback":       [3, 4, 5],
    "avg_enabled":         [False, True],
    "avg_risk_per_level":  [0.5, 1.0, 1.5, 2.0],
    "avg_incremental_pct": [1.0, 2.0, 3.0],
    "avg_max_increments":  [20, 30, 50],
    "avg_min_move_pct":    [0.3, 0.5, 0.8],
    "avg_use_delta_delta": [True, False],
    "avg_only_if_profit_others": [True, False],
    "pyramiding_enabled":  [True, False],
    "pyramiding_trigger_pct": [1.5, 2.0, 3.0],
    "pyramiding_size_pct": [0.1, 0.15, 0.2],
    "pyramiding_max_count":[3, 5, 10],
    "pyramiding_only_trailing_profit": [True, False],
    "pyramiding_trailing_mode": ["step", "off"],
    "pyramiding_trailing_step_pct": [0.5, 1.0, 2.0],
    "pyramiding_trailing_alloc_pct": [0.15, 0.25, 0.5],
    "pyramiding_trailing_buffer_steps": [1, 2, 3],
    "pyramiding_global_breakeven": [True, False],
    "pyramiding_use_dd": [True, False],
    "pyramiding_dd_min_ratio": [1.2, 1.5, 2.0],
    "pyramiding_volume_ratio": [1.0, 1.2, 1.5],
    "max_deposit_risk_pct":[1.0, 1.5, 2.0],
    "global_stop_loss_pct":[2.0, 2.4, 3.0],
    "max_drawdown_pct":    [2.5, 3.0, 4.0],
    "skip_accumulation":   [True, False],
    "volatility_filter_enabled": [True, False],
    "rsi_long_min":        [35, 40, 45],
    "rsi_short_max":       [55, 60, 65],
    "use_dd_filter":       [True, False],
    "dd_threshold":        [10_000_000, 20_000_000, 50_000_000],
    "use_asset_dd_filter": [True, False],
    "asset_dd_threshold":  [250000, 500000, 1000000],
    "enable_per_asset_optimization": [False, True],
    "min_volume_24h":      [5_000_000, 10_000_000, 20_000_000],
    "timeout_bars":        [24, 48, 72],
    "commission_taker":    [0.0004],
    "commission_maker":    [0.0002],
}


def random_params(rng=None):
    """Сгенерировать случайную комбинацию параметров из PARAM_GRID."""
    import random as _random
    if rng is None:
        rng = _random.Random()
    p = {}
    for key, choices in PARAM_GRID.items():
        p[key] = rng.choice(choices)
    return BacktestParams(**p)


def params_to_key(p: BacktestParams) -> str:
    """Уникальный ключ для дедупликации комбинаций."""
    return json.dumps(asdict(p), sort_keys=True)


# ─── ATR ───────────────────────────────────────────────────────────────────

def _compute_atr(candles: list, end_idx: int, period: int = 14) -> float:
    """ATR до end_idx включительно."""
    if end_idx < period:
        return None
    relevant = candles[end_idx - period : end_idx]
    if len(relevant) < period:
        return None
    trs = []
    for i in range(len(relevant)):
        hi = float(relevant[i][2])
        lo = float(relevant[i][3])
        prev_close = float(relevant[i-1][4]) if i > 0 else hi
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)


# ─── Impulse Detection ─────────────────────────────────────────────────────

def detect_impulse(candles_1h: list, end_idx: int, params: BacktestParams) -> bool:
    """
    Проверить последние N свечей на импульсность.
    True = это импульс, не брать.
    """
    if not params.impulse_enabled:
        return False
    lookback = params.impulse_lookback
    if end_idx < lookback + 1:
        return False
    relevant = candles_1h[end_idx - lookback : end_idx]
    if len(relevant) < lookback:
        return False
    atr = _compute_atr(candles_1h, end_idx)
    if atr is None or atr <= 0:
        return False
    closes = [float(c[4]) for c in relevant]
    slope = (closes[-1] - closes[0]) / lookback
    atr_pct = atr / float(relevant[-1][4])
    max_slope_pct = params.impulse_max_slope * atr_pct * 100
    slope_pct = slope / closes[0] * 100
    if abs(slope_pct) > max_slope_pct:
        return True
    for i, c in enumerate(relevant):
        o, h, l, cls = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        body = abs(cls - o)
        body_ratio = body / atr if atr > 0 else 0
        if body_ratio > params.impulse_max_body_ratio:
            return True
        expected = closes[0] + slope * (i + 1)
        deviation = abs(cls - expected) / atr if atr > 0 else 0
        if deviation > params.impulse_max_deviation:
            return True
    return False


# ─── Trend Detection ───────────────────────────────────────────────────────

def detect_trend(candles_1h, lookback=3, allow_ongoing=True):
    """Детект трендовых сигналов."""
    events = []
    prev_trend = None
    streak = 0
    for i, c in enumerate(candles_1h):
        ts = c[0]
        o, h, l, cls = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        curr = "UP" if cls > o else ("DOWN" if cls < o else "FLAT")
        if curr == prev_trend and curr != "FLAT":
            streak += 1
        else:
            streak = 1 if curr != "FLAT" else 0
        prev_trend = curr
        if streak >= lookback and curr != "FLAT":
            events.append({
                "timestamp": ts, "index": i, "trend": curr,
                "price": cls, "open": o, "high": h, "low": l, "streak": streak,
            })
            if not allow_ongoing:
                streak = 0
    return events


# ─── SL/TP v3 — риск-ориентированный ─────────────────────────────────────

def _calc_risk_sl_pct(entry_price: float, direction: str,
                      deposit: float, max_deposit_risk_pct: float,
                      max_sl_pct: float) -> float:
    """
    SL рассчитывается через риск от депо:
      max_risk_usdt = deposit * max_deposit_risk_pct / 100
      sl_pct = max_risk_usdt / position * 100
    """
    max_risk_usdt = deposit * max_deposit_risk_pct / 100  # $150 на $10k
    # Используем entry_price как proxy для 1 единицы (1 USDT позиция = sl_pct = max_risk_pct / 1 * 100?)
    # sl_pct = max_risk_usdt / position * 100, где position —
    # это размер позиции, который будет посчитан ниже.
    # Здесь возвращаем просто % от entry_price на основе риска.
    # Позиция = deposit * size_factor, где size_factor ~ 1 / sl_pct
    # Решаем: sl_pct = max_risk_usdt / position * 100
    # position = deposit * (1 / sl_pct)   ← примерно
    # => sl_pct = max_risk_usdt / (deposit / sl_pct) * 100
    # => sl_pct^2 = max_risk_usdt / deposit * 100 = (deposit * max_deposit_risk_pct / 100) / deposit * 100
    # => sl_pct = sqrt(max_deposit_risk_pct)
    # Это даёт: при 1.5% → sl_pct ~ 1.22%, при 2% → 1.41%
    # Но лучше: sl_pct = max_deposit_risk_pct (как запас)
    # Используем простой подход: size_factor = 1.0, sl_pct = max_deposit_risk_pct * 0.6
    # Нет, сделаем как в ТЗ: max_risk_usdt = deposit * max_deposit_risk_pct / 100
    # position будет = deposit * (1.0), тогда sl_pct = max_risk_usdt / deposit * 100 = max_deposit_risk_pct
    # Т.е. sl_pct = max_deposit_risk_pct при full position
    # Но position может быть меньше. В compute_sl_tp мы решим итеративно.
    sl_pct = max_deposit_risk_pct  # baseline
    sl_pct = min(max(sl_pct, 0.3), max_sl_pct)
    return sl_pct


def _calc_position_size(deposit: float, sl_pct: float,
                        max_deposit_risk_pct: float) -> float:
    """
    Размер позиции: чтобы max_risk_usdt = deposit * max_deposit_risk_pct / 100
    position = max_risk_usdt / (sl_pct / 100) = deposit * max_deposit_risk_pct / sl_pct
    """
    if sl_pct <= 0:
        return deposit  # fallback
    return deposit * max_deposit_risk_pct / sl_pct


def compute_sl_tp(entry_price: float, direction: str,
                  prev_candle: dict, params: BacktestParams,
                  candles_1h: list, idx: int) -> dict:
    """
    SL через риск от депо. TP — configurable count + first RR.

    Новая логика:
    - sl_pct = min(max(sl_pct, 0.3), max_sl_pct)
    - position = deposit * max_deposit_risk_pct / sl_pct
    """
    # SL через risk-based
    max_risk_usdt = params.deposit * params.max_deposit_risk_pct / 100.0
    # sl_pct — это % от entry_price
    # position = max_risk_usdt / (sl_pct / 100)
    # Если позиция = депозит * 1, то sl_pct = max_deposit_risk_pct
    # Мы не знаем position заранее, решаем просто:
    sl_pct = params.max_deposit_risk_pct  # baseline
    atr = _compute_atr(candles_1h, idx)
    if atr and atr > 0:
        atr_pct = atr / entry_price * 100
        # risk-based SL не может быть меньше ATR*sl_atr_mult (для запаса)
        atr_based = max(atr_pct * params.sl_atr_mult, 0.3)
        sl_pct = max(sl_pct, atr_based)
    else:
        sl_pct = max(sl_pct, params.sl_fallback_pct)

    # Ограничения
    sl_pct = min(max(sl_pct, 0.3), params.max_sl_pct)

    if direction == "LONG":
        sl = entry_price * (1 - sl_pct / 100.0)
    else:
        sl = entry_price * (1 + sl_pct / 100.0)

    # Размер позиции через risk budget
    position = _calc_position_size(params.deposit, sl_pct,
                                   params.max_deposit_risk_pct)

    # TP уровни
    tp_rrs = _generate_tp_rrs(params.tp_count, params.tp_first_rr)
    tp_alloc = _generate_tp_alloc(params.tp_count)

    tps = []
    for rr in tp_rrs:
        tp_pct = sl_pct * rr
        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100.0)
        else:
            tp_price = entry_price * (1 - tp_pct / 100.0)
        alloc = tp_alloc.get(rr, 0)
        tps.append({"level": rr, "price": round(tp_price, 8), "alloc": alloc})

    return {
        "sl": sl,
        "sl_pct": round(sl_pct, 3),
        "tps": tps,
        "position_usdt": round(position, 2),
    }


def _generate_tp_rrs(count: int, first_rr: float) -> list:
    if count == 3:
        return [first_rr, 1.0, 2.0]
    if count == 5:
        return [first_rr, 0.8, 1.2, 2.0, 4.0]
    return [0.4, 0.8, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def _generate_tp_alloc(count: int) -> dict:
    if count == 3:
        return {0.4: 0.40, 1.0: 0.35, 2.0: 0.25}
    if count == 5:
        return {0.4: 0.25, 0.8: 0.20, 1.2: 0.20, 2.0: 0.20, 4.0: 0.15}
    return {0.4: 0.20, 0.8: 0.20, 1.2: 0.20, 2.0: 0.15, 3.0: 0.10,
            4.0: 0.05, 5.0: 0.03, 6.0: 0.03, 7.0: 0.02, 8.0: 0.02}


# ─── Trade Simulation v3 ─────────────────────────────────────────────────

def simulate_trade(signal: dict, candles_1h: list, params: BacktestParams,
                   global_state: dict = None) -> dict:
    """
    Симуляция одной сделки на 1h свечах.

    Особенности v3:
    - Risk-based SL через депозит
    - Комиссии taker на TP/SL/AVG/pyramiding entry
    - Усреднение (1-е и 2-е уровни)
    - Догрузка (pyramiding)
    - TP проверяется ПЕРЕД SL в той же свече
    - Глобальный риск-менеджмент (total_open_pnl)

    global_state: dict с ключами:
      total_open_pnl — используется для avg_only_if_profit_others
      deposit — используется для глобальных стопов
      get_delta_delta_at — функция lookup ΔΔ по timestamp
    """
    direction = signal["direction"]
    entry_ts = signal["timestamp"]
    entry_price = signal["price"]
    idx = signal["index"]
    prev_candle = signal["prev_candle"]

    # SL/TP
    sl_tp = compute_sl_tp(entry_price, direction, prev_candle,
                          params, candles_1h, idx)
    sl = sl_tp["sl"]
    sl_pct = sl_tp["sl_pct"]
    tps = sl_tp["tps"]
    initial_position = sl_tp["position_usdt"]

    candles_after = candles_1h[idx + 1:]
    if not candles_after:
        return None

    # ── Состояние сделки ──
    remaining = initial_position
    total_pnl = 0.0
    fills = []
    exit_reason = "none"
    exit_price = entry_price
    exit_ts = entry_ts
    bars_held = 0

    # Для усреднения
    avg_count = 0
    current_entry = entry_price
    current_position = initial_position

    # Для догрузки
    pyramiding_count = 0
    pyramiding_entries = [{"price": entry_price, "size": initial_position}]
    # Трейлинг-стоп для догрузки — самая последняя цена, где трейлинг в +
    trailing_trigger_price = entry_price  # цена, выше которой мы в безубытке
    trailing_high = entry_price if direction == "LONG" else entry_price

    # ── v4: trailing mode state ──
    pyramiding_trailing_active = False      # вкл после первой догрузки
    pyramiding_frontier = entry_price if direction == "LONG" else entry_price
    pyramiding_tp_level = None              # текущий trailing TP level
    pyramiding_last_booked_steps = 0        # последний зафиксированный шаг
    pyramiding_first_pyra_entry = None     # цена первой догрузки
    pyramiding_breakeven_set = False        # breakeven стоп установлен

    # Для отслеживания новых фич
    pyramiding_dd_used = False
    pyramiding_tp_trailed = False

    # Для распределения аллокаций между pyramiding-частями:
    # каждая часть TP-аллокации применяется к proportion от total_position

    for ci, c in enumerate(candles_after):
        ts = c[0]
        o, h, l, cls = float(c[1]), float(c[2]), float(c[3]), float(c[4])

        if remaining <= 0:
            exit_reason = "all_TP_hit"
            exit_price = fills[-1]["price"] if fills else current_entry
            exit_ts = ts
            bars_held += 1
            break

        # ── Trailing-high / low + frontier update ──
        if direction == "LONG":
            if cls > trailing_high:
                trailing_high = cls
            if pyramiding_trailing_active and cls > pyramiding_frontier:
                pyramiding_frontier = cls
        else:
            if cls < trailing_high:
                trailing_high = cls
            if pyramiding_trailing_active and cls < pyramiding_frontier:
                pyramiding_frontier = cls

        # ════════════════════════════════════════════════════════════════
        # ── Догрузка (pyramiding) — ПЕРЕД TP/SL ──
        # ════════════════════════════════════════════════════════════════
        # Фаза 1: базовая сделка — SL по ATR, стандартные TP
        # Фаза 2: после первой догрузки → trailing mode
        if (params.pyramiding_enabled
                and pyramiding_count < params.pyramiding_max_count
                and remaining > 0):
            # Проверка: цена прошла +trigger_pct от входа
            if direction == "LONG":
                move_from_entry = (cls - pyramiding_entries[0]["price"]) / pyramiding_entries[0]["price"] * 100
                pyramiding_profit_ok = True
                if params.pyramiding_only_trailing_profit:
                    pyramiding_profit_ok = trailing_high >= pyramiding_entries[0]["price"] * (1 + params.pyramiding_trigger_pct / 100)
            else:
                move_from_entry = (pyramiding_entries[0]["price"] - cls) / pyramiding_entries[0]["price"] * 100
                pyramiding_profit_ok = True
                if params.pyramiding_only_trailing_profit:
                    pyramiding_profit_ok = trailing_high <= pyramiding_entries[0]["price"] * (1 - params.pyramiding_trigger_pct / 100)

            if move_from_entry >= params.pyramiding_trigger_pct and pyramiding_profit_ok:
                is_impulse = detect_impulse(candles_1h, idx + ci + 1, params)

                # ── ΔΔ подтверждение догрузки ──
                dd_confirmed = True
                dd_strong = False
                if params.pyramiding_use_dd:
                    pyramiding_dd_used = True
                    local_dd = _compute_local_dd(candles_1h, idx + ci + 1, params.pyramiding_dd_lookback)
                    dd_values = []
                    for ddi in range(idx + ci + 1 - params.pyramiding_dd_lookback, idx + ci + 1):
                        ddv = _compute_local_dd(candles_1h, ddi, 1)
                        dd_values.append(ddv)
                    avg_dd = sum(dd_values) / len(dd_values) if dd_values else 0

                    if direction == "LONG":
                        dd_ok = local_dd > avg_dd * params.pyramiding_dd_min_ratio
                        dd_strong = local_dd > avg_dd * (params.pyramiding_dd_min_ratio * 2)
                    else:
                        dd_ok = local_dd < avg_dd * params.pyramiding_dd_min_ratio * (-1)
                        dd_strong = local_dd < avg_dd * (params.pyramiding_dd_min_ratio * 2) * (-1)

                    vol_span = min(params.pyramiding_dd_lookback, idx + ci + 1)
                    avg_vol = 0.0
                    if vol_span > 0:
                        avg_vol = sum(float(candles_1h[idx + ci + 1 - ddi][5]) for ddi in range(1, vol_span + 1)) / vol_span
                    current_vol = float(c[5])
                    vol_ok = True
                    if avg_vol > 0:
                        vol_ok = current_vol >= avg_vol * params.pyramiding_volume_ratio

                    dd_confirmed = dd_ok and vol_ok

                if dd_confirmed:
                    if is_impulse:
                        pyramiding_size = initial_position * params.pyramiding_size_pct * 0.5
                    else:
                        pyramiding_size = initial_position * params.pyramiding_size_pct
                    if dd_strong:
                        pyramiding_size *= 1.5
                    pyramiding_size = max(pyramiding_size, 1.0)

                    commission = pyramiding_size * params.commission_taker
                    pnl_commission = -commission
                    total_pnl += pnl_commission
                    fills.append({
                        "type": f"PYRAMID{pyramiding_count + 1}",
                        "price": cls,
                        "usdt": round(pnl_commission, 2),
                        "qty": round(pyramiding_size, 2),
                        "commission": round(commission, 4),
                    })
                    current_position += pyramiding_size
                    remaining += pyramiding_size
                    pyramiding_entries.append({"price": cls, "size": pyramiding_size})
                    pyramiding_count += 1

                    # ── После первой догрузки → включаем trailing mode ──
                    if pyramiding_count == 1 and params.pyramiding_trailing_mode == "step":
                        pyramiding_trailing_active = True
                        pyramiding_first_pyra_entry = cls
                        pyramiding_frontier = cls
                        pyramiding_last_booked_steps = 0
                        pyramiding_tp_level = None

        # ════════════════════════════════════════════════════════════════
        # ── Trailing TP mode (после первой догрузки) ──
        # ════════════════════════════════════════════════════════════════
        if pyramiding_trailing_active and remaining > 0:
            step_pct = params.pyramiding_trailing_step_pct
            alloc_pct = params.pyramiding_trailing_alloc_pct
            buffer_steps = params.pyramiding_trailing_buffer_steps
            first_pyra = pyramiding_first_pyra_entry

            # Обновляем frontier (макс цена с момента активации trailing)
            if direction == "LONG":
                if cls > pyramiding_frontier:
                    pyramiding_frontier = cls
            else:
                if cls < pyramiding_frontier:
                    pyramiding_frontier = cls

            # TP level = frontier - buffer_steps * step_pct
            if direction == "LONG":
                pyramiding_tp_level = pyramiding_frontier * (1 - buffer_steps * step_pct / 100.0)
            else:
                pyramiding_tp_level = pyramiding_frontier * (1 + buffer_steps * step_pct / 100.0)

            # Сколько шагов прошла цена от первой догрузки
            if direction == "LONG" and first_pyra > 0:
                steps_passed = int((pyramiding_frontier - first_pyra) / first_pyra * 100.0 / step_pct)
            elif direction == "SHORT" and first_pyra > 0:
                steps_passed = int((first_pyra - pyramiding_frontier) / first_pyra * 100.0 / step_pct)
            else:
                steps_passed = 0

            # Фиксируем alloc_pct на каждом новом шаге
            if steps_passed > pyramiding_last_booked_steps and remaining > 0:
                for step_n in range(pyramiding_last_booked_steps + 1, steps_passed + 1):
                    if remaining <= 0:
                        break
                    # Цена фиксации на этом шаге
                    if direction == "LONG":
                        step_price = first_pyra * (1 + step_n * step_pct / 100.0)
                        # Не фиксируем выше текущей close (цена может не дойти)
                        step_price = min(step_price, cls)
                    else:
                        step_price = first_pyra * (1 - step_n * step_pct / 100.0)
                        step_price = max(step_price, cls)

                    # Размер фиксации = alloc_pct% от оставшейся позиции
                    step_qty = remaining * alloc_pct
                    if step_qty < 1.0:
                        step_qty = remaining  # close remainder if too small

                    # PnL от этой фиксации
                    pnl = _calc_pnl_from_entries(step_qty, step_price,
                                                 current_entry, direction,
                                                 params, is_sl=False)
                    total_pnl += pnl
                    fills.append({
                        "type": f"TRAIL_STEP{step_n}",
                        "price": step_price,
                        "usdt": round(pnl, 2),
                        "qty": round(step_qty, 2),
                    })
                    remaining -= step_qty

                pyramiding_last_booked_steps = steps_passed
                pyramiding_tp_trailed = True

            # Если цена ниже/выше TP level → закрываем остаток
            tp_hit = False
            if direction == "LONG" and l <= pyramiding_tp_level:
                tp_hit = True
                close_price = min(pyramiding_tp_level, cls)
            elif direction == "SHORT" and h >= pyramiding_tp_level:
                tp_hit = True
                close_price = max(pyramiding_tp_level, cls)

            if tp_hit and remaining > 0:
                pnl = _calc_pnl_from_entries(remaining, close_price,
                                             current_entry, direction,
                                             params, is_sl=False)
                total_pnl += pnl
                fills.append({
                    "type": "TRAIL_TP",
                    "price": close_price,
                    "usdt": round(pnl, 2),
                    "qty": round(remaining, 2),
                })
                exit_reason = "trailing_tp"
                exit_price = close_price
                exit_ts = ts
                bars_held += 1
                remaining = 0
                break

        # ── TP FIRST: проверяем все TP в этой свече ──
        # В trailing mode стандартные TP не используются
        if not pyramiding_trailing_active:
            for tp in tps:
                alloc = tp["alloc"] * initial_position
                if alloc <= 0 or remaining <= 0:
                    continue
                hit = (direction == "LONG" and h >= tp["price"]) or \
                      (direction == "SHORT" and l <= tp["price"])
                if hit:
                    fill_qty = min(alloc, remaining)
                    pnl = _calc_pnl_from_entries(fill_qty, tp["price"],
                                                 current_entry, direction, params, is_sl=False)
                    total_pnl += pnl
                    fills.append({"type": f"TP{tp['level']}", "price": tp["price"],
                                  "usdt": round(pnl, 2), "qty": round(fill_qty, 2)})
                    remaining -= fill_qty

        # ── Breakeven stop (после догрузки) ──
        # Добавляем безубыточный стоп для всей позиции, чтобы даже при SL
        # позиция в целом не ушла в минус
        breakeven_hit = False
        if (params.pyramiding_global_breakeven
                and pyramiding_count > 0
                and remaining > 0
                and not pyramiding_breakeven_set):
            # Рассчитываем средневзвешенный вход
            total_sz = sum(e["size"] for e in pyramiding_entries)
            avg_entry = sum(e["price"] * e["size"] for e in pyramiding_entries) / total_sz if total_sz > 0 else entry_price
            # Breakeven стоп — чуть ниже/выше среднего входа
            if direction == "LONG":
                be_stop = avg_entry * (1 - 0.01 / 100.0)  # 0.01% ниже
            else:
                be_stop = avg_entry * (1 + 0.01 / 100.0)  # 0.01% выше
            # Проверяем breakeven
            be_check = (direction == "LONG" and l <= be_stop) or \
                       (direction == "SHORT" and h >= be_stop)
            if be_check:
                # Закрываем остаток по be_stop (≈ breakeven)
                pnl = _calc_pnl_from_entries(remaining, be_stop,
                                             current_entry, direction,
                                             params, is_sl=False)
                total_pnl += pnl
                fills.append({
                    "type": "BREAKEVEN",
                    "price": be_stop,
                    "usdt": round(pnl, 2),
                    "qty": round(remaining, 2),
                })
                exit_reason = "breakeven_stop"
                exit_price = be_stop
                exit_ts = ts
                bars_held += 1
                remaining = 0
                breakeven_hit = True
            pyramiding_breakeven_set = True

        # ── Global breakeven: проверяем каждый раз после установки ──
        if (params.pyramiding_global_breakeven
                and pyramiding_breakeven_set
                and not breakeven_hit
                and remaining > 0):
            total_sz = sum(e["size"] for e in pyramiding_entries)
            avg_entry = sum(e["price"] * e["size"] for e in pyramiding_entries) / total_sz if total_sz > 0 else entry_price
            if direction == "LONG":
                be_stop = avg_entry * (1 - 0.01 / 100.0)
            else:
                be_stop = avg_entry * (1 + 0.01 / 100.0)
            be_check = (direction == "LONG" and l <= be_stop) or \
                       (direction == "SHORT" and h >= be_stop)
            if be_check:
                pnl = _calc_pnl_from_entries(remaining, be_stop,
                                             current_entry, direction,
                                             params, is_sl=False)
                total_pnl += pnl
                fills.append({
                    "type": "BREAKEVEN",
                    "price": be_stop,
                    "usdt": round(pnl, 2),
                    "qty": round(remaining, 2),
                })
                exit_reason = "breakeven_stop"
                exit_price = be_stop
                exit_ts = ts
                bars_held += 1
                remaining = 0
                breakeven_hit = True

        sl_hit = (direction == "LONG" and l <= sl) or \
                 (direction == "SHORT" and h >= sl)

        # ── AVERAGING: микро-усреднение (только если нет SL в этой свече) ──
        if (params.avg_enabled
                and remaining > 0
                and avg_count < params.avg_max_increments
                and not sl_hit):

            avg_state = {
                "avg_count": avg_count,
                "current_entry": current_entry,
                "current_position": current_position,
                "remaining": remaining,
                "sl": sl,
                "sl_pct": sl_pct,
                "swing_extreme": entry_price,       # последний экстремум
                "last_avg_extreme": entry_price,     # экстремум на момент посл. avg
                "last_avg_delta_delta": 0.0,         # ΔΔ на момент посл. avg
                "cumulative_avg_risk": 0.0,          # накопленный риск avg в $
                "trade_entry_price": entry_price,    # первоначальный вход
            }
            _try_average(candles_after, ci, o, h, l, cls, params, direction, fills,
                         avg_state, global_state)
            avg_count = avg_state["avg_count"]
            current_entry = avg_state["current_entry"]
            current_position = avg_state["current_position"]
            remaining = avg_state["remaining"]
            sl = avg_state["sl"]
            sl_pct = avg_state["sl_pct"]

        # ── SL SECOND — только если осталась позиция ──
        if remaining > 0 and sl_hit:
            pnl = _calc_pnl_from_entries(remaining, sl,
                                         current_entry, direction, params, is_sl=True)
            total_pnl += pnl
            fills.append({"type": "SL", "price": sl,
                          "usdt": round(pnl, 2), "qty": round(remaining, 2)})
            exit_reason = "SL"
            exit_price = sl
            exit_ts = ts
            bars_held += 1
            remaining = 0
            break

        bars_held += 1

        # Таймаут
        if bars_held >= params.timeout_bars:
            pnl = _calc_pnl_from_entries(remaining, cls,
                                         current_entry, direction, params, is_sl=False)
            total_pnl += pnl
            fills.append({"type": "timeout", "price": cls,
                          "usdt": round(pnl, 2), "qty": round(remaining, 2)})
            exit_reason = "timeout"
            exit_price = cls
            exit_ts = ts
            remaining = 0
            break

    # Если осталась позиция в конце данных
    if remaining > 0 and candles_after:
        last_price = float(candles_after[-1][4])
        pnl = _calc_pnl_from_entries(remaining, last_price,
                                     current_entry, direction, params, is_sl=False)
        total_pnl += pnl
        fills.append({"type": "end_of_data", "price": last_price,
                      "usdt": round(pnl, 2), "qty": round(remaining, 2)})
        exit_reason = "end_of_data"
        exit_price = last_price
        exit_ts = candles_after[-1][0]

    # ── Критическое: проверка PnL после догрузки ──
    # Если сделка с догрузкой закрылась в минус — это баг
    pyramiding_pnl_bug = False
    if pyramiding_count > 0 and total_pnl < 0:
        pyramiding_pnl_bug = True

    return {
        "entry_ts": entry_ts,
        "entry_price": round(current_entry, 8),
        "exit_ts": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "direction": direction,
        "total_pnl_usdt": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / initial_position * 100, 2) if initial_position else 0,
        "sl_pct": round(sl_pct, 2),
        "position_usdt": round(initial_position, 2),
        "fills": fills,
        "bars_held": bars_held,
        "win": total_pnl > 0,
        "avg_count": avg_count,
        "pyramiding_count": pyramiding_count,
        "pyramiding_dd_used": pyramiding_dd_used,
        "pyramiding_tp_trailed": pyramiding_tp_trailed,
        "pyramiding_trailing_active": pyramiding_trailing_active,
        "pyramiding_pnl_bug": pyramiding_pnl_bug,
    }


def _calc_pnl_from_entries(qty: float, exit_price: float,
                            entry_price: float, direction: str,
                            params: BacktestParams, is_sl: bool) -> float:
    """Рассчитать PnL с учётом комиссии taker для market-ордеров."""
    if direction == "LONG":
        raw_pnl = qty * (exit_price - entry_price) / entry_price
    else:
        raw_pnl = qty * (entry_price - exit_price) / entry_price

    # Комиссия taker: TP, SL, timeout — всё market order
    abs_pnl = abs(raw_pnl)
    commission = abs_pnl * params.commission_taker
    return raw_pnl - commission


def _try_avg_zone(candles_after, ci, direction, entry_price, params):
    """
    Зонное усреднение: AVG только после проторговки + разворота.
    1. Цена прошла X% против нас (start_move_pct)
    2. Затем N свечей консолидации (low spread)
    3. Затем M свечей в нашу сторону (подтверждение)
    Возвращает цену для AVG или None, если условия не выполнены.
    """
    lookback = min(ci, 60)
    start_bars = max(ci - lookback, 0)

    if direction == "LONG":
        # Ищем минимум движения против нас
        low_idx = ci
        for i in range(ci - 1, start_bars - 1, -1):
            if float(candles_after[i][3]) < float(candles_after[low_idx][3]):
                low_idx = i

        move_against = (entry_price - float(candles_after[low_idx][3])) / entry_price * 100
        if move_against < params.avg_zone_start_move_pct:
            return None

        # Ищем зону проторговки после минимума
        for z_start in range(low_idx, ci - params.avg_zone_min_bars + 1):
            zone_end = z_start + params.avg_zone_min_bars
            if zone_end > ci:
                break
            zone_high = max(float(candles_after[j][2]) for j in range(z_start, zone_end))
            zone_low = min(float(candles_after[j][3]) for j in range(z_start, zone_end))
            zone_spread = (zone_high - zone_low) / zone_low * 100 if zone_low > 0 else 999
            if zone_spread <= params.avg_zone_max_spread_pct:
                # Зона найдена — проверяем свечи после зоны на разворот вверх
                conf_start = zone_end
                for rev_bar in range(conf_start, ci + 1):
                    bullish_count = 0
                    for ck in range(rev_bar, min(rev_bar + params.avg_zone_confirmation_bars, ci + 1)):
                        candle_close = float(candles_after[ck][4])
                        candle_open = float(candles_after[ck][1])
                        if candle_close > candle_open:
                            bullish_count += 1
                    if bullish_count >= params.avg_zone_confirmation_bars:
                        return float(candles_after[rev_bar][4])

    else:  # SHORT
        # Ищем максимум движения против нас
        high_idx = ci
        for i in range(ci - 1, start_bars - 1, -1):
            if float(candles_after[i][2]) > float(candles_after[high_idx][2]):
                high_idx = i

        move_against = (float(candles_after[high_idx][2]) - entry_price) / entry_price * 100
        if move_against < params.avg_zone_start_move_pct:
            return None

        # Ищем зону проторговки после максимума
        for z_start in range(high_idx, ci - params.avg_zone_min_bars + 1):
            zone_end = z_start + params.avg_zone_min_bars
            if zone_end > ci:
                break
            zone_high = max(float(candles_after[j][2]) for j in range(z_start, zone_end))
            zone_low = min(float(candles_after[j][3]) for j in range(z_start, zone_end))
            zone_spread = (zone_high - zone_low) / zone_low * 100 if zone_low > 0 else 999
            if zone_spread <= params.avg_zone_max_spread_pct:
                # Зона найдена — проверяем свечи после зоны на разворот вниз
                conf_start = zone_end
                for rev_bar in range(conf_start, ci + 1):
                    bearish_count = 0
                    for ck in range(rev_bar, min(rev_bar + params.avg_zone_confirmation_bars, ci + 1)):
                        candle_close = float(candles_after[ck][4])
                        candle_open = float(candles_after[ck][1])
                        if candle_close < candle_open:
                            bearish_count += 1
                    if bearish_count >= params.avg_zone_confirmation_bars:
                        return float(candles_after[rev_bar][4])

    return None


def _try_average(candles_after, ci, o, h, l, cls, params, direction, fills,
                 state: dict, global_state: dict) -> bool:
    """
    Микро-усреднение: небольшие добавки (avg_incremental_pct от тек. позиции)
    на каждом новом свинг-экстремуме с подтверждением ΔΔ.
    """
    entry_price = state["trade_entry_price"]
    current_entry = state["current_entry"]
    position = state["current_position"]
    avg_price = cls  # по умолчанию — текущее закрытие

    # ── ЗОННЫЙ РЕЖИМ: первое усреднение по проторговке + разворот ──
    if params.avg_zone_enabled and state["avg_count"] == 0:
        zone_price = _try_avg_zone(candles_after, ci, direction, entry_price, params)
        if zone_price is None:
            return False
        avg_price = zone_price  # AVG по цене разворота
    else:
        # ── СТАНДАРТНЫЙ РЕЖИМ: обновляем и проверяем свинг-экстремум ──
        if direction == "LONG":
            if l < state["swing_extreme"]:
                state["swing_extreme"] = l
            is_new_extreme = l <= state["last_avg_extreme"] * 0.9999
        else:
            if h > state["swing_extreme"]:
                state["swing_extreme"] = h
            is_new_extreme = h >= state["last_avg_extreme"] * 1.0001

        if not is_new_extreme:
            return False

        # Первое усреднение (станд.) — только после min_move_pct от entry
        if state["avg_count"] == 0:
            if direction == "LONG":
                move_against = (entry_price - l) / entry_price * 100
            else:
                move_against = (h - entry_price) / entry_price * 100
            if move_against < params.avg_min_move_pct:
                return False

    # 3. Кап накопленного риска усреднений
    max_avg_risk_usdt = params.deposit * params.avg_risk_per_level / 100.0
    if state["cumulative_avg_risk"] >= max_avg_risk_usdt:
        return False

    # 4. Проверка PnL других сделок
    if params.avg_only_if_profit_others and global_state:
        if global_state.get("total_open_pnl", 0) <= 0:
            return False

    # 5. ΔΔ подтверждение (на основе свечей самой монеты)
    if params.avg_use_delta_delta and global_state and not params.avg_zone_enabled:
        local_dd = _compute_local_dd(candles_after, ci)
        if direction == "LONG":
            dd_ok = local_dd > state["last_avg_delta_delta"] * 0.5
        else:
            dd_ok = local_dd < state["last_avg_delta_delta"] * 0.5
        if not dd_ok:
            return False

    # 6. Размер микро-добавки (% от текущей позиции)
    avg_size = max(position * params.avg_incremental_pct / 100.0, 1.0)

    if avg_size <= 0:
        return False

    # 7. Исполнение
    commission = avg_size * params.commission_taker
    old_value = current_entry * position
    new_value = avg_price * avg_size
    new_total = position + avg_size
    new_entry = (old_value + new_value) / new_total
    pnl_commission = -commission
    fills.append({
        "type": f"AVG{state['avg_count'] + 1}", "price": avg_price,
        "usdt": round(pnl_commission, 2),
        "qty": round(avg_size, 2),
        "commission": round(commission, 6),
        "old_entry": round(current_entry, 6),
        "new_entry": round(new_entry, 6),
    })

    # 8. Пересчёт SL
    initial_risk = params.deposit * params.max_deposit_risk_pct / 100.0
    total_avg_risk = state["cumulative_avg_risk"] + commission
    new_sl_pct = (initial_risk + total_avg_risk) / new_total * 100
    new_sl_pct = max(new_sl_pct, 0.3)
    new_sl_pct = min(new_sl_pct, params.max_sl_pct)

    if direction == "LONG":
        sl_price = new_entry * (1 - new_sl_pct / 100.0)
    else:
        sl_price = new_entry * (1 + new_sl_pct / 100.0)

    # 9. Обновление состояния
    state["avg_count"] += 1
    state["current_entry"] = new_entry
    state["current_position"] = new_total
    state["remaining"] += avg_size
    state["sl_pct"] = new_sl_pct
    state["sl"] = round(sl_price, 8)
    state["last_avg_extreme"] = l if direction == "LONG" else h
    state["last_avg_delta_delta"] = _compute_local_dd(candles_after, ci)
    state["cumulative_avg_risk"] = total_avg_risk
    return True


# ─── Пре-бэктест для подбора параметров ──────────────────────────────────

def optimize_params_for_asset(asset_symbol: str, base_params: BacktestParams,
                              candles_1h: list) -> dict:
    """
    Быстрый пре-бэктест по одному активу для подбора ключевых параметров.
    Перебирает комбинации trigger_pct, trailing_step_pct, size_pct.
    Возвращает лучшие параметры для данного актива.
    """
    from copy import deepcopy

    # Кандидаты для перебора
    trigger_opts = [1.5, 2.0, 3.0]
    step_opts = [0.5, 1.0, 2.0]
    size_opts = [0.10, 0.15, 0.20]

    best_result = None
    best_pnl = -float('inf')
    best_combo = {}

    total_combos = len(trigger_opts) * len(step_opts) * len(size_opts)
    tested = 0

    for trig in trigger_opts:
        for step in step_opts:
            for sz in size_opts:
                tested += 1
                p = deepcopy(base_params)
                p.pyramiding_trigger_pct = trig
                p.pyramiding_trailing_step_pct = step
                p.pyramiding_size_pct = sz
                p.pyramiding_enabled = True
                p.pyramiding_trailing_mode = "step"

                # Ограничиваем количество сделок для скорости (первые 100 сигналов)
                events = detect_trend(candles_1h, allow_ongoing=True)
                if not events:
                    continue

                total_pnl = 0.0
                trade_count = 0
                max_trades = 30  # лимит на пре-тест

                for ev in events[:100]:
                    if trade_count >= max_trades:
                        break
                    ts_ms = ev["timestamp"]
                    idx_h = ev["index"]
                    direction = "LONG" if ev["trend"] == "UP" else "SHORT"

                    # Минимальные фильтры
                    if detect_impulse(candles_1h, idx_h, p):
                        continue

                    prev_candle = {
                        "high": float(candles_1h[idx_h - 1][2]) if idx_h > 0 else ev["price"] * 1.02,
                        "low": float(candles_1h[idx_h - 1][3]) if idx_h > 0 else ev["price"] * 0.98,
                    }
                    signal = {
                        "direction": direction,
                        "timestamp": ts_ms,
                        "price": ev["price"],
                        "prev_candle": prev_candle,
                        "symbol": asset_symbol,
                        "index": idx_h,
                    }
                    trade = simulate_trade(signal, candles_1h, p, None)
                    if trade:
                        total_pnl += trade["total_pnl_usdt"]
                        trade_count += 1

                if trade_count > 0 and total_pnl > best_pnl:
                    best_pnl = total_pnl
                    best_combo = {
                        "trigger_pct": trig,
                        "trailing_step_pct": step,
                        "size_pct": sz,
                        "total_pnl": round(total_pnl, 2),
                        "trade_count": trade_count,
                    }

    return best_combo if best_combo else {}


# ─── ΔΔ ───────────────────────────────────────────────────────────────────

def compute_delta_delta(btc_1h_candles):
    """ΔΔ = разница net_delta между соседними 1h свечами."""
    results = []
    deltas = []
    for c in btc_1h_candles:
        quote_vol = float(c[7])
        taker_buy = float(c[10])
        net_delta = 2 * taker_buy - quote_vol
        deltas.append(net_delta)
    for i in range(1, len(deltas)):
        results.append({
            "timestamp": btc_1h_candles[i][0],
            "net_delta": deltas[i],
            "delta_delta": deltas[i] - deltas[i-1],
            "prev_delta": deltas[i-1],
        })
    return results


def _compute_local_dd(candles_1h: list, idx: int, lookback: int = 3) -> float:
    """
    Вычислить ΔΔ для отдельной монеты на idx позиции.
    Возвращает delta_delta (net_delta[i] - net_delta[i-1]).
    """
    if idx < 1 or idx >= len(candles_1h):
        return 0.0
    try:
        c_prev = candles_1h[idx - 1]
        c_cur = candles_1h[idx]
        qv_prev = float(c_prev[7])
        tb_prev = float(c_prev[10])
        nd_prev = 2 * tb_prev - qv_prev
        qv_cur = float(c_cur[7])
        tb_cur = float(c_cur[10])
        nd_cur = 2 * tb_cur - qv_cur
        return nd_cur - nd_prev
    except (IndexError, ValueError, TypeError):
        return 0.0


# ─── Phase Subtype Detection ───────────────────────────────────────────────

def detect_phase_subtype(candles_1h: list, end_idx: int, phase: str) -> str:
    """
    Определить подтип фазы на основе волатильности.

    Args:
        candles_1h: 1h свечи
        end_idx: текущий индекс
        phase: "Markup" | "Markdown" | "Distribution" | "Accumulation"

    Returns:
        str: подтип фазы
    """
    lookback = 20
    if end_idx < lookback:
        return "insufficient_data"

    relevant = candles_1h[end_idx - lookback : end_idx]
    if len(relevant) < lookback:
        return "insufficient_data"

    atr = _compute_atr(candles_1h, end_idx, period=14)
    if atr is None or atr <= 0:
        return "unknown"

    # Средний ATR% за lookback
    atr_percents = []
    for c in relevant:
        atr_pct = atr / float(c[4]) * 100
        atr_percents.append(atr_pct)
    median_atr_pct = sorted(atr_percents)[len(atr_percents) // 2]

    # Средний размер тела свечи как % от ATR
    body_ratios = []
    for c in relevant:
        o, cls = float(c[1]), float(c[4])
        body = abs(cls - o)
        body_ratio = body / atr if atr > 0 else 0
        body_ratios.append(body_ratio)
    avg_body_ratio = sum(body_ratios) / max(1, len(body_ratios))

    if phase == "Markup":
        if avg_body_ratio > 2.0:
            return "impulse_pump"
        elif avg_body_ratio < 1.0:
            return "gradual_trend"
        else:
            return "normal_markup"
    elif phase == "Markdown":
        if avg_body_ratio > 2.0:
            return "crash"
        elif avg_body_ratio < 1.0:
            return "gradual_decline"
        else:
            return "normal_markdown"
    elif phase == "Distribution":
        current_atr_pct = atr_percents[-1] if atr_percents else 0
        if current_atr_pct > median_atr_pct * 1.5:
            return "high_volatility"
        elif current_atr_pct < median_atr_pct * 0.5:
            return "low_volatility"
        else:
            return "normal_distribution"
    elif phase == "Accumulation":
        return "accumulation_phase"
    return "unknown"


# ─── BacktestRunner v3 ────────────────────────────────────────────────────

class BacktestRunner:
    """Основной класс бэктеста v3."""

    def __init__(self, params=None, assets=None):
        self.params = params or DEFAULT_PARAMS_V3
        self.assets = assets
        self.btc_1h = None
        self.btc_4h = None
        self.btc_1d = None
        self.btc_15m = None
        self.btc_1h_dict = []
        self.btc_4h_dict = []
        self.btc_1d_dict = []
        self.btc_delta_deltas = []
        self.asset_pool = {}

        self.results = {
            "trades": [],
            "btc_phases": [],
            "filter_stats": defaultdict(int),
            "asset_stats": defaultdict(
                lambda: {"signals": 0, "trades": 0, "wins": 0, "losses": 0, "pnl": 0}
            ),
        }

    def load_btc_data(self, force=False):
        """Загрузить данные BTC для всех ТФ."""
        print("\n[Backtest v3] Loading BTC data...", flush=True)
        self.btc_1h = download_symbol_range(BTC_SYMBOL, "1h", START_TS, END_TS, force)
        self.btc_4h = download_symbol_range(BTC_SYMBOL, "4h", START_TS, END_TS, force)
        self.btc_1d = download_symbol_range(BTC_SYMBOL, "1d", START_TS, END_TS, force)
        self.btc_15m = download_symbol_range(BTC_SYMBOL, "15m", START_TS, END_TS, force)

        self.btc_1h_dict = self._to_dict(self.btc_1h)
        self.btc_4h_dict = self._to_dict(self.btc_4h)
        self.btc_1d_dict = self._to_dict(self.btc_1d)
        self.btc_delta_deltas = compute_delta_delta(self.btc_1h)

        print(f"  BTC: {len(self.btc_1h)} 1h + {len(self.btc_4h)} 4h + "
              f"{len(self.btc_1d)} 1d candles, {len(self.btc_delta_deltas)} ΔΔ",
              flush=True)

    @staticmethod
    def _to_dict(raw):
        return [{"timestamp": c[0], "open": float(c[1]), "high": float(c[2]),
                  "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
                for c in raw]

    def get_btc_phase_at(self, ts_ms):
        """Фаза BTC на момент ts_ms + подтип фазы."""
        c1h = [c for c in self.btc_1h_dict if c["timestamp"] <= ts_ms]
        c4h = [c for c in self.btc_4h_dict if c["timestamp"] <= ts_ms]
        c1d = [c for c in self.btc_1d_dict if c["timestamp"] <= ts_ms]
        if len(c1h) < 50:
            return None
        phase = mp.detect_market_phase(c1h[-100:], c4h[-100:], c1d[-100:])
        if phase:
            # Добавляем подтип фазы
            end_idx = len(c1h) - 1
            subtype = detect_phase_subtype(self.btc_1h, min(end_idx, len(self.btc_1h) - 1),
                                           phase.get("phase", "unknown"))
            phase["subtype"] = subtype
        return phase

    def get_delta_delta_at(self, ts_ms):
        recent = [d for d in self.btc_delta_deltas if d["timestamp"] < ts_ms]
        return recent[-1] if recent else None

    def get_rsi_at(self, ts_ms):
        c1h = [c for c in self.btc_1h_dict if c["timestamp"] <= ts_ms]
        if len(c1h) < 15:
            return None
        closes = [c["close"] for c in c1h]
        return mp._rsi(closes, 14)

    def get_24h_volume(self, symbol, ts_ms):
        candles = self.asset_pool.get(symbol, [])
        recent = [c for c in candles
                  if c[0] <= ts_ms and c[0] > ts_ms - 24 * 3600 * 1000]
        if len(recent) < 6:
            all_candles = [c for c in candles if c[0] <= ts_ms]
            if len(all_candles) < 6:
                return 0
            avg_per_candle = sum(float(c[7]) for c in all_candles[-50:]) / min(50, len(all_candles))
            return avg_per_candle * 24
        return sum(float(c[7]) for c in recent[-24:])

    def _quick_asset_backtest(self, candles_1h, symbol):
        """Быстрый бэктест актива: запускаем trend detection и симуляцию.
        Возвращает True, если актив прошёл фильтр."""
        params = self.params
        events = detect_trend(candles_1h, allow_ongoing=True)
        if not events:
            return False

        total_pnl = 0.0
        trades = 0
        wins = 0
        for ev in events[:50]:  # макс 50 сделок для скорости
            idx_h = ev["index"]
            ts_ms = ev["timestamp"]
            direction = "LONG" if ev["trend"] == "UP" else "SHORT"

            # BTC phase check
            btc_phase_info = self.get_btc_phase_at(ts_ms)
            phase = btc_phase_info
            if phase and params.use_btc_phase:
                if params.skip_accumulation and phase.get("phase") == "Accumulation":
                    continue
                if direction == "LONG" and not phase.get("can_long", True):
                    continue
                if direction == "SHORT" and not phase.get("can_short", True):
                    continue

            prev_candle = {
                "high": float(candles_1h[idx_h - 1][2]) if idx_h > 0 else ev["price"] * 1.02,
                "low": float(candles_1h[idx_h - 1][3]) if idx_h > 0 else ev["price"] * 0.98,
            }

            signal = {
                "direction": direction,
                "timestamp": ts_ms,
                "price": ev["price"],
                "prev_candle": prev_candle,
                "symbol": symbol,
                "index": idx_h,
            }

            trade = simulate_trade(signal, candles_1h, params, None)
            if trade:
                total_pnl += trade["total_pnl_usdt"]
                trades += 1
                if trade["win"]:
                    wins += 1

        if trades == 0:
            return False

        winrate = wins / trades * 100
        return (total_pnl >= params.pre_trade_min_pnl and
                winrate >= params.pre_trade_min_winrate)

    def run(self, max_assets=None, progress_cb=None):
        """
        Основной прогон v3.
        Включает глобальный риск-менеджмент (total_open_pnl),
        пропуск сигналов в Accumulation, проверку просадки.
        """
        if self.btc_1h is None:
            self.load_btc_data()

        # Активы
        assets = self.assets
        if assets is None:
            print("\n[Backtest v3] Fetching top USDT pairs...", flush=True)
            assets = get_top_usdt_pairs(limit=607, min_volume=0)

        if max_assets:
            assets = assets[:max_assets]

        print(f"\n[Backtest v3] Testing {len(assets)} assets | "
              f"sl_atr={self.params.sl_atr_mult}, "
              f"tp_count={self.params.tp_count}, "
              f"avg={self.params.avg_enabled}, "
              f"pyramiding={self.params.pyramiding_enabled}, "
              f"risk={self.params.max_deposit_risk_pct}%", flush=True)

        # BTC phase timeline
        if not self.results["btc_phases"]:
            for ts in range(START_TS * 1000, END_TS * 1000, 6 * 3600 * 1000):
                ph = self.get_btc_phase_at(ts)
                if ph:
                    self.results["btc_phases"].append({
                        "timestamp": ts,
                        "phase": ph.get("phase"),
                        "subtype": ph.get("subtype"),
                        "direction": ph.get("direction"),
                        "rsi_1h": ph.get("rsi", {}).get("1h"),
                        "rsi_4h": ph.get("rsi", {}).get("4h"),
                    })

        total_signals = 0
        total_trades = 0

        # ── Глобальный риск-менеджмент ──
        global_state = {
            "total_open_pnl": 0.0,
            "peak_equity": self.params.deposit,
            "current_equity": self.params.deposit,
            "get_delta_delta_at": self.get_delta_delta_at,
        }

        for idx, symbol in enumerate(assets):
            if progress_cb and (idx + 1) % 10 == 0:
                progress_cb(total_trades)

            # Проверка хард-капа просадки
            drawdown = (global_state["peak_equity"] - global_state["current_equity"]) / global_state["peak_equity"] * 100
            if drawdown >= self.params.max_drawdown_pct:
                print(f"\n⚠️  HARD STOP: Drawdown {drawdown:.1f}% >= {self.params.max_drawdown_pct}% "
                      f"at {symbol} (asset #{idx})", flush=True)
                self.results["filter_stats"]["max_drawdown_hit"] = 1
                break

            # Глобальный стоп-лосс
            if global_state["total_open_pnl"] <= -self.params.deposit * self.params.global_stop_loss_pct / 100:
                print(f"\n⚠️  GLOBAL STOP: total_open_pnl=${global_state['total_open_pnl']:.2f} "
                      f"<= -${self.params.deposit * self.params.global_stop_loss_pct / 100:.2f} "
                      f"at {symbol}", flush=True)
                self.results["filter_stats"]["global_stop_loss_hit"] = 1
                break

            # Загрузка данных актива
            candles_1h = load_cached(symbol, "1h")
            if candles_1h is None:
                try:
                    candles_1h = download_symbol_range(symbol, "1h", START_TS, END_TS)
                except Exception:
                    continue

            self.asset_pool = {symbol: candles_1h}

            # Средний 24h объём
            all_volumes = [float(c[7]) for c in candles_1h[:500]]
            if not all_volumes:
                continue
            avg_hourly = sum(all_volumes) / len(all_volumes)
            vol_24h = avg_hourly * 24
            if vol_24h < self.params.min_volume_24h:
                self.results["filter_stats"]["volume_blocked"] += 1
                continue

            # ── Пре-трейд бэктест-фильтр: проверяем качество актива ──
            if self.params.pre_trade_backtest_filter:
                bt_ok = self._quick_asset_backtest(candles_1h, symbol)
                if not bt_ok:
                    self.results["filter_stats"]["pre_trade_backtest_blocked"] += 1
                    continue

            # ── Пре-оптимизация параметров под актив ──
            if self.params.enable_per_asset_optimization:
                from copy import deepcopy
                opt_result = optimize_params_for_asset(symbol,
                                                       self.params, candles_1h)
                if opt_result and opt_result.get("total_pnl", 0) > 0:
                    # Временно применяем оптимизированные параметры
                    self.params.pyramiding_trigger_pct = opt_result["trigger_pct"]
                    self.params.pyramiding_trailing_step_pct = opt_result["trailing_step_pct"]
                    self.params.pyramiding_size_pct = opt_result["size_pct"]
                    self.params.pyramiding_enabled = True
                    self.results["filter_stats"]["per_asset_optimized"] = \
                        self.results["filter_stats"].get("per_asset_optimized", 0) + 1

            # Тренды
            events = detect_trend(candles_1h, allow_ongoing=True)
            if not events:
                continue

            for ev in events:
                total_signals += 1
                self.results["filter_stats"]["total_signals"] += 1

                ts_ms = ev["timestamp"]
                idx_h = ev["index"]
                direction = "LONG" if ev["trend"] == "UP" else "SHORT"

                # ── FILTER 1: BTC Phase ──
                btc_phase_info = self.get_btc_phase_at(ts_ms)
                phase = btc_phase_info
                if phase and self.params.use_btc_phase:
                    # Accumulation — пропуск
                    if self.params.skip_accumulation and phase.get("phase") == "Accumulation":
                        self.results["filter_stats"]["accumulation_blocked"] += 1
                        continue

                    if direction == "LONG" and not phase.get("can_long", True):
                        self.results["filter_stats"]["phase_blocked"] += 1
                        continue

                    if direction == "SHORT" and not phase.get("can_short", True):
                        self.results["filter_stats"]["phase_blocked"] += 1
                        continue

                # ── FILTER 2: RSI gate ──
                rsi_1h = self.get_rsi_at(ts_ms)
                if rsi_1h is not None:
                    if direction == "LONG" and rsi_1h < self.params.rsi_long_min:
                        self.results["filter_stats"]["rsi_blocked"] += 1
                        continue
                    if direction == "SHORT" and rsi_1h > self.params.rsi_short_max:
                        self.results["filter_stats"]["rsi_blocked"] += 1
                        continue

                # ── FILTER 3: ΔΔ (BTC) ──
                if self.params.use_dd_filter:
                    dd_info = self.get_delta_delta_at(ts_ms)
                    if dd_info:
                        dd_val = dd_info["delta_delta"]
                        thr = self.params.dd_threshold
                        if direction == "LONG" and dd_val < -thr:
                            self.results["filter_stats"]["dd_blocked"] += 1
                            continue
                        if direction == "SHORT" and dd_val > thr:
                            self.results["filter_stats"]["dd_blocked"] += 1
                            continue

                # ── FILTER 3.5: ΔΔ самого актива ──
                if self.params.use_asset_dd_filter:
                    asset_dd = _compute_local_dd(candles_1h, idx_h, 3)
                    thr_a = self.params.asset_dd_threshold
                    if direction == "LONG" and asset_dd < -thr_a:
                        self.results["filter_stats"]["asset_dd_blocked"] += 1
                        continue
                    if direction == "SHORT" and asset_dd > thr_a:
                        self.results["filter_stats"]["asset_dd_blocked"] += 1
                        continue

                # ── FILTER 4A: Анти-стопхантинг (аномальная волатильность) ──
                if self.params.anti_stophunting_filter:
                    tail_check_idx = idx_h - 1
                    lookback = min(self.params.anti_stophunting_lookback, tail_check_idx)
                    if lookback > 5:
                        anomalies = 0
                        for bi in range(tail_check_idx - lookback, tail_check_idx):
                            c = candles_1h[bi]
                            co, ch, cl, cc = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                            body = abs(cc - co)
                            tail = max(ch - max(co, cc), min(co, cc) - cl)
                            atr_val = _compute_atr(candles_1h, bi, 14)
                            if atr_val and atr_val > 0:
                                tail_ratio = tail / atr_val
                                if tail_ratio > self.params.anti_stophunting_tail_threshold:
                                    anomalies += 1
                        if anomalies > self.params.anti_stophunting_max_anomalies:
                            self.results["filter_stats"]["stophunt_blocked"] += 1
                            continue

                # ── FILTER 4B: ATR актива (аномальный всплеск) ──
                if self.params.asset_atr_filter:
                    atr_check_idx = idx_h - 1
                    lookback = min(self.params.anti_stophunting_lookback, atr_check_idx)
                    if lookback > 10:
                        atrs = []
                        for bi in range(atr_check_idx - lookback, atr_check_idx):
                            av = _compute_atr(candles_1h, bi, 14)
                            if av and av > 0:
                                atrs.append(av)
                        if len(atrs) >= 10:
                            sorted_atrs = sorted(atrs)
                            median_atr = sorted_atrs[len(sorted_atrs) // 2]
                            current_atr = atrs[-1]
                            if current_atr > median_atr * self.params.asset_atr_multiplier:
                                self.results["filter_stats"]["atr_blocked"] += 1
                                continue

                # ── FILTER 5: Impulse ──
                if detect_impulse(candles_1h, idx_h, self.params):
                    self.results["filter_stats"]["impulse_blocked"] += 1
                    continue

                # ── Все фильтры пройдены → сделка ──
                self.results["filter_stats"]["trades_opened"] += 1
                total_trades += 1

                prev_candle = {
                    "high": float(candles_1h[idx_h - 1][2]) if idx_h > 0 else ev["price"] * 1.02,
                    "low": float(candles_1h[idx_h - 1][3]) if idx_h > 0 else ev["price"] * 0.98,
                }

                signal_data = {
                    "direction": direction,
                    "timestamp": ts_ms,
                    "price": ev["price"],
                    "prev_candle": prev_candle,
                    "symbol": symbol,
                    "index": idx_h,
                    "btc_phase": phase.get("phase", "unknown") if phase else "unknown",
                    "btc_subtype": phase.get("subtype", "unknown") if phase else "unknown",
                }

                trade = simulate_trade(signal_data, candles_1h, self.params, global_state)
                if trade:
                    trade["symbol"] = symbol
                    trade["btc_phase"] = signal_data["btc_phase"]
                    trade["btc_subtype"] = signal_data["btc_subtype"]
                    trade["signal_ts"] = ts_ms
                    self.results["trades"].append(trade)

                    # Обновляем глобальный PnL
                    global_state["total_open_pnl"] += trade["total_pnl_usdt"]
                    global_state["current_equity"] = self.params.deposit + global_state["total_open_pnl"]
                    if global_state["current_equity"] > global_state["peak_equity"]:
                        global_state["peak_equity"] = global_state["current_equity"]

                    astat = self.results["asset_stats"][symbol]
                    astat["signals"] += 1
                    astat["trades"] += 1
                    astat["wins"] += 1 if trade["win"] else 0
                    astat["losses"] += 0 if trade["win"] else 1
                    astat["pnl"] += trade["total_pnl_usdt"]

        print(f"\n[Backtest v3] Done: {total_signals} signals → {total_trades} trades", flush=True)
        return self.results

    def summary(self, verbose=True):
        """Вывести сводку v3. Возвращает dict со статистикой."""
        trades = self.results["trades"]
        filters = dict(self.results["filter_stats"])

        if not trades:
            if verbose:
                print("\n❌ No trades.")
            return None

        total_pnl = sum(t["total_pnl_usdt"] for t in trades)
        wins = sum(1 for t in trades if t["win"])
        losses = sum(1 for t in trades if not t["win"])
        win_rate = wins / len(trades) * 100 if trades else 0

        # По фазам + подтипам
        phase_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0,
                                           "pnl_usdt": 0})
        subtype_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0,
                                              "pnl_usdt": 0})
        for t in trades:
            ps = phase_stats[t.get("btc_phase", "unknown")]
            ps["trades"] += 1
            ps["wins"] += 1 if t["win"] else 0
            ps["losses"] += 0 if t["win"] else 1
            ps["pnl_usdt"] += t["total_pnl_usdt"]

            st = subtype_stats[t.get("btc_subtype", "unknown")]
            st["trades"] += 1
            st["wins"] += 1 if t["win"] else 0
            st["losses"] += 0 if t["win"] else 1
            st["pnl_usdt"] += t["total_pnl_usdt"]

        # По exit_reason
        reasons = defaultdict(int)
        for t in trades:
            reasons[t["exit_reason"]] += 1

        # Статистика по усреднению и догрузке
        avg_stats = {"trades_with_avg": 0, "total_avgs": 0,
                     "trades_with_pyra": 0, "total_pyras": 0}
        for t in trades:
            if t.get("avg_count", 0) > 0:
                avg_stats["trades_with_avg"] += 1
                avg_stats["total_avgs"] += t["avg_count"]
            if t.get("pyramiding_count", 0) > 0:
                avg_stats["trades_with_pyra"] += 1
                avg_stats["total_pyras"] += t["pyramiding_count"]

        if verbose:
            self._print_summary(trades, filters, total_pnl, wins, losses,
                                win_rate, phase_stats, subtype_stats, reasons,
                                avg_stats)

        max_dd_pct = 0
        if trades:
            equity = self.params.deposit
            peak = equity
            for t in trades:
                equity += t["total_pnl_usdt"]
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd_pct:
                    max_dd_pct = dd

        return {
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "wins": wins,
            "losses": losses,
            "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
            "max_win": round(max(t["total_pnl_usdt"] for t in trades), 2) if trades else 0,
            "max_loss": round(min(t["total_pnl_usdt"] for t in trades), 2) if trades else 0,
            "max_drawdown_pct": round(max_dd_pct, 2),
            "exit_reasons": dict(reasons),
            "filters": {k: v for k, v in filters.items()},
            "phases": {k: dict(v) for k, v in phase_stats.items()},
            "subtypes": {k: dict(v) for k, v in subtype_stats.items()},
            "avg_stats": avg_stats,
            "global_stop_hit": filters.get("global_stop_loss_hit", 0),
            "max_drawdown_hit": filters.get("max_drawdown_hit", 0),
        }

    def _print_summary(self, trades, filters, total_pnl, wins, losses,
                        win_rate, phase_stats, subtype_stats, reasons,
                        avg_stats):
        print(f"""
╔══════════════════════════════════════════╗
║         BACKTEST RESULTS v3             ║
║         Risk-based SL/TP + Commissions  ║
╠══════════════════════════════════════════╣
  Assets:  {len(set(t["symbol"] for t in trades))}
  Signals: {filters.get('total_signals', 0):>6}
  Trades:  {len(trades):>6}
  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)
  Total PnL: ${total_pnl:+.2f}
  Avg/Trade: ${total_pnl/len(trades):+.2f}
  Max Win: ${max(t['total_pnl_usdt'] for t in trades):+.2f}
  Max Loss: ${min(t['total_pnl_usdt'] for t in trades):+.2f}
╠══════════════════════════════════════════╣
  Filters:""")
        for k, v in sorted(filters.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
        print(f"""
╠══════════════════════════════════════════╣
  Avg / Pyramiding:
    With AVG: {avg_stats['trades_with_avg']}T ({avg_stats['total_avgs']} entries)
    With pyramiding: {avg_stats['trades_with_pyra']}T ({avg_stats['total_pyras']} entries)
╠══════════════════════════════════════════╣
  Exit Reasons:""")
        for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = n / len(trades) * 100
            print(f"    {r}: {n} ({pct:.0f}%)")

        print("\n  By BTC Phase:")
        for phase, ps in sorted(phase_stats.items(), key=lambda x: -x[1]["trades"]):
            wr = ps["wins"] / ps["trades"] * 100 if ps["trades"] else 0
            avg = ps["pnl_usdt"] / ps["trades"] if ps["trades"] else 0
            print(f"    {phase}: {ps['trades']}T {wr:.0f}%WR ${ps['pnl_usdt']:+.0f}PnL ${avg:+.2f}avg")

        print("\n  By Phase Subtype:")
        for st, ps in sorted(subtype_stats.items(), key=lambda x: -x[1]["trades"]):
            wr = ps["wins"] / ps["trades"] * 100 if ps["trades"] else 0
            avg = ps["pnl_usdt"] / ps["trades"] if ps["trades"] else 0
            print(f"    {st}: {ps['trades']}T {wr:.0f}%WR ${ps['pnl_usdt']:+.0f}PnL ${avg:+.2f}avg")

        sorted_assets = sorted(self.results["asset_stats"].items(), key=lambda x: x[1]["pnl"])
        if sorted_assets:
            print("\n  Best 3:")
            for sym, s in sorted_assets[-3:]:
                wr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
                print(f"    +{sym}: {s['trades']}T {wr:.0f}%WR ${s['pnl']:+.0f}")
            print("  Worst 3:")
            for sym, s in sorted_assets[:3]:
                wr = s["wins"] / s["trades"] * 100 if s["trades"] else 0
                print(f"    -{sym}: {s['trades']}T {wr:.0f}%WR ${s['pnl']:+.0f}")
        print("╚══════════════════════════════════════════╝")

    def save_results(self, path=None):
        """Сохранить результаты в JSON."""
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(__file__).parent / "reports" / f"backtest_v3_{ts}.json"
        path.parent.mkdir(exist_ok=True)

        summary = self.summary(verbose=False)
        data = {
            "params": asdict(self.params),
            "summary": summary,
            "trades_count": len(self.results["trades"]),
            "timestamp": datetime.now().isoformat(),
            "engine_version": "v4",
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n💾 Saved: {path}")
        return path


# ─── Self-test / Test Runner ─────────────────────────────────────────────

def run_test(params_overrides: dict, max_assets: int, report_name: str):
    """Запустить тест с заданными параметрами и сохранить результат."""
    import time

    params = BacktestParams()
    for k, v in params_overrides.items():
        if hasattr(params, k):
            setattr(params, k, v)

    print(f"\n{'='*60}")
    print(f"  TEST: {report_name}")
    print(f"{'='*60}")
    print(f"  Params: {json.dumps(params_overrides, indent=2)}")
    print(f"  max_assets={max_assets}, deposit=${params.deposit}")

    # Загружаем кешированные активы
    import glob
    files = glob.glob(str(Path(__file__).parent / "data" / "*_1h.pkl"))
    cached = sorted(set(f.replace('_1h.pkl','').split('/')[-1] for f in files))
    selected = cached[:max_assets]

    t0 = time.time()
    runner = BacktestRunner(params=params, assets=selected)
    runner.load_btc_data()
    runner.run(max_assets=len(selected))
    s = runner.summary(verbose=True)
    elapsed = time.time() - t0

    print(f"\n⏱️  Elapsed: {elapsed/60:.1f} min")

    # Сохраняем
    report_path = Path(__file__).parent / "reports" / f"{report_name}.json"
    report_path.parent.mkdir(exist_ok=True)
    data = {
        "test_name": report_name,
        "params": asdict(params),
        "summary": s,
        "trades_count": len(runner.results["trades"]),
        "elapsed_sec": round(elapsed, 1),
        "engine_version": "v4",
    }
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Saved: {report_path}")

    return s


def run_test_with_weekly(params_overrides: dict, max_assets: int, report_name: str,
                         initial_deposit: float = 100.0):
    """
    Запустить тест + трек понедельного PnL для DCA-анализа.
    Каждую неделю (по понедельникам UTC) фиксирует equity.
    """
    import time, json
    from datetime import datetime, timezone, timedelta
    from pathlib import Path
    import glob

    params = BacktestParams()
    for k, v in params_overrides.items():
        if hasattr(params, k):
            setattr(params, k, v)
    params.deposit = initial_deposit

    print(f"\n{'='*60}")
    print(f"  TEST: {report_name} (weekly tracking)")
    print(f"  Initial deposit: ${initial_deposit}")
    print(f"{'='*60}")

    files = glob.glob(str(Path(__file__).parent / "data" / "*_1h.pkl"))
    cached = sorted(set(f.replace('_1h.pkl','').split('/')[-1] for f in files))
    selected = cached[:max_assets]

    t0 = time.time()
    runner = BacktestRunner(params=params, assets=selected)
    runner.load_btc_data()
    runner.run(max_assets=len(selected))

    trades = runner.results["trades"]
    if not trades:
        print("❌ No trades.")
        return None

    # Сортируем сделки по времени
    trades_sorted = sorted(trades, key=lambda t: t["signal_ts"])

    # Определяем недельные границы
    start_ts = trades_sorted[0]["signal_ts"]
    end_ts = trades_sorted[-1]["signal_ts"]

    start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc)

    # Находим первый понедельник
    current = start_dt
    while current.weekday() != 0:  # Monday
        current += timedelta(days=1)
    week_start = current.replace(hour=0, minute=0, second=0, microsecond=0)

    # Трекинг
    equity = initial_deposit
    peak = equity
    max_dd_pct = 0
    trade_idx = 0
    total_trades = len(trades_sorted)

    print(f"\n{'='*90}")
    print(f"  ПОНЕДЕЛЬНЫЙ РОСТ ЭКВИТИ (начало: ${initial_deposit:.0f})")
    print(f"  Период: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")
    print(f"  Всего сделок: {total_trades}")
    print(f"{'='*90}")
    print(f"{'Неделя':<22} {'Дата':<14} {'Сделок':>8} {'Equity':>12} {'PnL нед':>10} {'PnL всего':>12} {'ДД':>8}")
    print(f"{'-'*90}")

    weekly_rows = []
    week_num = 0
    week_pnl = 0.0
    week_trades = 0
    start_equity = initial_deposit

    while week_start <= end_dt:
        week_end = week_start + timedelta(days=7)
        week_start_ms = int(week_start.timestamp() * 1000)
        week_end_ms = int(week_end.timestamp() * 1000)

        # Собираем сделки за эту неделю
        week_trades = 0
        week_pnl = 0.0
        while trade_idx < total_trades and trades_sorted[trade_idx]["signal_ts"] < week_end_ms:
            t = trades_sorted[trade_idx]
            equity += t["total_pnl_usdt"]
            week_pnl += t["total_pnl_usdt"]
            week_trades += 1
            trade_idx += 1

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

        weekly_rows.append({
            "week": week_num,
            "date": week_start.strftime("%Y-%m-%d"),
            "trades": week_trades,
            "equity": round(equity, 2),
            "week_pnl": round(week_pnl, 2),
            "total_pnl": round(equity - initial_deposit, 2),
            "dd": round(dd, 2),
        })

        print(f"W{week_num:<5} {week_start.strftime('%Y-%m-%d'):<14} {week_trades:>8} ${equity:>8,.2f} {week_pnl:>+9,.2f} ${equity-initial_deposit:>8,.2f} {dd:>6.1f}%")

        week_num += 1
        week_start = week_end

    elapsed = time.time() - t0
    print(f"{'='*90}")
    print(f"  ИТОГО: ${equity:,.2f} | PnL: ${equity-initial_deposit:,.2f} | Max ДД: {max_dd_pct:.1f}% | Сделок: {total_trades}")
    print(f"  Время: {elapsed/60:.1f} мин | Ростов: {((equity/initial_deposit)-1)*100:.1f}%")
    print(f"{'='*90}")

    # Сохраняем
    report_path = Path(__file__).parent / "reports" / f"{report_name}.json"
    report_path.parent.mkdir(exist_ok=True)
    s = runner.summary(verbose=True)
    data = {
        "test_name": report_name,
        "params": {k: getattr(params, k) for k in dir(params) if not k.startswith('_') and not callable(getattr(params, k))},
        "summary": s,
        "weekly": weekly_rows,
        "trades_count": total_trades,
        "elapsed_sec": round(elapsed, 1),
        "engine_version": "v4",
    }
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"💾 Weekly data saved: {report_path}")

    return s


if __name__ == "__main__":
    # ── Запуск тестов ──
    print("""
╔══════════════════════════════════════════╗
║   ENGINE v4 — Self-Test Suite           ║
║   Trailing pyramiding + asset ΔΔ + opt  ║
╚══════════════════════════════════════════╝
    """)

    # Test 1: базовые правила БЕЗ догрузки (проверка совместимости с v3)
    test1_params = {
        "sl_atr_mult": 3.0,
        "tp_count": 5,
        "tp_first_rr": 0.8,
        "impulse_enabled": False,
        "avg_enabled": False,
        "pyramiding_enabled": False,
        "skip_accumulation": True,
        "deposit": 10000,
        "max_deposit_risk_pct": 1.5,
        "global_stop_loss_pct": 2.4,
        "max_drawdown_pct": 3.0,
        "commission_maker": 0.0002,
        "commission_taker": 0.0004,
    }
    s1 = run_test(test1_params, max_assets=10, report_name="engine_v4_test1")

    # Test 2: новая trailing догрузка
    test2_params = dict(test1_params)
    test2_params.update({
        "avg_enabled": False,
        "pyramiding_enabled": True,
        "pyramiding_trigger_pct": 2.0,
        "pyramiding_size_pct": 0.15,
        "pyramiding_max_count": 5,
        "pyramiding_trailing_mode": "step",
        "pyramiding_trailing_step_pct": 1.0,
        "pyramiding_trailing_alloc_pct": 0.25,
        "pyramiding_trailing_buffer_steps": 1,
        "pyramiding_global_breakeven": True,
        "use_asset_dd_filter": True,
        "asset_dd_threshold": 500000,
    })
    s2 = run_test(test2_params, max_assets=10, report_name="engine_v4_test2_trailing")
