# ETH Perp Market Making — бэктест

Маркет-мейкер на ETH perpetual и событийный бэктестер на L2-снимках стакана. Данные: 3 дня, 2026-03-19 … 21.
Задание: [DESCRIPTION.md](DESCRIPTION.md). Отчёт: [analytics/report.html](analytics/report.html).

**Итог.** ML fair value + inventory skew, half-spread 0.45 USD, лимит позиции 2 ETH. На двух днях вне выборки
результат **+79 USD** (+2.4 bp от оборота). 95% интервал **−154 … +294 USD**, t = 0.61. Допущения оптимистичные:
colocation (задержка 1 / 2 / 2 мс), комиссии maker 0 / taker 1.7 bp. Стратегия около breakeven. Данных мало для точного вывода.

## 1. Установка

Нужен Python 3.11 или новее.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q          # 93 теста на синтетических данных, ~3 с; данные не нужны
```

## 2. Данные

Данные задания не входят в репозиторий. Положите parquet-файлы так:

```
data/
  orderbook/2026-03-19.parquet  2026-03-20.parquet  2026-03-21.parquet
  trades/   2026-03-19.parquet  2026-03-20.parquet  2026-03-21.parquet
  fundings/ 2026-03-19.parquet  2026-03-20.parquet  2026-03-21.parquet
```

| Папка | Колонки |
|---|---|
| `orderbook/` | `datetime` (нс), `bid_price_i`, `bid_qty_i`, `ask_price_i`, `ask_qty_i`, i = 1…20 |
| `trades/` | `datetime`, `price`, `size`, `is_maker_ask` (1 — агрессор покупатель) |
| `fundings/` | `datetime`, `funding_rate` (часовая ставка, ~каждые 20 с) |

## 3. Воспроизведение отчёта

Одна команда:

```bash
python reproduce.py               # ~11 мин на 12 ядрах; 6 бэктестов параллельно
python reproduce.py --jobs 2      # если мало памяти (~1.5 ГБ на бэктест)
python reproduce.py --skip-train  # модели в models/ уже есть
```

Скрипт выполняет 4 шага:

| Шаг | Команда | Результат |
|---|---|---|
| 1. ML-сигнал 1 с | `train_fair_value.py` (обучение на 03-19) | `models/fair_value_signal.parquet` |
| 2. Бэктесты | `run_backtest.py` × 32 прогона | `output/<папка>/<run>_{metrics.json,fills.parquet,pnl.png}` |
| 3. ML-сигналы 30 с и 60 мин | `train_horizon_models.py`, затем 1 бэктест | `models/fv_h30000.parquet`, `output/fv_h30s/` |
| 4. Отчёт | `analytics/build_charts.py` | `analytics/charts/*.png`, `analytics/data/summary.json` |

Шаг 3 идёт после шага 2: отчёт моделей использует fills из `output/fv_wide`.
Точные параметры каждого прогона — в функции `backtests()` в [reproduce.py](reproduce.py).

Прогоны шага 2:

| Папка | Что внутри | Где в отчёте |
|---|---|---|
| `fv_wide` | 03-20…21, лимит 2 ETH: ML fair value (skew 0.05, 0.1) и mid, half-spread 0.15…0.65 | Основной результат, фазы PnL, markout |
| `fv_h30s` | То же, ML-сигнал на 30 с | ML по фазам |
| `latency_grid`, `latency_hl` | ML и mid 0.45 при задержке 1/10/10, 1/50/10, 100/150/50, 180/450/450 мс | Чувствительность к задержке |
| `spread_sweep`, `spread_sweep_lowinv` | mid, 3 дня, half-spread 0.05…0.95, лимит 10 и 2 ETH | Контекст по глубине, round trips |

Все параметры прогона сохранены в его `metrics.json` (`config`). Модели обучены с `random_state=0`.

**Проверка.** Сравните `analytics/data/summary.json` с этими значениями. Прогон с нуля в чистой копии репозитория дал тот же `summary.json` до 1e-6:

| Ключ | Значение |
|---|---|
| `runs["ML fair value 0.45"].pnl` | +79.4 USD |
| `runs["mid 0.45"].pnl` | −11.2 USD |
| `pnl_interval_95["ML fair value 0.45"]` | [70, −154, +294] |
| `waterfall.bp` | премия +2.46, влияние −6.33, возврат +5.51 |
| `ml_30s_backtest_pnl` | −46.9 USD |
| `fee_scenarios`, `latency` | таблицы раздела 5 отчёта |

**Комиссии в отчёте.** Комиссия не меняет fills (правило cross = `confirmed`). Поэтому `build_charts.py`
пересчитывает комиссию каждого fill по пресету `cex_pro`. Остальные пресеты есть в `fee_scenarios`.

## 4. Один бэктест

Все настройки — в [configs/default.toml](configs/default.toml). Любой ключ можно изменить флагом `--set`.

```bash
python run_backtest.py                                              # настройки по умолчанию
python run_backtest.py --set strategy.half_spread=0.45 --set strategy.max_position=2
python run_backtest.py --sweep strategy.half_spread=0.25,0.45 --sweep fills.cancel_rule=back,front   # сетка 2 × 2
python run_backtest.py --set data.days=2026-03-20,2026-03-21 --set strategy.name=fair_value \
    --set strategy.signal_path=models/fair_value_signal.parquet --set strategy.inventory_skew=0.05
```

| Секция | Главные ключи |
|---|---|
| `strategy` | `name` (`mid`, `fair_value`), `half_spread`, `lot`, `max_position`, `signal_path`, `inventory_skew` |
| `latency` | `market_data_ms`, `order_ms`, `ack_ms` |
| `fees` | `venue` (`cex_pro`, `cex_mm_rebate`, `hyperliquid`, `bybit`), `maker_bps`, `taker_bps` |
| `fills` | `model` (`queue`, `simple`), `cross_rule`, `cancel_rule` |
| `data` | `days`, `levels` |

## 5. Структура

```
mm/core/        типы: стакан, сделка, ордер, fill
mm/data/        чтение parquet, единый поток событий
mm/strategy/    MidQuoter, FairValueQuoter (ML-сигнал + skew)
mm/signals/     фичи и target для ML
mm/trading/     бот и менеджер ордеров (видит рынок с задержкой)
mm/backtester/  event loop, биржа, модели исполнения, портфель, пресеты комиссий
mm/analytics/   метрики, markout, экономика PnL
mm/config.py    все настройки; mm/facade.py собирает компоненты из config
analytics/      build_charts.py, графики, summary.json, report.html
tests/          ручные тесты на синтетических данных
```

## 6. Документы

| Файл | Содержание |
|---|---|
| [analytics/report.html](analytics/report.html) | Отчёт: допущения, результат, фазы PnL, чувствительность, выводы |
| [docs/SIMULATOR.md](docs/SIMULATOR.md) | Архитектура, допущения исполнения, определения метрик |
| [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) | Подробный разбор кода по компонентам |
| [results.md](results.md) | Журнал исследования. Часть чисел посчитана с комиссиями Hyperliquid |

## 7. Ограничения

- Данные L2, а не L3: точную позицию в очереди мы не знаем.
- Метки времени — время приёма. Задержку до биржи по данным измерить нельзя.
- Post-only ордера не моделируются: ордер, который пересёк стакан при прибытии, исполняется как taker.
- Глубину 0.45 и skew мы выбрали, уже глядя на 03-20…21. Вне выборки только ML-модель.
