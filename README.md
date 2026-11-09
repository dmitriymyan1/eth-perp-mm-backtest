# Market Making — Take-Home

## Задание

Спроектировать **market making** бота и **отбэктестить** его на **3 календарных дня** данных (2026-03-19 … 2026-03-21).

Ожидаемый результат: описание стратегии, симулятор (fills, инвентарь, PnL), метрики за период и краткие выводы.

## Данные

Инструмент — **ETH perpetual** (~2200 USD), один рынок, без колонки символа.

| Папка | Файлы | Содержимое |
|-------|--------|------------|
| `data/orderbook/` | по файлу на день | Снимки стакана: 20 уровней bid/ask (`bid_price_i`, `ask_price_i`, `bid_qty_i`, `ask_qty_i`), `datetime` (наносекунды) |
| `data/trades/` | по файлу на день | Сделки: `datetime`, `price`, `size`, `is_maker_ask` (1 — агрессор покупатель, 0 — агрессор продавец) |
| `data/fundings/` | по файлу на день | Прогноз/ставка фандинга: `datetime`, `funding_rate` (~каждые 20 с) |

---

# Реализация

## Запуск

```bash
pip install pandas pyarrow numpy matplotlib pytest
python -m pytest tests -q                                          # 83 ручных теста на синтетических данных
python run_backtest.py                                             # configs/default.toml
python run_backtest.py --config my.toml                            # свой config
python run_backtest.py --set fills.cancel_rule=back --set fees.maker_bps=-0.5
python run_backtest.py --sweep fills.cross_rule=never,confirmed,profitable,always
python run_backtest.py --set fills.cross_rule=profitable \
    --sweep fills.pickoff_fee_bps=2.4,4.5 --sweep fills.pickoff_fair_ms=0,200   # сетка 2 × 2
python toxicity.py output/<run>_fills.parquet                      # токсичность fills по горизонтам
python train_fair_value.py                                         # ML fair value: train 03-19, test 03-20..21
python run_backtest.py --set data.days=2026-03-20,2026-03-21 --set strategy.name=fair_value     --set strategy.signal_path=models/fair_value_signal.parquet --set strategy.pull_threshold_bp=0.3
```

**Все настройки — в одном config** ([configs/default.toml](configs/default.toml), типы и проверки — `mm/config.py`):

| Секция | Ключи |
|---|---|
| `data` | `path`, `levels`, `tick`, `days` |
| `strategy` | `name`, `half_spread`, `lot`, `max_position`, `signal_path`, `inventory_skew`, `signal_cap_bp`, `pull_threshold_bp` |
| `latency` | `market_data_ms`, `order_ms`, `ack_ms`, `max_book_age_s` |
| `fees` | `venue`, `maker_bps`, `taker_bps` |
| `funding` | `interval_hours`, `rate_hours` |
| `fills` | `model`, `cross_rule`, `cancel_rule`, `pickoff_fee_bps`, `pickoff_fair_ms` |
| `output` | `dir`, `sample_interval_s` |

Единицы входят в имя ключа (`_ms`, `_s`, `_hours`, `_bps`). Пропущенный ключ берёт значение по умолчанию.
Неизвестный ключ, неверное значение или не конечное число дают понятную ошибку.
`mm/facade.py` собирает все компоненты из одного config — CLI и тесты используют один и тот же путь.

Результаты: `<output.dir>/<run>_{pnl.png,metrics.json,fills.parquet}`. Имя run содержит модель, правила,
биржу, лот и hash config (прогоны с разными настройками не перезаписывают друг друга);
`metrics.json` содержит полный config этого прогона.

## Архитектура

```
configs/*.toml ─► BacktestConfig ─► facade.run_backtest ─┐ (собирает всё ниже)
                                                         ▼
ParquetDataSource ─► build_events ─► Backtester (event loop + Scheduler)
   (день за днём)    (merge по ts)      │
          ── истинное время ──          │            ── что видит бот (с задержкой) ──
   SimulatedExchange ◄── order entry ───┼───────────── TradingBot
     FillModel (Simple | QueueAware)    │  book (+md latency) ──► Strategy → Quote
     latency, cancel, matching          │  fill (+ack latency) ─► OrderManager → submit/cancel
   Portfolio (истинная позиция и PnL)   │                          known position
   EquityRecorder (раз в 1 с)           │
BacktestResult ─► metrics.compute_metrics / plotting.plot_pnl
```

| Пакет | Модули | Ответственность |
|---|---|---|
| `mm/core/` | `domain.py` | Общие типы: `Side`, `BookSnapshot`, `Trade`, `FundingRate`, `Order`, `Fill`, `FillKind`, `Quote`, хелперы цен |
| `mm/data/` | `loader.py` | Чтение parquet по дням, единый поток событий по времени, mid path |
| `mm/strategy/` | `base.py`, `mid_quoter.py`, `fair_value.py` | `Strategy`, `InventoryLimit`, `MidQuoter`, `FairValueQuoter` (fair value + skew + pull) |
| `mm/signals/` | `features.py`, `signal.py` | Причинные фичи (окна 100 мс…10 с, горизонт 1 с, правило ≤ 100×), target, `SignalTable` |
| `mm/trading/` | `bot.py`, `order_manager.py` | Сторона бота: `TradingBot` (только наблюдения), `OrderManager`, `OrderGateway` (интерфейс биржи) |
| `mm/backtester/` | `engine.py`, `exchange.py`, `fill_models.py`, `portfolio.py`, `venues.py` | Симулятор: event loop + `Scheduler`, биржа, fill models, учёт PnL, пресеты комиссий |
| `mm/analytics/` | `metrics.py`, `plotting.py`, `toxicity.py` | Метрики, графики PnL, токсичность fills по горизонтам |
| `mm/` | `config.py`, `facade.py` | Все настройки; сборка компонентов из config |

Зависимости идут в одну сторону: `core` ← `data`, `strategy` ← `trading` ← `backtester` ← `analytics`;
`config` и `facade` собирают всё. `trading` не зависит от `backtester`: бот говорит с биржей через `OrderGateway`.

Новая стратегия или fill model — новый класс, наследующий `Strategy` / `FillModel`; движок не меняется.

## Допущения симулятора

**Общие**
- Наши ордера не видны в исторических данных и не влияют на них (нет market impact).
  Реальный ордер поглощал бы часть sweep и уменьшал движение цены, поэтому adverse selection, скорее, завышен.
- Сделка и снимок стакана с одинаковым ts: сначала сделка (проверено на данных: снимок уже post-trade).
- Сообщение боту со временем, равным времени рыночного события, приходит **после** события.
  Поэтому sweep (несколько сделок с одним ts) атомарен, даже при нулевой latency.
  Для одного события бот получает снимок стакана раньше fill report, поэтому после fill он котирует от нового стакана.
- Ордер или cancel, который приходит на биржу точно во время события, применяется **до** события
  (на реальных данных таких совпадений нет: время в нс, latency в целых мс).
- Повторяющиеся снимки стакана **не** удаляются: каждый снимок — это наблюдение (для `confirmed` и для таймера).

**Задержки** (номинальные; при снимках раз в ~53 мс выводы о latency делать нельзя)
- `--md-latency-ms 1`: бот видит снимок стакана через 1 мс.
- `--order-latency-ms 2`: ордер и cancel доходят до биржи через 2 мс; до прихода cancel ордер может исполниться.
- `--ack-latency-ms 2`: бот узнаёт о fill через 2 мс; до этого он считает позицию старой.
- `--max-book-age-s 5`: таймер; если новый стакан не пришёл за 5 с, бот снимает котировки (`0` — выкл.).
- `--max-position` (по умолчанию 10 лотов) — **мягкий** лимит: ордера в пути могут его немного превысить.

**Исполнение (`mm/fill_models.py`, `mm/exchange.py`)**
- Maker fill — по нашей лимитной цене.
- У агрессора фиксированный размер. Сделка *сквозь* нашу цену исполняет нас на объём этой сделки
  (стоял бы наш ордер — агрессор сначала взял бы его). Принты одного sweep складываются.
- Сделка исполняет наши ордера по price-time priority; объём принта не используется дважды.
- Ордер, который при приходе на биржу уже пересекает стакан, исполняется как **taker** (GTC):
  весь размер по лучшей встречной цене, taker fee, время = время прихода. Глубина стакана и сделки
  между последним снимком и приходом не проверяются (на данных: лучший уровень < 1 ETH в ~4% снимков).
- **Simple**: мы первые в очереди.
- **QueueAware**: мы встаём в конец видимой очереди. Принт по нашей цене целиком съедает объём перед нами.
  Рост уровня — новые ордера встают за нами. Уменьшение уровня (cancel) — по `--cancel-rule`:
  `back` (сзади нас, пессимистично), `pro_rata` (по умолчанию), `front` (перед нами, оптимистично).
  Объём, который встал в очередь между снимком и приходом нашего ордера, считается стоящим за нами (немного оптимистично).
- Пересечение стакана (`--cross-fills`): противоположная лучшая цена дошла до нашей цены.
  - `never`: исполняют только сделки (нижняя граница объёма).
  - `confirmed` (по умолчанию): остаток исполняется, только если с прошлого снимка встречная сделка
    дошла до нашей цены **и** очередь перед нами пуста (её съели сделки или отмены).
    Немного оптимистично по объёму: на Hyperliquid post-only ордер на нашей цене был бы отклонён.
  - `profitable`: `confirmed` **или** sniper-у выгодно взять нашу старую котировку:
    `(mid нового стакана − наш ask)` или `(наш bid − mid)` > `pickoff fee × наша цена`.
    Логика: если сделок нет, то естественного taker-потока не было; остаться может только sniper,
    а он не торгует в убыток. `fills.pickoff_fee_bps` — комиссия sniper (по умолчанию taker fee биржи, 1.7 bp у `cex_pro`).
    `fills.pickoff_fair_ms` — информационное опережение sniper: его справедливая цена = mid через столько мс
    (он знает цену с других бирж). Это модель контрагента на стороне биржи; бот её не видит, это не look-ahead.
    По анализу токсичности (`toxicity.py`) 90% быстрого хода против нас проходит за ~150–300 мс → разумно ~200 мс.
    При ETH ≈ 2150 и tick 0.1 комиссия 4.5 bp ≈ 9.6 тика, а медианный скачок за cross — 0.5 тика,
    поэтому при 4.5 bp и 0 мс правило почти совпадает с `confirmed`.
  - `always`: любой cross. **Физически невозможная верхняя граница**: на этих данных объём больше всего рынка
    (118%), потому что 87% сдвигов цены без сделок — это перестановка котировок (cancel/repost).
- Частично исполненный ордер не доливается до полного лота, пока цена котировки не изменится.

**Биржа, комиссии, funding**
- **Биржа в задании не указана.** По данным: funding сбрасывается каждый час (скользящее среднее premium за час),
  tick 0.1 при ~2150 USD. Но поток похож на CEX: снимки каждые ~50 мс, метки времени — время приёма,
  сделки с одной меткой всегда одной стороны (непрерывный matching, не блоки Hyperliquid).
- Задержка по умолчанию 1 / 2 / 2 мс (данные / ордер / подтверждение) — colocation на CEX, лучший случай.
  При 10 мс на ордер PnL ML-варианта падает примерно вдвое, при 50 мс уходит в минус.
- Комиссии — пресет `fees.venue`: `cex_pro` (по умолчанию, топ-VIP CEX, как Binance VIP 9) maker **0** / taker 0.017%;
  `cex_mm_rebate` −0.005% / 0.015%; `hyperliquid` 0.015% / 0.045%; `bybit` 0.02% / 0.055% (базовые уровни).
- Post-only (ALO) не моделируется: ордер, который пересёк стакан при прибытии, исполняется как taker.
- Funding — свойство данных, не пресета: расчёт каждые `--funding-hours` (1 ч) по последней ставке до границы.
  Ставка в данных — за `--funding-rate-hours` (1 ч); платёж масштабируется на интервал / период ставки.
  Последний интервал (до полуночи после последнего дня) рассчитывается в последнюю нс данных.
  `payment = -position · mid · rate · scale`; Hyperliquid берёт oracle price, у нас его нет — используем mid.

**Метрики (`mm/metrics.py`)**
- PnL в bp от оборота раскладывается точно: `pnl = spread_capture + inventory - fees + funding`.
  `spread_capture` — цена fill против mid прямо перед событием fill; `inventory` — всё, что после fill
  (adverse selection + движение цены инвентаря). `markout_1s/60s` — цена fill против mid через 1 с / 60 с;
  `adverse_1s = markout_1s - spread_capture`.
- Каждый fill помечен видом (`kind`): `at_price`, `through`, `cross`, `taker`; для каждого вида есть
  доля объёма, spread capture и markout 1 с. Для `cross` «mid перед fill» — это mid до пересечения,
  поэтому его spread capture — условность; смотрите markout.
- Sharpe — по **часовому** PnL, нормировка `sqrt(365·24)`; рядом t-stat Newey–West (5 лагов).
  На 3 днях надёжен только знак.
- `volume_share_of_market` — наш объём / объём всех сделок в данных (sanity check: должно быть намного меньше 1).
- `fill_rate` — доля ордеров хотя бы с одним fill (включая частичные); `taker_fills` — ордера, ставшие taker.
- `mean_abs_inventory` — взвешено по времени; `max_abs_inventory` — точный максимум по fills.

## Результаты (queue model, pro_rata, комиссии Hyperliquid, лот 1 ETH, 2026-03-19 … 21)

Объём всех сделок в данных: 79 652 ETH. bp — от нашего оборота.

| Правило cross | PnL, USD | Доля рынка | PnL, bp | Spread, bp | Adverse 1 с, bp | Fees, bp | NW t (час) |
|---|---|---|---|---|---|---|---|
| `never` | −5 919 | 9.9% | −3.49 | +0.20 | −1.74 | −1.56 | −7.0 |
| **`confirmed`** | **−11 012** | **17.1%** | **−3.76** | +0.22 | −1.82 | −1.54 | −7.1 |
| `profitable`, sniper 4.5 bp, fair 0 мс | −11 102 | 17.2% | −3.77 | +0.22 | −1.85 | −1.53 | −7.0 |
| `profitable`, sniper 4.5 bp, fair 200 мс | −11 688 | 17.8% | −3.85 | +0.22 | −2.03 | −1.53 | −6.7 |
| `profitable`, sniper 2.4 bp, fair 0 мс | −11 582 | 17.8% | −3.80 | +0.22 | −1.93 | −1.53 | −6.7 |
| `profitable`, sniper 2.4 bp, fair 200 мс | −14 904 | 21.9% | −3.99 | +0.23 | −2.35 | −1.53 | −5.8 |
| `always` (невозможно) | −62 034 | 117.7% | −3.09 | +0.24 | −1.45 | −1.50 | −5.3 |

Markout 1 с по виду fill (`confirmed`): `at_price` −0.85 bp, `cross` −1.71 bp, `through` −2.62 bp, `taker` −2.10 bp.

**Выводы**
- **Знак надёжен: стратегия убыточна** при любом правиле cross, правиле очереди и уровне комиссий.
  Причина структурная: при спреде 1 тик (≈0.47 bp) fill на лучшей цене даёт не больше ≈0.23 bp,
  а комиссия maker 1.5 bp и adverse selection ≈1.8 bp за 1 с намного больше.
- **Размер в USD не надёжен** (≈2× между `never` и `confirmed`); используйте bp от оборота.
- Около 75–80% `inventory` — это adverse selection в первую секунду, а не медленный дрейф цены инвентаря.
- Проверка: 3 раунда red-team (13 агентов); точный пересчёт PnL, fees, funding и метрик из сырых данных
  совпадает с движком до ~1e-9 USD; look-ahead не найден.
