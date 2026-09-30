# BiForexV3

Торговий бот для Binance USDT-M Futures на Python. Наступник системи MT4 + MATLAB
(старі файли — у `info/`).

Щоденна робота з ботом, команди й поради: **[ІНСТРУКЦІЯ.md](ІНСТРУКЦІЯ.md)**.

## Встановлення

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,ml]"
copy config\.env.example config\.env   # і вписати ключі testnet
```

## Дані

```powershell
# повне завантаження або дозавантаження нових даних
.venv\Scripts\python scripts\download_history.py
# лише частина
.venv\Scripts\python scripts\download_history.py --symbols BTCUSDT --intervals 1h --no-metrics
```

Джерела: архіви https://data.binance.vision (місячні свічки й funding, денні metrics) +
REST `fapi.binance.com` для останніх днів. Результат — Parquet у `data/`:

| Файл | Вміст |
|---|---|
| `data/klines/<interval>/<SYMBOL>.parquet` | Закриті свічки (OHLCV, taker buy volume, кількість угод) |
| `data/funding/<SYMBOL>.parquet` | Funding rate (кожні 8 год) |
| `data/metrics/<SYMBOL>.parquet` | Open interest, long/short ratios (5 хв, затримка 1 день) |

Налаштування — `config/config.yaml`.

## Ознаки

```powershell
.venv\Scripts\python scripts\build_features.py
```

Результат — `data/features/<timeframe>/<SYMBOL>.parquet`, рядок на кожну закриту свічку (індекс
`open_time`, рішення приймається о `open_time + timeframe`). Групи ознак:

| Група | Приклади |
|---|---|
| Дохідності | `ret_1` … `ret_144` (лаги Фібоначчі, як у старому експерті) |
| Тренд | Відстань до EMA 8…610 та їх нахил в одиницях ATR, MACD |
| Осцилятори | RSI 14/24, Stochastic, Williams %R, CCI, MFI |
| Волатильність | ATR %, realized volatility, Parkinson, ширина Bollinger |
| Обсяги | Відносний обсяг, частка taker buy, дисбаланс покупців/продавців |
| Свічка й рівні | Тіло/тіні, відстань до max/min за 24 і 168 барів |
| Старші ТФ | `h4h_*`, `h1d_*` — лише з уже закритих 4h/1d свічок |
| Ф'ючерси | Funding, зміна open interest, long/short ratios (NaN там, де даних ще не було) |
| Контекст | `ctx_*` — рух BTC для інших пар, кореляція з BTC |
| Час | Година та день тижня (sin/cos) |

Тест `tests/test_features.py::test_no_lookahead` гарантує, що ознаки не змінюються, коли
дописуються майбутні дані.

## Мітки, модель, бектест

```powershell
.venv\Scripts\python scripts\build_labels.py   # triple-barrier: TP 2 ATR / SL 1 ATR / 12 барів
.venv\Scripts\python scripts\train.py          # walk-forward + фінальна модель у models/
.venv\Scripts\python scripts\backtest.py       # симуляція рахунку на out-of-sample прогнозах
```

- **Мітки**: для кожного бару — що першим досягнуто: тейк лонга (1), тейк шорта (−1) чи нічого
  за горизонт (0). Бари, де торкнулися і тейка, і стопа, розбираються за 1m-свічками.
- **Модель**: LightGBM, одна спільна на всі пари (`model.pooled`, ознака `symbol_id`).
  Walk-forward: щомісяця навчання на попередніх 36 місяцях з очищеним зазором, тест — наступний
  місяць. Звіти — `reports/POOLED/`.
- **Бектест** відтворює правила бота: одна позиція на пару, ризик 1 % до стопу, лімітний вхід
  (живе 1 бар), SL стоп-маркет, TP лімітом, вихід за часом або сильним протилежним сигналом,
  комісії, funding, денний ліміт втрат і зупинка при просадці 15 %.

## Бот

```powershell
.venv\Scripts\python scripts\run_bot.py --dry-run --once   # один цикл рішень, без ордерів
.venv\Scripts\python scripts\run_bot.py                    # demo-рахунок (execution.testnet: true)
.venv\Scripts\python scripts\run_bot.py --live             # РЕАЛЬНИЙ рахунок (testnet: false)
.venv\Scripts\python scripts\bot_status.py                 # капітал, позиції, сигнали, угоди
```

- Кожні 30 с: звірка з біржею. Виконаний вхід → одразу стоп (algo STOP_MARKET, closePosition)
  і тейк (LIMIT reduceOnly); позиція без стопу отримує стоп; після закриття — журнал угоди.
- Після закриття 4h-свічки (+20 с): дозавантаження даних, прогноз, вихід за часом (12 барів)
  або за сильним протилежним сигналом, нові лімітні входи (живуть 1 бар).
- Розмір: 1 % ризику до стопу, але не менше мінімального ордера біржі; угода пропускається,
  якщо мінімальний ордер ризикує > `risk.max_trade_risk_pct` або загальне плече > `max_leverage`.
- Зупинки: файл `STOP` у корені блокує нові входи; денний ліміт втрат; при просадці
  `max_dd_pct` нові входи зупиняються до `bot_status.py --reset-halt`.
- Стан і журнал — `data/bot/<demo|live>.sqlite`, лог — `logs/biforex.log`.
- Один екземпляр на рахунок (файл-блокування). Windows не засинає, поки бот працює.

## Тести

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check src tests scripts
```
