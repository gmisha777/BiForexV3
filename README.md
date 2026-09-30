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

## Мітки, стратегії, ансамбль, бектест

```powershell
.venv\Scripts\python scripts\build_labels.py   # triple-barrier: TP 2 ATR / SL 1 ATR / 12 барів
.venv\Scripts\python scripts\train.py          # walk-forward + фінальні моделі ввімкнених стратегій
.venv\Scripts\python scripts\train.py --strategies lgbm_tb --no-final   # лише перевірка
.venv\Scripts\python scripts\backtest.py       # ансамбль зі стратегіями й вагами з конфігу
.venv\Scripts\python scripts\backtest.py --strategies lgbm_tb other --weights 1 2
```

- **Мітки** — спільні правила виходу для всіх стратегій: для кожного бару що першим досягнуто —
  тейк лонга (1), тейк шорта (−1) чи нічого за горизонт (0). Бари, де торкнулися і тейка, і
  стопа, розбираються за 1m-свічками.
- **Стратегія** — окремий підхід до рішення (`src/biforex/strategies/`). Кожна на закритті свічки
  дає для кожної пари **оцінку від −1 (вниз) до +1 (вгору)**, 0 — нейтрально. Спільні для всіх:
  дані, таймфрейм (`timeframe`), стоп / тейк / ліміт часу (`label`).
- **Ансамбль** (`src/biforex/ensemble.py`): підсумок = Σ(вага × оцінка) / Σ(ваг) по ввімкнених
  стратегіях (`strategies:` у конфігу). Вхід — якщо |підсумок| ≥ `ensemble.enter_score`, вихід
  — якщо підсумок проти позиції досягає `ensemble.exit_score`. Стратегія, що не змогла дати
  оцінку, тимчасово виключається, ваги решти перераховуються.
- **Стратегія `lgbm_tb`** — LightGBM, одна спільна модель на всі пари (`model:` у конфігу).
  Оцінка = сила переважного напрямку: `±(p − 1/3)/(2/3)`, тож `|оцінка| ≥ 0.10` ⇔ «ймовірність
  напрямку ≥ 0.40 і вона переважає». Walk-forward: щомісяця навчання на попередніх 36 місяцях
  з очищеним зазором, тест — наступний місяць.
- **Файли стратегії**: моделі — `models/<стратегія>/`, оцінки на даних, яких модель не бачила,
  і звіти — `reports/<стратегія>/` (`oos_scores.parquet`), бектест ансамблю —
  `reports/ensemble/<стратегії×ваги>/`.
- **Бектест** відтворює правила бота: одна позиція на пару, ризик 1 % до стопу, лімітний вхід
  (живе 1 бар), SL стоп-маркет, TP лімітом, вихід за часом або сильним протилежним сигналом,
  комісії, funding, денний ліміт втрат і зупинка при просадці.

### Як додати нову стратегію

1. Створити `src/biforex/strategies/<назва>.py` з класом-нащадком `Strategy`
   (`strategies/base.py`), позначеним `@register`, з методами:
   - `walk_forward()` → оцінки на історії, яких модель не бачила (`open_time`, `symbol`, `score`);
   - `fit_final()` → навчити на останніх даних і зберегти модель;
   - `load()` і `score(symbol, bar)` → для живого бота.
2. Додати `from . import <назва>` унизу `strategies/__init__.py`.
3. Додати в `config.yaml`:
   ```yaml
   strategies:
     <назва>:
       enabled: true
       weight: 1.0
       params: {...}          # власні налаштування стратегії
   ```
4. `train.py --strategies <назва>` → `backtest.py --strategies lgbm_tb <назва>` — порівняти
   окремо й разом.

Оцінки різних стратегій мають бути співставними: 0.10 — «слабкий, але робочий» сигнал.

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
