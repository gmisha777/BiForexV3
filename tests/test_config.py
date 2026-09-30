from biforex.config import PROJECT_ROOT, load_settings


def test_default_config_loads():
    s = load_settings()
    assert s.symbols == ["BTCUSDT", "ETHUSDT"]
    assert s.timeframe == "4h"
    assert s.model.pooled is True
    assert s.data.dir == PROJECT_ROOT / "data"
