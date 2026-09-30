import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from biforex.data import bulk, store
from biforex.data.quality import check_klines
from biforex.data.schema import normalize_funding, normalize_klines

HOUR_MS = 3_600_000
T0 = 1_577_836_800_000  # 2020-01-01 00:00 UTC


def kline_row(i: int, scale: int = 1) -> list:
    t = T0 + i * HOUR_MS
    return [t * scale, 100, 110, 90, 105, 10.0, (t + HOUR_MS - 1) * scale, 1050.0, 5, 4.0, 420.0, 0]


def zipped(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, timeout):
        return self.response


def csv(rows: list[list], header: str | None = None) -> str:
    lines = [",".join(map(str, r)) for r in rows]
    return "\n".join(([header] if header else []) + lines)


@pytest.mark.parametrize("header", [None, ",".join([
    "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
    "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])])
def test_fetch_zip_csv_detects_header(header):
    session = FakeSession(FakeResponse(zipped(csv([kline_row(0), kline_row(1)], header))))
    df = normalize_klines(bulk.fetch_zip_csv("u", session))
    assert len(df) == 2
    assert df["open_time"].iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert df["close"].dtype == "float64"
    assert "ignore" not in df.columns


def test_fetch_zip_csv_missing_archive_returns_none():
    assert bulk.fetch_zip_csv("u", FakeSession(FakeResponse(b"", 404))) is None


def test_microsecond_timestamps_are_detected():
    df = normalize_klines(pd.DataFrame([kline_row(0, scale=1000)]))
    assert df["open_time"].iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")


def test_funding_archive_and_rest_formats_match():
    archive = normalize_funding(pd.DataFrame(
        {"calc_time": [T0 + 1], "funding_interval_hours": [8], "last_funding_rate": [0.0001]}))
    api = normalize_funding(pd.DataFrame({"fundingTime": [T0 + 3], "fundingRate": ["0.0001"]}))
    assert archive.equals(api)


def test_month_range_crosses_year():
    assert bulk.month_range(date(2023, 11, 15), date(2024, 2, 1)) == [
        (2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_merge_deduplicates_newest_wins():
    a = normalize_klines(pd.DataFrame([kline_row(0), kline_row(1)]))
    b = a.iloc[[1]].copy()
    b["close"] = 999.0
    merged = store.merge([a, b, None], "open_time")
    assert len(merged) == 2
    assert merged["close"].iloc[1] == 999.0


def test_quality_reports_gaps_and_bad_bars():
    df = normalize_klines(pd.DataFrame([kline_row(i) for i in (0, 1, 2, 5, 6)]))
    df.loc[4, "high"] = 50  # high below open/close
    report = check_klines(df, "1h")
    assert len(report.gaps) == 1 and report.missing_bars == 2
    assert report.bad_ohlc == 1
    assert not report.ok


def test_save_load_roundtrip(tmp_path):
    df = normalize_klines(pd.DataFrame([kline_row(0)]))
    path = store.klines_path(tmp_path, "BTCUSDT", "1h")
    store.save(df, path)
    assert store.load(path).equals(df)
    assert store.load(tmp_path / "none.parquet") is None
