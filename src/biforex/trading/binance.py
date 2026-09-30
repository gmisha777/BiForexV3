"""Minimal signed REST client for Binance USDT-M futures (real or demo account)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any

import requests

from ..notify import net_error

log = logging.getLogger(__name__)

LIVE_URL = "https://fapi.binance.com"
DEMO_URL = "https://demo-fapi.binance.com"  # Demo Trading (former futures testnet)


class BinanceError(Exception):
    def __init__(self, status: int, code: int | None, msg: str):
        super().__init__(f"HTTP {status}, code {code}: {msg}")
        self.status, self.code, self.msg = status, code, msg


class BinanceFutures:
    def __init__(self, key: str, secret: str, demo: bool, recv_window: int = 10_000):
        if not key or not secret:
            raise ValueError("API key/secret missing (see config/.env)")
        self.base = DEMO_URL if demo else LIVE_URL
        self.demo = demo
        self._key, self._secret = key, secret.encode()
        self.recv_window = recv_window
        self.session = requests.Session()
        self.session.headers["X-MBX-APIKEY"] = key
        self.time_offset = 0
        self.sync_time()

    # --- transport -----------------------------------------------------------------------
    def sync_time(self) -> None:
        server = self._request("GET", "/fapi/v1/time", {}, signed=False)["serverTime"]
        self.time_offset = int(server) - int(time.time() * 1000)

    def _request(self, method: str, path: str, params: dict, signed: bool, retries: int = 3):
        for attempt in range(retries):
            p = {k: v for k, v in params.items() if v is not None}
            if signed:
                p["timestamp"] = int(time.time() * 1000) + self.time_offset
                p["recvWindow"] = self.recv_window
                query = urllib.parse.urlencode(p)
                sig = hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()
                query = f"{query}&signature={sig}"
            else:
                query = urllib.parse.urlencode(p)
            url = f"{self.base}{path}?{query}"
            try:
                resp = self.session.request(method, url, timeout=20)
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                log.warning("%s %s: %s, retrying", method, path, net_error(exc))
                time.sleep(2**attempt)
                continue
            data = resp.json() if resp.content else {}
            if resp.status_code == 200:
                return data
            code = data.get("code") if isinstance(data, dict) else None
            msg = data.get("msg", resp.text) if isinstance(data, dict) else resp.text
            if code == -1021 and attempt < retries - 1:  # timestamp outside recvWindow
                self.sync_time()
                continue
            if resp.status_code in (418, 429) or resp.status_code >= 500:
                if attempt < retries - 1:
                    wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                    log.warning("%s %s: HTTP %d, waiting %ds", method, path, resp.status_code,
                                wait)
                    time.sleep(wait)
                    continue
            raise BinanceError(resp.status_code, code, msg)
        raise BinanceError(0, None, f"{path}: retries exhausted")

    def get(self, path: str, signed: bool = True, **params) -> Any:
        return self._request("GET", path, params, signed)

    def post(self, path: str, **params) -> Any:
        return self._request("POST", path, params, signed=True)

    def delete(self, path: str, **params) -> Any:
        return self._request("DELETE", path, params, signed=True)

    # --- market & account ----------------------------------------------------------------
    def exchange_info(self) -> dict:
        return self.get("/fapi/v1/exchangeInfo", signed=False)

    def last_price(self, symbol: str) -> float:
        return float(self.get("/fapi/v1/ticker/price", signed=False, symbol=symbol)["price"])

    def account(self) -> dict:
        return self.get("/fapi/v3/account")

    def usdt_equity(self) -> tuple[float, float]:
        """(margin balance = wallet + unrealized PnL, available balance) in USDT."""
        acc = self.account()
        for a in acc.get("assets", []):
            if a["asset"] == "USDT":
                return float(a["marginBalance"]), float(a["availableBalance"])
        return 0.0, 0.0

    def position(self, symbol: str) -> dict:
        """One-way mode: signed positionAmt (+ long, - short) and entry price."""
        rows = self.get("/fapi/v3/positionRisk", symbol=symbol)
        for r in rows:
            if r.get("positionSide", "BOTH") == "BOTH":
                return {"qty": float(r["positionAmt"]), "entry": float(r["entryPrice"]),
                        "mark": float(r["markPrice"])}
        return {"qty": 0.0, "entry": 0.0, "mark": 0.0}

    def open_orders(self, symbol: str) -> list[dict]:
        return self.get("/fapi/v1/openOrders", symbol=symbol)

    def order(self, symbol: str, client_id: str) -> dict:
        return self.get("/fapi/v1/order", symbol=symbol, origClientOrderId=client_id)

    def user_trades(self, symbol: str, start_ms: int) -> list[dict]:
        return self.get("/fapi/v1/userTrades", symbol=symbol, startTime=start_ms, limit=1000)

    def income(self, symbol: str, start_ms: int, income_type: str | None = None) -> list[dict]:
        return self.get("/fapi/v1/income", symbol=symbol, startTime=start_ms,
                        incomeType=income_type, limit=1000)

    # --- settings ------------------------------------------------------------------------
    def position_mode_is_one_way(self) -> bool:
        return not self.get("/fapi/v1/positionSide/dual")["dualSidePosition"]

    def set_isolated(self, symbol: str) -> None:
        try:
            self.post("/fapi/v1/marginType", symbol=symbol, marginType="ISOLATED")
        except BinanceError as exc:
            if exc.code != -4046:  # "No need to change margin type"
                raise

    def set_leverage(self, symbol: str, leverage: int) -> None:
        self.post("/fapi/v1/leverage", symbol=symbol, leverage=leverage)

    # --- orders --------------------------------------------------------------------------
    def new_order(self, **params) -> dict:
        log.info("new order %s", params)
        return self.post("/fapi/v1/order", newOrderRespType="RESULT", **params)

    def cancel(self, symbol: str, client_id: str) -> None:
        try:
            self.delete("/fapi/v1/order", symbol=symbol, origClientOrderId=client_id)
        except BinanceError as exc:
            if exc.code != -2011:  # unknown order: already filled or cancelled
                raise

    # Conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET, ...) live in the Algo Order API;
    # /fapi/v1/order rejects them with -4120.
    def new_algo_order(self, **params) -> dict:
        log.info("new algo order %s", params)
        return self.post("/fapi/v1/algoOrder", algoType="CONDITIONAL", **params)

    def open_algo_orders(self, symbol: str) -> list[dict]:
        data = self.get("/fapi/v1/openAlgoOrders", symbol=symbol)
        return data.get("orders", data) if isinstance(data, dict) else data

    def cancel_algo(self, client_algo_id: str) -> None:
        try:
            self.delete("/fapi/v1/algoOrder", clientAlgoId=client_algo_id)
        except BinanceError as exc:
            if exc.code not in (-2011, -2013):  # unknown / already gone
                raise
