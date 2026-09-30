import requests

from biforex.notify import net_error

TOKEN = "123456:SECRET_TOKEN_VALUE"


def test_network_error_text_never_contains_url_or_token():
    exc = requests.exceptions.ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: "
        f"/bot{TOKEN}/sendMessage (Caused by NameResolutionError(...))")
    text = net_error(exc)
    assert TOKEN not in text and "api.telegram.org" not in text
    assert text == "ConnectionError (NameResolutionError)"
