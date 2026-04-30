"""
core/binance_client.py — Binance API client dengan multi-endpoint fallback

Fitur:
  - Multi-endpoint fallback (fapi, data-api.binance.vision, spot endpoints)
  - SSL verify=False untuk bypass ISP certificate issues
  - Rate limiting dengan exponential backoff
  - FAPI block detection dengan cooldown
"""

import time
import warnings
from typing import Optional

import requests
from requests.packages import urllib3  # type: ignore

# Disable SSL warnings untuk verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from core.utils import setup_logger

logger = setup_logger("binance_client")

# ─── Multi-endpoint configuration ─────────────────────────────────────────────
# Prioritas: fapi (futures) → data-api.binance.vision (CDN) → spot endpoints
KLINE_ENDPOINTS = [
    ("https://fapi.binance.com/fapi/v1/klines", "futures"),
    ("https://data-api.binance.vision/api/v3/klines", "spot_cdn"),
    ("https://api1.binance.com/api/v3/klines", "spot_api1"),
    ("https://api2.binance.com/api/v3/klines", "spot_api2"),
    ("https://api.binance.com/api/v3/klines", "spot_main"),
]

FAPI_ENDPOINTS = [
    "https://fapi.binance.com",
]

# Cache untuk endpoint yang berhasil
_last_working_kline_endpoint: Optional[str] = None

# FAPI block detection
_fapi_blocked: dict = {"blocked": False, "ts": 0.0}
_FAPI_RETRY_COOLDOWN = 30 * 60  # 30 menit cooldown


class BinanceClient:
    """
    Binance API client dengan multi-endpoint fallback.
    
    Mendukung:
      - Futures API (fapi.binance.com)
      - Spot API dengan CDN (data-api.binance.vision)
      - Automatic fallback jika endpoint utama gagal
      - SSL verify=False untuk bypass ISP issues
    """
    
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        sleep_between: float = 0.12,
        sleep_rate_limit: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        verify_ssl: bool = False,  # Default False untuk Indonesia
    ):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.base_url          = base_url
        self.sleep_between     = sleep_between
        self.sleep_rate_limit  = sleep_rate_limit
        self.max_retries       = max_retries
        self.backoff_base      = backoff_base
        self.verify_ssl        = verify_ssl
        self._last_request_time = 0.0

    # ─── FAPI Block Detection ───────────────────────────────────────────────

    def _is_fapi_available(self) -> bool:
        """Return False jika fapi sedang di-block (cooldown 30 menit)."""
        now = time.time()
        if _fapi_blocked["blocked"] and (now - _fapi_blocked["ts"]) < _FAPI_RETRY_COOLDOWN:
            return False
        return True

    def _mark_fapi_blocked(self) -> None:
        """Tandai fapi sebagai blocked setelah menerima 403."""
        _fapi_blocked["blocked"] = True
        _fapi_blocked["ts"] = time.time()
        logger.warning(
            f"[fapi] fapi.binance.com mengembalikan 403 — kemungkinan geo/IP block. "
            f"Akan di-retry dalam {_FAPI_RETRY_COOLDOWN//60} menit."
        )

    def _mark_fapi_ok(self) -> None:
        """Reset flag jika fapi berhasil diakses."""
        _fapi_blocked["blocked"] = False
        _fapi_blocked["ts"] = 0.0

    # ─── Core Request Methods ───────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict, url_override: str = None) -> Optional[list | dict]:
        """
        GET request dengan retry dan rate limiting.
        
        Args:
            endpoint: API endpoint path (e.g., "/fapi/v1/klines")
            params: Query parameters
            url_override: Full URL override (untuk multi-endpoint)
        """
        url = url_override if url_override else f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.sleep_between:
                time.sleep(self.sleep_between - elapsed)
            
            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=30, verify=self.verify_ssl)

                # Handle 403 Forbidden (geo/IP block)
                if resp.status_code == 403:
                    if "fapi.binance.com" in url:
                        self._mark_fapi_blocked()
                    logger.warning(f"403 Forbidden: {url} — kemungkinan geo block")
                    return None

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", self.sleep_rate_limit))
                    logger.warning(f"Rate limit 429 — tunggu {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if resp.status_code == 418:
                    wait = self.sleep_rate_limit * (attempt + 1)
                    logger.warning(f"IP banned 418 — tunggu {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = self.backoff_base ** attempt
                    logger.warning(f"Server error {resp.status_code} [attempt {attempt+1}] — retry {wait:.1f}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                
                # Handle empty response
                if not resp.text or resp.text.strip() == "":
                    logger.warning(f"Empty response dari {url}")
                    return None
                
                # Mark fapi OK jika berhasil
                if "fapi.binance.com" in url:
                    self._mark_fapi_ok()
                
                try:
                    return resp.json()
                except requests.exceptions.JSONDecodeError:
                    logger.warning(f"JSON decode error dari {url} — response bukan JSON valid")
                    return None

            except requests.exceptions.Timeout:
                wait = self.backoff_base ** attempt
                logger.warning(f"Timeout [attempt {attempt+1}] — retry {wait:.1f}s")
                time.sleep(wait)
            except requests.exceptions.ConnectionError as e:
                wait = self.backoff_base ** attempt
                logger.warning(f"ConnectionError: {e} — retry {wait:.1f}s")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e} | URL: {url} | params: {params}")
                return None
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL Error: {e} — endpoint mungkin di-block")
                return None

        logger.error(f"Semua {self.max_retries} retry gagal: {url}")
        return None

    def _get_multi_endpoint(self, params: dict, endpoints: list, endpoint_type: str = "klines") -> Optional[list]:
        """
        GET request dengan multi-endpoint fallback.
        
        Coba semua endpoint secara berurutan sampai ada yang berhasil.
        Simpan endpoint yang berhasil untuk prioritas berikutnya.
        """
        global _last_working_kline_endpoint
        
        # Prioritaskan endpoint yang terakhir berhasil
        ordered_endpoints = list(endpoints)
        if _last_working_kline_endpoint:
            for i, (url, _) in enumerate(ordered_endpoints):
                if url == _last_working_kline_endpoint:
                    ordered_endpoints.insert(0, ordered_endpoints.pop(i))
                    break
        
        for url, endpoint_name in ordered_endpoints:
            try:
                elapsed = time.time() - self._last_request_time
                if elapsed < self.sleep_between:
                    time.sleep(self.sleep_between - elapsed)
                
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=30, verify=self.verify_ssl)
                
                if resp.status_code == 200:
                    _last_working_kline_endpoint = url
                    logger.info(f"✅ {endpoint_type} via {endpoint_name}: {url.split('/')[2]}")
                    return resp.json()
                elif resp.status_code == 403:
                    logger.debug(f"403 dari {endpoint_name}, coba endpoint berikutnya")
                    continue
                elif resp.status_code == 429:
                    logger.warning(f"Rate limit dari {endpoint_name}, coba endpoint berikutnya")
                    continue
                else:
                    logger.debug(f"{resp.status_code} dari {endpoint_name}, coba endpoint berikutnya")
                    continue
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Request gagal ke {endpoint_name}: {e}")
                continue
        
        logger.warning(f"Semua endpoint gagal untuk {endpoint_type}")
        return None

    # ─── API Methods ─────────────────────────────────────────────────────────

    def get_klines(self, symbol, interval, start_time_ms, end_time_ms, limit=1500):
        """
        Fetch klines dengan multi-endpoint fallback.
        
        Mencoba: fapi → data-api.binance.vision → spot endpoints
        """
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_time_ms,
            "endTime":   end_time_ms,
            "limit":     limit,
        }
        
        # Coba multi-endpoint fallback
        result = self._get_multi_endpoint(params, KLINE_ENDPOINTS, "klines")
        if result:
            return result
        
        # Fallback ke method lama jika semua gagal
        return self._get("/fapi/v1/klines", params)

    def get_klines_spot(self, symbol, interval, limit=500):
        """
        Fetch klines dari spot API (berguna jika fapi di-block).
        """
        params = {
            "symbol":   symbol,
            "interval": interval,
            "limit":    limit,
        }
        return self._get_multi_endpoint(params, KLINE_ENDPOINTS, "klines")

    def get_open_interest_hist(self, symbol, period="1h", start_time_ms=None, end_time_ms=None, limit=500):
        """
        Fetch Open Interest historis.
        Return None jika fapi tidak tersedia (akan di-handle synthetic OI).
        """
        if not self._is_fapi_available():
            logger.debug(f"[OI] fapi skip (cooldown aktif) untuk {symbol}")
            return None
            
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        
        result = self._get("/futures/data/openInterestHist", params)
        if result is None:
            self._mark_fapi_blocked()
        return result

    def get_funding_rate(self, symbol, start_time_ms=None, end_time_ms=None, limit=1000):
        """
        Fetch Funding Rate.
        Return None jika fapi tidak tersedia.
        """
        if not self._is_fapi_available():
            logger.debug(f"[FR] fapi skip (cooldown aktif) untuk {symbol}")
            return None
            
        params = {"symbol": symbol, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        
        result = self._get("/fapi/v1/fundingRate", params)
        if result is None:
            self._mark_fapi_blocked()
        return result

    def get_taker_long_short_ratio(self, symbol, period="1h", start_time_ms=None, end_time_ms=None, limit=500):
        """Fetch Taker Long/Short Ratio."""
        if not self._is_fapi_available():
            return None
            
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        return self._get("/futures/data/takerlongshortRatio", params)

    def get_global_long_short_ratio(self, symbol, period="1h", start_time_ms=None, end_time_ms=None, limit=500):
        """Fetch Global Long/Short Account Ratio."""
        if not self._is_fapi_available():
            return None
            
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms
        return self._get("/futures/data/globalLongShortAccountRatio", params)

    def get_server_time(self):
        """Get server time dari endpoint manapun yang tersedia."""
        # Coba fapi dulu
        result = self._get("/fapi/v1/time", {})
        if result:
            return result.get("serverTime")
        
        # Fallback ke spot API
        for url, name in KLINE_ENDPOINTS[1:]:  # Skip fapi
            try:
                base = url.rsplit("/", 2)[0]  # Remove /klines
                resp = self.session.get(f"{base}/time", timeout=10, verify=self.verify_ssl)
                if resp.status_code == 200:
                    return resp.json().get("serverTime")
            except Exception:
                continue
        return None

    def test_connection(self) -> bool:
        """Test koneksi ke Binance."""
        # Coba ping ke berbagai endpoint
        for url, name in KLINE_ENDPOINTS:
            try:
                base = url.rsplit("/", 2)[0]
                resp = self.session.get(f"{base}/ping", timeout=10, verify=self.verify_ssl)
                if resp.status_code == 200:
                    logger.info(f"✅ Connection OK via {name}")
                    return True
            except Exception as e:
                logger.debug(f"Ping gagal ke {name}: {e}")
                continue
        return False

    def is_fapi_blocked(self) -> bool:
        """Check apakah fapi sedang di-block."""
        return _fapi_blocked["blocked"]
