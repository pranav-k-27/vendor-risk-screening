"""Tiny HTTP helper with a clear User-Agent and timeouts."""
from __future__ import annotations

import httpx

USER_AGENT = "vendor-risk-screening/0.1 (portfolio research project)"


def get(url: str, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
    with httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp
