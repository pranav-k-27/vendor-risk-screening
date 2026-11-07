"""Download the full OFAC SDN and UN consolidated lists into data/sanctions/.

Run:  python -m scripts.download_sanctions
Then: python -m scripts.download_sanctions --check   (prints entry counts per list)

URLs change occasionally. If a download fails, get the file manually from the
official page and save it under the file name shown below:
  OFAC: https://ofac.treasury.gov/specially-designated-nationals-and-blocked-persons-list-sdn-human-readable-lists
        -> sdn.csv saved as ofac_sdn.csv, alt.csv saved as ofac_alt.csv
  UN:   https://main.un.org/securitycouncil/en/content/un-sc-consolidated-list  (XML)
        -> saved as un_consolidated.xml
"""
from __future__ import annotations

import argparse
import sys

from core import http
from core.config import sanctions_dir

SOURCES = {
    "ofac_sdn.csv": [
        "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV",
        "https://www.treasury.gov/ofac/downloads/sdn.csv",
    ],
    "ofac_alt.csv": [
        "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ALT.CSV",
        "https://www.treasury.gov/ofac/downloads/alt.csv",
    ],
    "un_consolidated.xml": [
        "https://scsanctions.un.org/resources/xml/en/consolidated.xml",
    ],
}


def download() -> int:
    out = sanctions_dir()
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for fname, urls in SOURCES.items():
        for url in urls:
            try:
                resp = http.get(url, timeout=120)
                (out / fname).write_bytes(resp.content)
                print(f"OK   {fname:22s} {len(resp.content)/1e6:6.2f} MB  <- {url}")
                break
            except Exception as exc:
                print(f"FAIL {fname:22s} {url}: {exc}")
        else:
            failures += 1
    return failures


def check() -> None:
    from collections import Counter

    from core.sanctions_matcher import SanctionsMatcher

    m = SanctionsMatcher.from_directory()
    counts = Counter(e.list_name for e in m.entries)
    for name in ("OFAC_SDN", "UN_CONSOLIDATED", "SYNTHETIC_TEST"):
        print(f"{name:18s} {counts.get(name, 0):6d} names (incl. aliases)  version={m.versions.get(name, 'MISSING')}")
    if counts.get("OFAC_SDN", 0) < 1000 or counts.get("UN_CONSOLIDATED", 0) < 50:
        print("WARNING: list looks incomplete; sanctions source will be reported as FAILED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        sys.exit(1 if download() else 0)
