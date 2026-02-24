#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_pmc_oa_download.py - PMC Open Access XML Downloader (MINIMAL CHANGE + PAGINATION)

Minimal fixes added:
1) ESearch now uses history (usehistory=y) and returns WebEnv + QueryKey + Count
2) New CLI options: --retstart, --page_size
3) Pulls IDs in pages until max_results reached (starting at retstart)
4) Resume-safe: skips PMCIDs that already exist in raw_pmc_xml/
5) FIX: oa.fcgi is queried with "PMC{digits}" (oa.fcgi expects PMCID form)
"""

import argparse
import csv
import io
import random
import re
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import xml.etree.ElementTree as ET

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_FCGI = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_filename(name: str) -> str:
    return "".join(x if x.isalnum() or x in "._-" else "_" for x in name).strip("_")


def sleep_jitter(base_seconds: float) -> None:
    time.sleep(base_seconds + random.random() * base_seconds * 2.0)


def normalize_pmcid(x: str) -> str:
    """
    Normalize to 'PMC1234567' form (oa.fcgi expects this form reliably).
    Accepts: 'PMC123', '123'
    """
    x = (x or "").strip()
    if not x:
        return ""
    if x.upper().startswith("PMC"):
        digits = re.sub(r"\D", "", x)
        return f"PMC{digits}" if digits else ""
    digits = re.sub(r"\D", "", x)
    return f"PMC{digits}" if digits else ""


class HttpClient:
    def __init__(self, email: str, api_key: Optional[str], timeout: int = 45, min_delay: float = 0.34):
        self.sess = requests.Session()
        self.timeout = timeout
        self.min_delay = min_delay
        self.last_ts = 0.0
        self.email = email
        self.api_key = api_key

        self.sess.headers.update(
            {
                "User-Agent": f"PMC-OA-Downloader/1.1 (email: {email})",
                "Accept": "*/*",
            }
        )

    def _throttle(self) -> None:
        now = time.time()
        delta = now - self.last_ts
        if delta < self.min_delay:
            sleep_jitter(self.min_delay - delta)
        self.last_ts = time.time()

    def get(self, url: str, params: Optional[dict] = None, stream: bool = False, max_retries: int = 5) -> requests.Response:
        if params is None:
            params = {}
        if url.startswith(NCBI_EUTILS) or url.startswith(PMC_OA_FCGI):
            params["email"] = self.email
            if self.api_key:
                params["api_key"] = self.api_key

        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            try:
                print(f"[INFO] Attempt {attempt} to fetch: {url}")
                r = self.sess.get(url, params=params, timeout=self.timeout, stream=stream)
                r.raise_for_status()
                return r
            except Exception as e:
                print(f"[ERROR] Error during GET request (Attempt {attempt}): {str(e)}")
                if attempt >= max_retries:
                    raise
                backoff = min(20.0, (2 ** (attempt - 1)) * 0.8) + random.random() * 0.5
                print(f"[WARN] Retrying due to error, sleeping for {backoff:.1f} seconds...")
                time.sleep(backoff)


@dataclass
class ArticleMeta:
    pmcid: str
    pmid: str = ""
    doi: str = ""
    title: str = ""
    journal: str = ""
    pubyear: str = ""
    oa_status: str = ""
    license_text: str = ""
    license_ref: str = ""
    xml_url: str = ""
    pdf_url: str = ""
    tar_url: str = ""
    xml_path: str = ""
    retrieved_at_utc: str = ""


# -----------------------------
# MINIMAL CHANGE: ESearch history + pagination
# -----------------------------
def eutils_esearch_history_pmc(client: HttpClient, term: str) -> Tuple[int, str, str]:
    """
    Returns (count, query_key, webenv) for a history-backed PMC search.
    """
    full_term = f"({term}) AND open access[filter]"
    params = {
        "db": "pmc",
        "term": full_term,
        "retmode": "xml",
        "usehistory": "y",
        "retmax": "0",
    }
    r = client.get(f"{NCBI_EUTILS}/esearch.fcgi", params=params)
    root = ET.fromstring(r.text)

    count_txt = (root.findtext("./Count") or "0").strip()
    qk = (root.findtext("./QueryKey") or "").strip()
    webenv = (root.findtext("./WebEnv") or "").strip()

    try:
        count = int(count_txt)
    except Exception:
        count = 0

    return count, qk, webenv


def eutils_esearch_page_pmc(client: HttpClient, query_key: str, webenv: str, retstart: int, retmax: int) -> List[str]:
    """
    Fetch one page of IDs using history.
    Returns internal PMC IDs (numeric strings).
    """
    params = {
        "db": "pmc",
        "query_key": query_key,
        "WebEnv": webenv,
        "retmode": "xml",
        "retstart": str(retstart),
        "retmax": str(retmax),
    }
    r = client.get(f"{NCBI_EUTILS}/esearch.fcgi", params=params)
    root = ET.fromstring(r.text)
    ids = [elem.text.strip() for elem in root.findall("./IdList/Id") if elem.text]
    return ids


def eutils_esummary_pmc(client: HttpClient, pmc_ids: List[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    batch_size = 50
    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i : i + batch_size]
        params = {"db": "pmc", "id": ",".join(batch), "retmode": "xml"}
        r = client.get(f"{NCBI_EUTILS}/esummary.fcgi", params=params)
        root = ET.fromstring(r.text)
        for docsum in root.findall("./DocSum"):
            pmcid = ""
            pmid = ""
            doi = ""
            title = ""
            journal = ""
            pubyear = ""
            for item in docsum.findall("./Item"):
                name = item.get("Name", "")
                if name == "ArticleIds":
                    for sub_item in item.findall("./Item"):
                        sub_name = sub_item.get("Name", "")
                        sub_text = (sub_item.text or "").strip()
                        if sub_name == "pmcid":
                            pmcid = normalize_pmcid(sub_text)  # normalize to PMCxxxx
                        elif sub_name == "pmid":
                            pmid = sub_text
                        elif sub_name == "doi":
                            doi = sub_text
                elif name == "Title":
                    title = (item.text or "").strip()
                elif name == "FullJournalName":
                    journal = (item.text or "").strip()
                elif name == "Source" and not journal:
                    journal = (item.text or "").strip()
                elif name == "PubDate":
                    pubdate = (item.text or "").strip()
                    m = re.search(r"\b(19|20)\d{2}\b", pubdate)
                    if m:
                        pubyear = m.group(0)
            if not pmcid:
                continue
            out[pmcid] = {
                "pmcid": pmcid,
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "journal": journal,
                "pubyear": pubyear,
            }
    return out


def oa_fcgi_links(client: HttpClient, pmcid: str) -> Tuple[str, str, str, str, str, str]:
    """
    Query oa.fcgi to find OA links and license.

    Critical fix:
    - Always send id=PMCxxxxxxx (not bare numeric internal id).
    """
    pmcid = normalize_pmcid(pmcid)
    if not pmcid:
        return ("", "", "", "", "", "")

    params = {"id": pmcid}
    r = client.get(PMC_OA_FCGI, params=params)
    root = ET.fromstring(r.text)

    record = root.find(".//record")
    if record is None:
        return ("", "", "", "", "", "")

    license_text = record.get("license", "") or ""
    oa_status = record.get("status", "") or ("open-access" if license_text else "")

    xml_url = ""
    pdf_url = ""
    tar_url = ""

    for link in record.findall("link"):
        fmt = (link.get("format") or "").lower().strip()
        href = (link.get("href") or "").strip()
        if not href:
            continue
        if fmt == "xml":
            xml_url = href
        elif fmt == "pdf":
            pdf_url = href
        elif fmt in ("tgz", "tar.gz"):
            tar_url = href

    return (oa_status, license_text, "", xml_url, pdf_url, tar_url)


def download_file(client: HttpClient, url: str, out_path: Path, max_bytes: Optional[int] = None) -> None:
    """
    Download file with support for FTP and HTTP/HTTPS, plus tar.gz extraction.
    Minimal: keep your logic, plus apply max_bytes enforcement.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original_url = url
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        url = url.replace("ftp://", "https://", 1)
        print(f"[INFO] Converted FTP URL to HTTPS: {url}")

    print(f"[INFO] Downloading: {url}")
    try:
        if url.startswith("ftp://"):
            with urllib.request.urlopen(url, timeout=client.timeout) as response:
                data = response.read()
        else:
            with client.get(url, stream=False) as r:
                data = r.content
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

    if max_bytes is not None and len(data) > max_bytes:
        raise RuntimeError(f"Download exceeded max_bytes={max_bytes}: {original_url}")

    if original_url.endswith((".tar.gz", ".tgz")):
        print("[INFO] Extracting XML from tar.gz package...")
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                members = tar.getmembers()
                xml_members = [m for m in members if m.name.lower().endswith((".xml", ".nxml"))]
                if not xml_members:
                    raise RuntimeError(f"No XML/NXML in tar.gz. Files={ [m.name for m in members] }")
                m = xml_members[0]
                f = tar.extractfile(m)
                if f is None:
                    raise RuntimeError(f"Could not extract {m.name}")
                out_path.write_bytes(f.read())
        except Exception as e:
            raise RuntimeError(f"tar.gz extraction failed: {str(e)}")
    else:
        out_path.write_bytes(data)


def write_metadata_csv(path: Path, rows: List[ArticleMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(ArticleMeta(pmcid="").__dict__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="PMC search query (will be combined with open access[filter])")
    ap.add_argument("--max_results", type=int, default=500, help="How many records to process in this run")
    ap.add_argument("--retstart", type=int, default=0, help="Start offset for paging (0, 5000, 10000, ...)")
    ap.add_argument("--page_size", type=int, default=500, help="IDs per page fetched from esearch history (<= 5000 recommended)")
    ap.add_argument("--out_dir", required=True, help="Output project directory")
    ap.add_argument("--email", default="your_email@example.com", help="Email for NCBI etiquette")
    ap.add_argument("--api_key", default="", help="NCBI API key (optional but recommended)")
    ap.add_argument("--min_delay", type=float, default=0.34, help="Minimum seconds between requests")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing XML files if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    raw_xml_dir = out_dir / "raw_pmc_xml"
    meta_csv_path = out_dir / "metadata.csv"
    raw_xml_dir.mkdir(parents=True, exist_ok=True)

    client = HttpClient(email=args.email, api_key=(args.api_key or None), min_delay=args.min_delay)

    # Resume-safe: skip existing downloads
    existing = set()
    for p in raw_xml_dir.glob("PMC*.xml"):
        existing.add(p.stem)

    print(f"[INFO] Searching PMC OA for: {args.query}")
    count, qk, webenv = eutils_esearch_history_pmc(client, args.query)
    if not qk or not webenv or count <= 0:
        print("[WARN] No results returned (or history not available). Try relaxing the query.", file=sys.stderr)
        write_metadata_csv(meta_csv_path, [])
        return

    print(f"[INFO] Search Count={count} | retstart={args.retstart} | max_results(this run)={args.max_results}")

    # Collect internal IDs with pagination until max_results
    pmc_internal_ids: List[str] = []
    retstart = max(0, int(args.retstart))
    while len(pmc_internal_ids) < args.max_results and retstart < count:
        need = args.max_results - len(pmc_internal_ids)
        this_page = min(int(args.page_size), need)
        ids = eutils_esearch_page_pmc(client, qk, webenv, retstart=retstart, retmax=this_page)
        if not ids:
            break
        pmc_internal_ids.extend(ids)
        retstart += len(ids)
        print(f"[INFO] Collected {len(pmc_internal_ids)} internal PMC IDs so far...")

    if not pmc_internal_ids:
        print("[WARN] No IDs collected for this page range.", file=sys.stderr)
        write_metadata_csv(meta_csv_path, [])
        return

    print(f"[INFO] Found {len(pmc_internal_ids)} PMC records (internal IDs). Fetching summaries...")
    summaries = eutils_esummary_pmc(client, pmc_internal_ids)
    pmcids = sorted(summaries.keys())

    print(f"[INFO] Resolved {len(pmcids)} PMCIDs with metadata. Discovering OA links + downloading XML...")

    rows: List[ArticleMeta] = []
    downloaded = 0
    skipped_no_xml = 0
    skipped_exists = 0
    failed = 0

    for idx, pmcid in enumerate(pmcids, start=1):
        pmcid = normalize_pmcid(pmcid)
        base_meta = summaries.get(pmcid, {})

        art = ArticleMeta(
            pmcid=pmcid,
            pmid=base_meta.get("pmid", ""),
            doi=base_meta.get("doi", ""),
            title=base_meta.get("title", ""),
            journal=base_meta.get("journal", ""),
            pubyear=base_meta.get("pubyear", ""),
            retrieved_at_utc=utc_now_iso(),
        )

        try:
            oa_status, license_text, license_ref, xml_url, pdf_url, tar_url = oa_fcgi_links(client, pmcid)

            art.oa_status = oa_status
            art.license_text = license_text
            art.license_ref = license_ref
            art.xml_url = xml_url
            art.pdf_url = pdf_url
            art.tar_url = tar_url

            download_url = xml_url or tar_url
            if not download_url:
                skipped_no_xml += 1
                rows.append(art)
                continue

            xml_path = raw_xml_dir / f"{safe_filename(pmcid)}.xml"
            art.xml_path = str(xml_path)

            if (pmcid in existing or xml_path.exists()) and not args.overwrite:
                skipped_exists += 1
                rows.append(art)
                continue

            download_file(client, download_url, xml_path, max_bytes=50 * 1024 * 1024)
            existing.add(pmcid)
            downloaded += 1
            rows.append(art)

        except Exception as e:
            failed += 1
            rows.append(art)
            print(f"[ERROR] Failed PMCID={pmcid}: {e}", file=sys.stderr)

        if idx % 50 == 0:
            print(f"[INFO] Progress {idx}/{len(pmcids)} | downloaded={downloaded} exists={skipped_exists} no_xml={skipped_no_xml} failed={failed}")

    write_metadata_csv(meta_csv_path, rows)
    print("\n[INFO] Done.")
    print(f"[INFO] Output: {raw_xml_dir}")
    print(f"[INFO] Metadata: {meta_csv_path}")
    print(f"[INFO] Summary: downloaded={downloaded}, exists={skipped_exists}, no_xml={skipped_no_xml}, failed={failed}")
    print(f"[INFO] Next retstart suggestion: {args.retstart + args.max_results}")


if __name__ == "__main__":
    main()
