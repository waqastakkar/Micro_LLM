#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_pmc_xml_to_clean_text.py

Production-grade parser: PMC full-text XML -> cleaned plain text / JSONL docs.

What it does
------------
1) Reads PMC JATS XML files from: <in_dir>/raw_pmc_xml/*.xml
2) Extracts:
   - Title
   - Abstract (optional)
   - Body sections (sec)
3) Removes boilerplate and low-value content:
   - References
   - Footnotes
   - Acknowledgements, funding, author contributions (configurable)
4) Writes cleaned documents as JSONL:
   - <out_dir>/clean_docs.jsonl  (one record per PMCID)
   - Each record contains: pmcid, title, year, journal, license, sections[{heading,text}], full_text

Why JSONL (not .txt)
--------------------
- Keeps structure for later chunk labeling (AST vs blood culture vs resistance)
- Easier audit + reproducibility

Usage
-----
python 02_pmc_xml_to_clean_text.py \
  --project_dir PMC_Project \
  --out_dir PMC_Project \
  --keep_abstract \
  --min_section_chars 400

Inputs expected
---------------
- Raw XML:   <project_dir>/raw_pmc_xml/*.xml
- Metadata:  <project_dir>/metadata.csv  (from Script-01)

Outputs
-------
- Clean docs JSONL: <out_dir>/clean_docs.jsonl
- Error log:        <out_dir>/parse_errors.csv

Notes
-----
- This script does not do chunking yet (Script-03 will).
- Uses only Python stdlib (no lxml) for portability.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


# -------------------------
# Text utilities
# -------------------------

WS_RE = re.compile(r"\s+")
CIT_RE = re.compile(r"\[(\d+|[A-Za-z]+\d+|\d+(,\s*\d+)*)\]")  # simple [1], [1,2], [Smith2020]
PAREN_CIT_RE = re.compile(r"\(([^)]{0,120}\d{4}[^)]{0,120})\)")  # (Author, 2020) style (weak)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
URL_RE = re.compile(r"\bhttps?://\S+\b")


def norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


def strip_low_value_noise(s: str) -> str:
    # Remove obvious URLs/emails in text (rare but can appear in footnotes)
    s = URL_RE.sub("", s)
    s = EMAIL_RE.sub("", s)
    # Lightly remove bracket citation tokens; keep content readable
    s = CIT_RE.sub("", s)
    # Do NOT remove all parenthetical citations; it can remove biological statements.
    return norm_text(s)


def is_low_value_heading(h: str) -> bool:
    h0 = norm_text(h).lower()
    bad = {
        "references",
        "reference",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "funding",
        "author contributions",
        "contributions",
        "ethics",
        "ethical approval",
        "consent",
        "competing interests",
        "conflict of interest",
        "data availability",
        "availability of data and materials",
        "supplementary material",
        "supplementary materials",
        "abbreviations",
    }
    # Some articles use "Materials and methods" (valuable) - keep it.
    return h0 in bad


# -------------------------
# Metadata loading
# -------------------------

@dataclass
class MetaRow:
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


def load_metadata_csv(path: Path) -> Dict[str, MetaRow]:
    meta: Dict[str, MetaRow] = {}
    if not path.exists():
        return meta
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid:
                continue
            meta[pmcid] = MetaRow(**{k: (v or "").strip() for k, v in row.items()})
    return meta


# -------------------------
# XML extraction
# -------------------------

def _text_from_node(node: ET.Element) -> str:
    """
    Extract visible text content from an XML node, preserving paragraph breaks.
    This is a conservative text extractor for JATS.
    """
    parts: List[str] = []

    def rec(n: ET.Element):
        tag = n.tag.lower().split("}")[-1]
        # Skip some clearly non-body content
        if tag in {"xref", "sup", "sub"}:
            # Keep their text if it exists, but do not recurse deeply
            if n.text:
                parts.append(n.text)
            if n.tail:
                parts.append(n.tail)
            return

        if tag in {"table-wrap", "fig", "graphic", "media", "disp-formula"}:
            # Skip figures and tables here; we can parse tables later if needed
            if n.tail:
                parts.append(n.tail)
            return

        # Paragraph boundaries
        if tag in {"p", "title", "sec"}:
            if n.text:
                parts.append(n.text)
        else:
            if n.text:
                parts.append(n.text)

        for ch in list(n):
            rec(ch)

        if n.tail:
            parts.append(n.tail)

        if tag in {"p"}:
            parts.append("\n")

    rec(node)
    text = "".join(parts)
    # Normalize newlines (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return norm_text(text.replace("\n\n", "\n").replace("\n", "\n"))


def find_first_text(root: ET.Element, xpath_list: List[str]) -> str:
    for xp in xpath_list:
        el = root.find(xp)
        if el is not None:
            t = norm_text("".join(el.itertext()))
            if t:
                return t
    return ""


def extract_sections_from_article(root: ET.Element, keep_abstract: bool) -> Tuple[str, str, List[Dict[str, str]]]:
    """
    Returns (title, abstract_text, sections)
    sections: list of {heading, text}
    """
    # Title
    title = find_first_text(
        root,
        [
            ".//article-title",
            ".//front//title-group//article-title",
        ],
    )

    # Abstract
    abstract_text = ""
    if keep_abstract:
        abs_el = root.find(".//abstract")
        if abs_el is not None:
            abstract_text = strip_low_value_noise(" ".join(abs_el.itertext()))

    # Body sections
    sections: List[Dict[str, str]] = []
    body = root.find(".//body")
    if body is None:
        return title, abstract_text, sections

    # JATS structure: body/sec with nested secs
    for sec in body.findall("./sec"):
        heading_el = sec.find("./title")
        heading = strip_low_value_noise(" ".join(heading_el.itertext())) if heading_el is not None else ""
        # Collect visible text for this section including its subsections
        text = strip_low_value_noise(" ".join(sec.itertext()))
        # Remove heading text duplication if present
        if heading and text.lower().startswith(heading.lower()):
            text = text[len(heading):].strip()

        if heading and is_low_value_heading(heading):
            continue

        sections.append({"heading": heading, "text": text})

    return title, abstract_text, sections


def compact_sections(sections: List[Dict[str, str]], min_chars: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for s in sections:
        heading = norm_text(s.get("heading", ""))
        text = norm_text(s.get("text", ""))
        # Remove extremely short sections
        if len(text) < min_chars:
            continue
        out.append({"heading": heading, "text": text})
    return out


def build_full_text(title: str, abstract_text: str, sections: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    if title:
        parts.append(title)
    if abstract_text:
        parts.append("ABSTRACT: " + abstract_text)
    for s in sections:
        h = s.get("heading", "")
        t = s.get("text", "")
        if h:
            parts.append(h + ": " + t)
        else:
            parts.append(t)
    return norm_text("\n\n".join(parts))


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True, help="Project directory that contains raw_pmc_xml/ and metadata.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory (will write clean_docs.jsonl + parse_errors.csv)")
    ap.add_argument("--keep_abstract", action="store_true", help="Keep abstract text in output")
    ap.add_argument("--min_section_chars", type=int, default=350, help="Drop sections shorter than this (default: 350)")
    ap.add_argument("--max_docs", type=int, default=0, help="Optional cap for number of XML docs (0 = no cap)")
    args = ap.parse_args()

    project_dir = Path(args.project_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    raw_dir = project_dir / "raw_pmc_xml"
    meta_path = project_dir / "metadata.csv"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing directory: {raw_dir}")

    meta = load_metadata_csv(meta_path)

    out_jsonl = out_dir / "clean_docs.jsonl"
    err_csv = out_dir / "parse_errors.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(raw_dir.glob("PMC*.xml"))
    if args.max_docs and args.max_docs > 0:
        xml_files = xml_files[: args.max_docs]

    total = len(xml_files)
    if total == 0:
        print("[WARN] No XML files found in raw_pmc_xml/.", file=sys.stderr)
        return

    n_ok = 0
    n_err = 0

    with open(out_jsonl, "w", encoding="utf-8") as wj, open(err_csv, "w", newline="", encoding="utf-8") as we:
        err_writer = csv.DictWriter(we, fieldnames=["pmcid", "xml_path", "error"])
        err_writer.writeheader()

        for i, xml_path in enumerate(xml_files, start=1):
            pmcid = xml_path.stem
            try:
                tree = ET.parse(str(xml_path))
                root = tree.getroot()

                title, abstract_text, sections = extract_sections_from_article(root, keep_abstract=args.keep_abstract)
                sections = compact_sections(sections, min_chars=args.min_section_chars)

                # If title missing, fallback to metadata
                m = meta.get(pmcid)
                if not title and m and m.title:
                    title = m.title

                full_text = build_full_text(title, abstract_text, sections)

                # Drop empty docs
                if len(full_text) < 500:
                    raise ValueError("Extracted full_text too short or empty")

                rec = {
                    "pmcid": pmcid,
                    "pmid": (m.pmid if m else ""),
                    "doi": (m.doi if m else ""),
                    "title": title,
                    "journal": (m.journal if m else ""),
                    "pubyear": (m.pubyear if m else ""),
                    "license_text": (m.license_text if m else ""),
                    "license_ref": (m.license_ref if m else ""),
                    "source": "pmc_oa_xml",
                    "sections": sections,
                    "full_text": full_text,
                }
                wj.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1

            except Exception as e:
                n_err += 1
                err_writer.writerow({"pmcid": pmcid, "xml_path": str(xml_path), "error": str(e)})

            if i % 100 == 0 or i == total:
                print(f"[INFO] Parsed {i}/{total} | ok={n_ok} err={n_err}")

    print(f"[INFO] Wrote: {out_jsonl}")
    print(f"[INFO] Errors: {err_csv}")
    print(f"[INFO] Summary: ok={n_ok}, err={n_err}")


if __name__ == "__main__":
    main()
