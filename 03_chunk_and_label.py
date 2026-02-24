#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_chunk_and_label.py

Production-grade chunker + weak labeler for RAG corpus.

Reads
-----
- <project_dir>/clean_docs.jsonl   (from Script-02)

Writes
------
- <project_dir>/rag/rag_chunks.jsonl        (chunk records)
- <project_dir>/rag/rag_chunks_stats.json   (summary)
- <project_dir>/rag/rag_chunks_errors.csv   (bad records)

Design goals
------------
- Deterministic chunking (reproducible for a paper)
- Keeps doc-level provenance: pmcid, title, year, license, section heading
- Produces weak topic labels for routing/evaluation:
  * ast
  * blood_culture
  * gram_stain
  * resistance
  * identification
  * infection_site
  * methods_general
  * other

Usage
-----
python 03_chunk_and_label.py --project_dir PMC_Project --target_chunks 200000

Notes
-----
- Chunking uses word-based approximation for tokens (no external tokenizer dependency).
- Script-05 will create embeddings + vector index from rag_chunks.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])")


def norm_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def approx_tokens_from_words(n_words: int) -> int:
    # Rough: 1 token ~ 0.75 words in English biomedical text
    return int(math.ceil(n_words / 0.75))


def split_sentences(text: str) -> List[str]:
    text = norm_text(text)
    if not text:
        return []
    # Conservative sentence split (good enough for chunking)
    sents = SENT_SPLIT_RE.split(text)
    return [norm_text(x) for x in sents if norm_text(x)]


# -------------------------
# Weak labeler (heuristics)
# -------------------------

LABEL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ast", re.compile(r"\b(antimicrobial susceptibility|susceptibility testing|MIC\b|minimum inhibitory|disk diffusion|Etest|broth microdilution|CLSI|EUCAST|S/I/R|susceptible|resistant|intermediate|zone diameter)\b", re.I)),
    ("blood_culture", re.compile(r"\b(blood culture|bacteremia|septicaemia|sepsis|contaminant|contamination|coagulase-negative staphylococc|CoNS|bottle|set of cultures)\b", re.I)),
    ("gram_stain", re.compile(r"\b(gram stain|gram-positive|gram-negative|cocci|bacilli|rods|diplococci)\b", re.I)),
    ("resistance", re.compile(r"\b(ESBL|carbapenemase|CRE\b|MRSA|VRE|mecA|vanA|vanB|bla(KPC|NDM|OXA|CTX-M)|efflux pump|porin|resistance gene)\b", re.I)),
    ("identification", re.compile(r"\b(MALDI-TOF|biochemical identification|16S rRNA|species identification|isolate identified|API 20E|VITEK|BD Phoenix)\b", re.I)),
    ("infection_site", re.compile(r"\b(urinary tract|UTI\b|pneumonia|CSF\b|meningitis|wound|abscess|sputum|urine culture|catheter|central line|endocarditis)\b", re.I)),
    ("methods_general", re.compile(r"\b(methods?|materials and methods|protocol|assay|culture conditions|incubation|media|agar|broth)\b", re.I)),
]


def weak_label(text: str, heading: str = "") -> str:
    h = f"{heading} {text}".strip()
    for lab, pat in LABEL_PATTERNS:
        if pat.search(h):
            return lab
    return "other"


# -------------------------
# Chunking
# -------------------------

@dataclass
class ChunkConfig:
    chunk_min_words: int = 180
    chunk_max_words: int = 450
    chunk_overlap_words: int = 60


def chunk_sentences(
    sents: List[str],
    cfg: ChunkConfig,
) -> List[str]:
    """
    Sentence-aware sliding window chunker.
    - Builds chunks until reaching chunk_max_words
    - Ensures at least chunk_min_words when possible
    - Overlap by ~chunk_overlap_words (sentence-aligned)
    """
    chunks: List[str] = []
    if not sents:
        return chunks

    # Precompute word counts
    sent_words = [len(s.split()) for s in sents]

    i = 0
    n = len(sents)

    while i < n:
        w = 0
        j = i
        # grow until max
        while j < n and (w + sent_words[j]) <= cfg.chunk_max_words:
            w += sent_words[j]
            j += 1

        # If we did not reach min and we can extend a bit (but might exceed max slightly),
        # allow adding one more sentence if it is not too large.
        if w < cfg.chunk_min_words and j < n:
            # Add the next sentence even if it slightly exceeds max
            w2 = w + sent_words[j]
            if w2 <= int(cfg.chunk_max_words * 1.15):
                w = w2
                j += 1

        if j <= i:
            # fallback: force at least one sentence
            j = i + 1

        chunk_text = " ".join(sents[i:j]).strip()
        if chunk_text:
            chunks.append(norm_text(chunk_text))

        # overlap: move i forward, but keep last overlap_words
        if j >= n:
            break

        # Determine new i by stepping back overlap_words from end (sentence aligned)
        overlap_target = cfg.chunk_overlap_words
        back = 0
        k = j - 1
        while k > i and back < overlap_target:
            back += sent_words[k]
            k -= 1
        i = max(i + 1, k + 1)  # move forward at least 1 sentence

    return chunks


def iter_section_texts(doc: dict) -> List[Tuple[str, str]]:
    """
    Returns list of (heading, text) blocks.
    Prefers structured 'sections', but falls back to 'full_text'.
    """
    blocks: List[Tuple[str, str]] = []
    secs = doc.get("sections") or []
    if isinstance(secs, list) and secs:
        for s in secs:
            heading = norm_text(str(s.get("heading", "") or ""))
            text = norm_text(str(s.get("text", "") or ""))
            if text:
                blocks.append((heading, text))
    else:
        ft = norm_text(str(doc.get("full_text", "") or ""))
        if ft:
            blocks.append(("", ft))
    return blocks


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True, help="Project directory used in Script-01/02")
    ap.add_argument("--clean_docs", default="", help="Override input JSONL path (default: <project_dir>/clean_docs.jsonl)")
    ap.add_argument("--out_dir", default="", help="Override output dir (default: <project_dir>/rag/)")
    ap.add_argument("--target_chunks", type=int, default=200000, help="Stop after producing this many chunks (0 = no cap)")
    ap.add_argument("--chunk_min_words", type=int, default=180)
    ap.add_argument("--chunk_max_words", type=int, default=450)
    ap.add_argument("--chunk_overlap_words", type=int, default=60)
    ap.add_argument("--min_block_chars", type=int, default=400, help="Ignore section blocks shorter than this (default: 400)")
    args = ap.parse_args()

    project_dir = Path(args.project_dir).resolve()
    in_jsonl = Path(args.clean_docs).resolve() if args.clean_docs else (project_dir / "clean_docs.jsonl")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_dir / "rag")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "rag_chunks.jsonl"
    out_stats = out_dir / "rag_chunks_stats.json"
    out_err = out_dir / "rag_chunks_errors.csv"

    cfg = ChunkConfig(
        chunk_min_words=args.chunk_min_words,
        chunk_max_words=args.chunk_max_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )

    if not in_jsonl.exists():
        raise FileNotFoundError(f"Missing input: {in_jsonl} (run Script-02 first)")

    # Counters
    total_docs = 0
    total_blocks = 0
    total_chunks = 0
    label_counts: Dict[str, int] = {}
    avg_words_acc = 0
    avg_tok_acc = 0

    with open(in_jsonl, "r", encoding="utf-8") as f_in, \
         open(out_jsonl, "w", encoding="utf-8") as f_out, \
         open(out_err, "w", newline="", encoding="utf-8") as f_err:

        err_writer = csv.DictWriter(f_err, fieldnames=["pmcid", "error"])
        err_writer.writeheader()

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                total_docs += 1

                pmcid = str(doc.get("pmcid", "") or "").strip()
                title = str(doc.get("title", "") or "").strip()
                journal = str(doc.get("journal", "") or "").strip()
                pubyear = str(doc.get("pubyear", "") or "").strip()
                license_text = str(doc.get("license_text", "") or "").strip()
                license_ref = str(doc.get("license_ref", "") or "").strip()

                blocks = iter_section_texts(doc)
                for heading, text in blocks:
                    if len(text) < args.min_block_chars:
                        continue
                    total_blocks += 1

                    # Sentence-aware chunking
                    sents = split_sentences(text)
                    chunks = chunk_sentences(sents, cfg)

                    for k, ch in enumerate(chunks):
                        words = len(ch.split())
                        if words < 80:
                            continue

                        label = weak_label(ch, heading)
                        label_counts[label] = label_counts.get(label, 0) + 1

                        approx_tok = approx_tokens_from_words(words)
                        chunk_id = f"{pmcid}:{sha1((heading + ' ' + ch)[:4000])[:16]}"

                        rec = {
                            "chunk_id": chunk_id,
                            "pmcid": pmcid,
                            "title": title,
                            "journal": journal,
                            "pubyear": pubyear,
                            "license_text": license_text,
                            "license_ref": license_ref,
                            "section_heading": heading,
                            "weak_label": label,
                            "text": ch,
                            "n_words": words,
                            "approx_tokens": approx_tok,
                        }
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        total_chunks += 1
                        avg_words_acc += words
                        avg_tok_acc += approx_tok

                        if args.target_chunks and total_chunks >= args.target_chunks:
                            break

                    if args.target_chunks and total_chunks >= args.target_chunks:
                        break

                if total_docs % 200 == 0:
                    print(f"[INFO] docs={total_docs} blocks={total_blocks} chunks={total_chunks}")

                if args.target_chunks and total_chunks >= args.target_chunks:
                    break

            except Exception as e:
                pmcid = ""
                try:
                    pmcid = json.loads(line).get("pmcid", "")
                except Exception:
                    pmcid = ""
                err_writer.writerow({"pmcid": pmcid, "error": str(e)})

    avg_words = (avg_words_acc / total_chunks) if total_chunks else 0.0
    avg_tok = (avg_tok_acc / total_chunks) if total_chunks else 0.0

    stats = {
        "input_clean_docs": str(in_jsonl),
        "output_rag_chunks": str(out_jsonl),
        "total_docs_parsed": total_docs,
        "total_blocks_used": total_blocks,
        "total_chunks_written": total_chunks,
        "avg_words_per_chunk": round(avg_words, 2),
        "avg_approx_tokens_per_chunk": round(avg_tok, 2),
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "chunk_config": {
            "chunk_min_words": cfg.chunk_min_words,
            "chunk_max_words": cfg.chunk_max_words,
            "chunk_overlap_words": cfg.chunk_overlap_words,
            "min_block_chars": args.min_block_chars,
        },
    }

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote chunks: {out_jsonl}")
    print(f"[INFO] Stats: {out_stats}")
    print(f"[INFO] Errors: {out_err}")
    print(f"[INFO] Summary: docs={total_docs} blocks={total_blocks} chunks={total_chunks}")


if __name__ == "__main__":
    main()
