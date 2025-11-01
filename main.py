"""
E-Discovery Starter Pipeline – main.py

Purpose
-------
First runnable file to bootstrap an eDiscovery/forensic analytics project.
It ingests email data (.eml or .mbox), normalizes metadata + body text,
performs lightweight near-duplicate detection, builds a communication graph,
and writes tidy outputs for downstream NLP/SNA/ML work.

Design goals
------------
- Minimal external deps; graceful fallbacks if optional libs are missing
- Clear, testable functions with type hints
- CLI-first: `python main.py --help`

Suggested repo layout
---------------------
project_root/
  ├─ data/
  │   ├─ raw/               # place .eml/.mbox here
  │   └─ processed/         # outputs land here
  ├─ graphs/                # network exports
  ├─ reports/               # JSON summaries
  ├─ src/
  │   └─ main.py            # <— you are here in an initial single-file form
  └─ requirements.txt

Note: PST support typically needs libpff/pypff; omitted here for a smooth start.
You can add it later in a dedicated ingestor module.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import email
import email.policy
import html
import json
import logging
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Optional dependencies (used if available)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    from sklearn.ensemble import IsolationForest  # type: ignore
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    IsolationForest = None  # type: ignore


# -----------------------------
# Data model
# -----------------------------
@dataclasses.dataclass
class EmailRecord:
    doc_id: str
    path: str
    date: Optional[dt.datetime]
    sender: str
    to: List[str]
    cc: List[str]
    bcc: List[str]
    subject: str
    body_text: str
    in_reply_to: Optional[str] = None
    references: List[str] = dataclasses.field(default_factory=list)
    thread_key: Optional[str] = None  # simple subject-thread key


# -----------------------------
# Utilities
# -----------------------------
EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TAG_RX = re.compile(r"<[^>]+>")
WS_RX = re.compile(r"\s+")
SUBJECT_PREFIX_RX = re.compile(r"^(re:|fw:|fwd:)\s*", re.I)


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = TAG_RX.sub(" ", s)  # strip basic HTML tags
    s = s.replace("\r", "\n")
    s = WS_RX.sub(" ", s)
    return s.strip()


def norm_subject(subject: str) -> str:
    s = subject or ""
    while True:
        new = SUBJECT_PREFIX_RX.sub("", s).strip()
        if new == s:
            break
        s = new
    return s.lower()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Ingestion
# -----------------------------

def discover_files(input_path: Path) -> List[Path]:
    exts = {".eml", ".mbox"}
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in exts else []
    files: List[Path] = []
    for root, _, fnames in os.walk(input_path):
        for fn in fnames:
            p = Path(root) / fn
            if p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)


def parse_eml(path: Path, *, doc_prefix: str = "eml") -> EmailRecord:
    with path.open("rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    def get_payload_text(m: email.message.Message) -> str:
        if m.is_multipart():
            parts: List[str] = []
            for part in m.walk():
                ctype = part.get_content_type()
                if ctype in ("text/plain", "text/html"):
                    try:
                        parts.append(part.get_content().strip())
                    except Exception:
                        try:
                            parts.append(part.get_payload(decode=True).decode(errors="ignore"))
                        except Exception:
                            continue
            return "\n\n".join(parts)
        try:
            return m.get_content()
        except Exception:
            try:
                return m.get_payload(decode=True).decode(errors="ignore")
            except Exception:
                return ""

    payload = get_payload_text(msg)

    date_hdr = msg.get("date")
    parsed_date: Optional[dt.datetime] = None
    if date_hdr:
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_hdr)
        except Exception:
            parsed_date = None

    def split_addrs(v: Optional[str]) -> List[str]:
        if not v:
            return []
        return [a.strip() for a in EMAIL_RX.findall(v)]

    subject = msg.get("subject", "") or ""
    record = EmailRecord(
        doc_id=f"{doc_prefix}:{hash(path.as_posix()) & 0xFFFFFFFF:X}",
        path=str(path),
        date=parsed_date,
        sender=(EMAIL_RX.findall(msg.get("from", "")) or [""])[0],
        to=split_addrs(msg.get("to")),
        cc=split_addrs(msg.get("cc")),
        bcc=split_addrs(msg.get("bcc")),
        subject=subject,
        body_text=clean_text(payload),
        in_reply_to=msg.get("in-reply-to"),
        references=[r.strip() for r in (msg.get("references", "").split()) if r.strip()],
        thread_key=norm_subject(subject),
    )
    return record


def parse_mbox(path: Path) -> Iterator[EmailRecord]:
    import mailbox  # stdlib

    mbox = mailbox.mbox(path)
    for i, msg in enumerate(mbox):
        try:
            tmp_path = path.with_suffix("") / f"{path.stem}_{i}.eml"
            # Build an EmailRecord via the same logic as .eml for consistency
            yield parse_eml_from_message(msg, source_path=str(tmp_path))
        except Exception as exc:
            logging.warning("Failed to parse message %s #%s: %s", path.name, i, exc)


def parse_eml_from_message(msg: email.message.Message, *, source_path: str) -> EmailRecord:
    def get_payload_text(m: email.message.Message) -> str:
        if m.is_multipart():
            parts: List[str] = []
            for part in m.walk():
                ctype = part.get_content_type()
                if ctype in ("text/plain", "text/html"):
                    try:
                        parts.append(part.get_content().strip())
                    except Exception:
                        try:
                            parts.append(part.get_payload(decode=True).decode(errors="ignore"))
                        except Exception:
                            continue
            return "\n\n".join(parts)
        try:
            return m.get_content()
        except Exception:
            try:
                return m.get_payload(decode=True).decode(errors="ignore")
            except Exception:
                return ""

    payload = get_payload_text(msg)

    date_hdr = msg.get("date")
    parsed_date: Optional[dt.datetime] = None
    if date_hdr:
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_hdr)
        except Exception:
            parsed_date = None

    def split_addrs(v: Optional[str]) -> List[str]:
        if not v:
            return []
        return [a.strip() for a in EMAIL_RX.findall(v)]

    subject = msg.get("subject", "") or ""
    return EmailRecord(
        doc_id=f"mbox:{hash(source_path) & 0xFFFFFFFF:X}",
        path=source_path,
        date=parsed_date,
        sender=(EMAIL_RX.findall(msg.get("from", "")) or [""])[0],
        to=split_addrs(msg.get("to")),
        cc=split_addrs(msg.get("cc")),
        bcc=split_addrs(msg.get("bcc")),
        subject=subject,
        body_text=clean_text(payload),
        in_reply_to=msg.get("in-reply-to"),
        references=[r.strip() for r in (msg.get("references", "").split()) if r.strip()],
        thread_key=norm_subject(subject),
    )


# -----------------------------
# Near-duplicate detection
# -----------------------------

def compute_near_duplicates(texts: List[str], min_sim: float = 0.9) -> Dict[int, List[int]]:
    """Return mapping: i -> list of j indices that are near-duplicates of i (j>i).
    Uses TF-IDF cosine similarity if sklearn is available, else a SequenceMatcher fallback.
    """
    n = len(texts)
    dup_map: Dict[int, List[int]] = defaultdict(list)
    if n == 0:
        return dup_map

    if TfidfVectorizer and cosine_similarity:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf = vec.fit_transform(texts)
        # Block-wise similarity to save memory on large corpora (simple approach here)
        sim = cosine_similarity(tfidf)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= min_sim:
                    dup_map[i].append(j)
        return dup_map

    # Fallback: difflib (slower, rougher)
    import difflib

    for i in range(n):
        for j in range(i + 1, n):
            r = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
            if r >= min_sim:
                dup_map[i].append(j)
    return dup_map


# -----------------------------
# Entities (lightweight baseline)
# -----------------------------
ENTITY_PATTERNS = {
    "email": EMAIL_RX,
    "url": re.compile(r"https?://\S+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d\s\-()]{6,}\d)"),
}


def extract_simple_entities(text: str) -> Dict[str, List[str]]:
    ents: Dict[str, List[str]] = {}
    for k, rx in ENTITY_PATTERNS.items():
        vals = list({m.group(0) for m in rx.finditer(text)})
        if vals:
            ents[k] = sorted(vals)
    return ents


# -----------------------------
# Graph & anomaly basics
# -----------------------------

def build_comm_edges(records: List[EmailRecord]) -> Counter[Tuple[str, str]]:
    edges: Counter[Tuple[str, str]] = Counter()
    for r in records:
        sender = r.sender.lower().strip()
        if not sender:
            continue
        recipients = set([*(a.lower() for a in r.to), *(a.lower() for a in r.cc), *(a.lower() for a in r.bcc)])
        for rcpt in recipients:
            if rcpt:
                edges[(sender, rcpt)] += 1
    return edges


def detect_volume_anomalies(records: List[EmailRecord]) -> List[Tuple[dt.date, int, float]]:
    """Daily volume anomaly detection (simple baseline).
    Returns list of (date, count, anomaly_score) for flagged days.
    Uses IsolationForest if available; else z-score > 3 heuristic.
    """
    by_day: Dict[dt.date, int] = defaultdict(int)
    for r in records:
        if r.date:
            by_day[r.date.date()] += 1
    days = sorted(by_day.keys())
    counts = [by_day[d] for d in days]
    if not days:
        return []

    if IsolationForest:
        import numpy as np

        X = np.array(counts).reshape(-1, 1)
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        scores = -model.score_samples(X)  # higher means more anomalous here
        thresh = sorted(scores)[max(0, int(0.95 * len(scores)) - 1)]  # top 5% as a rough cut
        return [(d, c, float(s)) for d, c, s in zip(days, counts, scores) if s >= thresh]

    # Fallback: z-score heuristic
    import statistics as stats

    mu = stats.mean(counts)
    sd = stats.pstdev(counts) or 1.0
    anomalies: List[Tuple[dt.date, int, float]] = []
    for d, c in zip(days, counts):
        z = (c - mu) / sd
        if z >= 3.0:
            anomalies.append((d, c, float(z)))
    return anomalies


# -----------------------------
# Serialization helpers
# -----------------------------

def records_to_rows(records: List[EmailRecord]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for r in records:
        rows.append({
            "doc_id": r.doc_id,
            "path": r.path,
            "date": r.date.isoformat() if r.date else None,
            "sender": r.sender,
            "to": ";".join(r.to),
            "cc": ";".join(r.cc),
            "bcc": ";".join(r.bcc),
            "subject": r.subject,
            "thread_key": r.thread_key or "",
            "body_text": r.body_text,
        })
    return rows


def save_table(rows: List[Dict[str, object]], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
    else:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def save_edges(edges: Counter[Tuple[str, str]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if nx is not None:
        G = nx.DiGraph()
        for (u, v), w in edges.items():
            G.add_edge(u, v, weight=int(w))
        nx.write_gexf(G, out_path.with_suffix(".gexf"))
    # Always also write a CSV edge list for universal consumption
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight"]) 
        for (u, v), weight in edges.items():
            w.writerow([u, v, int(weight)])


def save_report(summary: Dict[str, object], out_json: Path) -> None:
    ensure_dir(out_json.parent)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline(input_path: Path, out_dir: Path, min_sim: float = 0.9) -> Dict[str, object]:
    files = discover_files(input_path)
    if not files:
        raise FileNotFoundError(f"No .eml/.mbox files found under: {input_path}")

    records: List[EmailRecord] = []
    for p in files:
        if p.suffix.lower() == ".eml":
            try:
                records.append(parse_eml(p))
            except Exception as exc:
                logging.warning("Failed to parse %s: %s", p, exc)
        elif p.suffix.lower() == ".mbox":
            for rec in parse_mbox(p):
                records.append(rec)

    # Near-duplicate clustering (simple pass)
    texts = [r.body_text for r in records]
    dup_map = compute_near_duplicates(texts, min_sim=min_sim)
    # Collapse duplicates: keep first occurrence, mark duplicates
    keep_mask = [True] * len(records)
    dup_of: Dict[str, str] = {}
    for i, js in dup_map.items():
        for j in js:
            keep_mask[j] = False
            dup_of[records[j].doc_id] = records[i].doc_id

    deduped = [r for r, keep in zip(records, keep_mask) if keep]

    # Entities (baseline)
    entity_counts: Counter[str] = Counter()
    for r in deduped:
        ents = extract_simple_entities(r.body_text)
        for k, vals in ents.items():
            entity_counts[k] += len(vals)

    # Communication edges + anomalies
    edges = build_comm_edges(deduped)
    anomalies = detect_volume_anomalies(deduped)

    # Write outputs
    rows = records_to_rows(deduped)
    save_table(rows, out_dir / "data" / "processed" / "emails.csv")
    save_edges(edges, out_dir / "graphs" / "comm_graph")
    report = {
        "input_path": str(input_path),
        "n_files": len(files),
        "n_records": len(records),
        "n_records_after_dedup": len(deduped),
        "duplicate_links": dup_of,  # doc_id -> canonical doc_id
        "entity_counts": dict(entity_counts),
        "n_edges": int(sum(edges.values())),
        "n_nodes": len({u for u, _ in edges.keys()} | {v for _, v in edges.keys()}),
        "anomalies": [(d.isoformat(), c, score) for (d, c, score) in anomalies],
    }
    save_report(report, out_dir / "reports" / "summary.json")

    return report


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="edisco-starter",
        description="Ingest .eml/.mbox, normalize, de-duplicate, and build a comms graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples
            --------
            # Basic run (input dir of emails, outputs under ./out)
            python main.py -i ./data/raw -o ./

            # Stricter duplicate threshold
            python main.py -i ./data/raw -o ./ --min-sim 0.95
            """
        ),
    )
    p.add_argument("-i", "--input", required=True, type=Path, help="Path to .eml/.mbox file or directory")
    p.add_argument("-o", "--outdir", required=False, type=Path, default=Path("."), help="Project root for outputs")
    p.add_argument("--min-sim", type=float, default=0.90, help="Near-duplicate similarity threshold (0..1)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v or -vv for more logs")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(levelname)s %(message)s")

    try:
        summary = run_pipeline(args.input, args.outdir, min_sim=args.min_sim)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
