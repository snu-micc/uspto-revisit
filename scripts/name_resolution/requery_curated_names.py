"""Query automatic chemical-name resolvers and capture every response (optional).

The default spellings are the names covered by the 46 curated overrides in
the current predictions. --names ai-baseline selects the Ai et al. predictions.
The queries themselves never apply overrides, row-specific decisions, fixed
name aliases or a shared cache. First-pass OPSIN and the automatic retry chain
are run as in analysis/name_resolution_replay.py. Every HTTP attempt is saved
with UTC time, URL, status/body or error, plus a capture file with a raw-file
SHA-256. --dry-run makes no network call. --live requires an explicit new
--output path and uses the documented per-host limits and retry settings.

This optional query is not needed to score the frozen published predictions.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for path in (REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.curation_ablation import (  # noqa: E402
    DEFAULT_OVERRIDES,
    DEFAULT_RESULTS_DIR,
    load_override_keys,
    prediction_names,
    normalize_name_key,
    read_model_output,
)
from analysis.evaluate_table3 import MODELS  # noqa: E402
from analysis.name_resolution_replay import RecordedResolver, capture_path, clean_name  # noqa: E402


MIN_INTERVAL = {  # seconds between requests to one host
    "pubchem.ncbi.nlm.nih.gov": 0.25,  # at most 4 requests per second (PubChem allows 5)
    "opsin.ch.cam.ac.uk": 0.2,
    "cactus.nci.nih.gov": 1.0,
    "www.ebi.ac.uk": 0.2,
}
ENDPOINTS = {
    "PubChem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/SMILES/JSON",
    "OPSIN": "https://opsin.ch.cam.ac.uk/opsin/{name}.smi",
    "CIR": "http://cactus.nci.nih.gov/chemical/structure/{name}/smiles",
    "ChEBI": "https://www.ebi.ac.uk/chebi/backend/api/public/es_search?term={name}&page=1&size=3 "
             "and https://www.ebi.ac.uk/chebi/backend/api/public/compound/{id}/",
}
TIMEOUT_SECONDS = 120


def curated_spellings(results_dir: Path, overrides: Path) -> list[str]:
    """Extract only spellings covered by the 46 current name overrides."""
    outputs = {model: read_model_output(results_dir, model) for model in MODELS}
    keys = load_override_keys(overrides)
    return sorted({clean_name(name) for frame in outputs.values()
                   for prediction in frame['prediction']
                   for name in prediction_names(prediction).values()
                   if normalize_name_key(name) in keys})


def ai_baseline_spellings(ai_output: Path) -> list[str]:
    from analysis import prepare_ai_baseline as AI

    rows = AI.current_benchmark_records(ai_output, REPOSITORY_ROOT / AI.CURRENT_INPUT)
    return AI.ai_spellings(rows)


class PlannedRequests(RecordedResolver):
    """No request is answered: the URLs a live run sends before any answer arrives."""

    def __init__(self):
        self.responses = {}
        self.missing_urls = set()
        self.requested = []


def planned_requests(spellings: list[str]) -> dict[str, int]:
    """Distinct URLs per host if every answer were empty: an estimate, since a first-pass hit skips the
    fixed-name query and a ChEBI hit adds detail requests."""
    planner = PlannedRequests()
    planner.resolve(spellings, order="two-pass")
    hosts: dict[str, int] = {}
    for url in planner.missing_urls:
        host = urlparse(url).netloc
        hosts[host] = hosts.get(host, 0) + 1
    return dict(sorted(hosts.items()))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class LiveRecorder(RecordedResolver):
    """Resolve with live HTTP requests and record every exchange."""

    def __init__(self):
        self.responses = {}
        self.missing_urls = set()
        self.requested = []
        self.records: list[dict] = []
        self._last_request: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._session = None

    async def _wait(self, url: str) -> None:
        host = urlparse(url).netloc
        async with self._lock:
            now = time.monotonic()
            ready = self._last_request.get(host, 0.0) + MIN_INTERVAL.get(host, 0.5)
            self._last_request[host] = max(now, ready)
        if ready > now:
            await asyncio.sleep(ready - now)

    async def _fetch(self, session, url, semaphore, max_retries):
        """Same status handling as ``smiles_fetch.fetch_smiles``, recorded."""
        from uspto_revisit import smiles_fetch as sf

        self.requested.append(url)
        if url in self.responses:  # identical request within this run
            record = self.responses[url]
            return record.get("body") if record.get("status") == 200 else None
        for attempt in range(max_retries):
            await self._wait(url)
            record = {"utc": utc_now(), "url": url, "attempt": attempt + 1}
            try:
                async with self._session.get(url) as response:
                    text = await response.text()
                    record.update(status=response.status, body=text)
                    self.records.append(record)
                    self.responses[url] = record
                    if response.status == 200:
                        return text
                    if response.status == 429 or "ServerBusy" in text:
                        await sf.exponential_backoff(attempt)
                        continue
                    if 400 <= response.status < 500:
                        return None
                    if attempt < max_retries - 1:
                        await sf.exponential_backoff(attempt)
            except Exception as exc:  # noqa: BLE001 - recorded, as the pipeline continues
                record.update(status=None, error=f"{type(exc).__name__}: {exc}")
                self.records.append(record)
                self.responses[url] = record
                if attempt < max_retries - 1:
                    await sf.exponential_backoff(attempt)
        return None

    async def _resolve_all(self, sf, names, order):
        import aiohttp
        import certifi
        import ssl

        connector = aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self._session = session
            return await self._resolve_names(sf, names, order)

    async def retry_unanswered(self, interval: float, attempts: int) -> dict:
        """Re-send, one at a time, the URLs whose last attempt got no HTTP answer."""
        import aiohttp
        import certifi
        import ssl

        pending = [url for url, record in self.responses.items() if record.get("status") is None]
        connector = aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
        started = utc_now()
        sent = 0
        async with aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        ) as session:
            for url in pending:
                for attempt in range(1, attempts + 1):
                    await asyncio.sleep(interval)
                    record = {"utc": utc_now(), "url": url, "attempt": attempt, "retry_of_unanswered": True}
                    try:
                        async with session.get(url) as response:
                            text = await response.text()
                            record.update(status=response.status, body=text)
                    except Exception as exc:  # noqa: BLE001
                        record.update(status=None, error=f"{type(exc).__name__}: {exc}")
                    self.records.append(record)
                    self.responses[url] = record
                    sent += 1
                    if record["status"] is not None and record["status"] != 429:
                        break
        return {
            "started_utc": started,
            "finished_utc": utc_now(),
            "interval_seconds": interval,
            "max_attempts_per_url": attempts,
            "urls_retried": len(pending),
            "requests": sent,
            "still_unanswered": sorted(
                url for url in pending if self.responses[url].get("status") is None
            ),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="List the spellings; no network access.")
    mode.add_argument("--live", action="store_true", help="Query the resolvers and record every exchange.")
    parser.add_argument("--output", type=Path, help="Recording to write (JSON lines); required with --live.")
    parser.add_argument("--names", choices=("curated", "ai-baseline"), default="curated",
                        help="Names covered by the 46 overrides (default), or extracted by the Ai et al. model.")
    parser.add_argument("--ai-output", type=Path, default=REPOSITORY_ROOT / "benchmarks" / "ai_et_al" / "output.json",
                        help="ai-baseline: stored Ai et al. predictions on the current benchmark.")
    parser.add_argument("--results-dir", type=Path,
                        help="Model outputs to read the names from. Default: the current curated outputs.")
    parser.add_argument("--overrides", type=Path, default=REPOSITORY_ROOT / DEFAULT_OVERRIDES)
    parser.add_argument("--retry-interval", type=float, default=5.0)
    parser.add_argument("--retry-attempts", type=int, default=6)
    parser.add_argument("--retry-rounds", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # spellings contain Greek letters
    except AttributeError:  # pragma: no cover - non-reconfigurable stream
        pass
    if args.names == "curated":
        spellings = curated_spellings(args.results_dir or REPOSITORY_ROOT / DEFAULT_RESULTS_DIR,
                                      args.overrides)
    else:
        spellings = ai_baseline_spellings(args.ai_output)
    if args.dry_run:
        for spelling in spellings:
            print(spelling)
        print(f"{len(spellings)} spellings would be queried (no request sent).", file=sys.stderr)
        hosts = planned_requests(spellings)
        floor = max((count * MIN_INTERVAL.get(host, 0.5) for host, count in hosts.items()), default=0.0)
        print(f"About {sum(hosts.values())} requests {hosts}; the per-host rate limits alone take "
              f"{floor / 60:.0f} min or more.", file=sys.stderr)
        return 0
    if args.output is None:
        parser.error("--live needs --output")
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new file name.")

    from uspto_revisit import smiles_fetch as sf

    recorder = LiveRecorder()
    started = utc_now()
    results = recorder.resolve(spellings, order="two-pass")
    retries = []
    for _round in range(args.retry_rounds):
        if not any(record.get("status") is None for record in recorder.responses.values()):
            break
        retries.append(asyncio.run(recorder.retry_unanswered(args.retry_interval, args.retry_attempts)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in recorder.records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    statuses: dict[str, int] = {}
    for record in recorder.records:
        key = f"{urlparse(record['url']).netloc} {record.get('status') or record.get('error', '')[:30]}"
        statuses[key] = statuses.get(key, 0) + 1
    capture = {
        "query_started_utc": started,
        "query_finished_utc": utc_now(),
        "resolvers": ["PubChem", "OPSIN", "CIR", "ChEBI"],
        "names_source": args.names,
        "lookup_tables": False,
        "endpoints": ENDPOINTS,
        "min_interval_seconds_per_host": MIN_INTERVAL,
        "timeout_seconds": TIMEOUT_SECONDS,
        "resolver_module_sha256": hashlib.sha256(Path(sf.__file__).read_bytes()).hexdigest(),
        "names": len(spellings),
        "requests": len(recorder.records),
        "responses_by_host_and_status": statuses,
        "retries": retries,
        "resolved": sum(1 for row in results.values() if row["selected_smiles"]),
        "raw_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    capture_path(args.output).write_text(json.dumps(capture, indent=2), encoding="utf-8")
    print(f"Recorded {len(recorder.records)} requests for {len(spellings)} spellings in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
