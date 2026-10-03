import time
from collections.abc import Iterable, Iterator
from hashlib import md5, sha1
from itertools import batched
from multiprocessing import Pool
from pathlib import Path

HASHES_FILE = Path(__file__).resolve().parent / "hashes.txt"
ALGOS = {"sha1": sha1, "md5": md5}
# How the first digest is fed into the second hash: raw bytes, or as lower/upper case hex text
FORMS = {
    "raw": lambda d: d,
    "hex": lambda d: d.hex().encode(),
    "HEX": lambda d: d.hex().upper().encode(),
}
BATCH = 32_000  # candidates per task sent to a worker

# For each inner hash: its name, and every way to chain an outer hash after it. Names are only built once, here.
CHAINS = [
    (
        inner,
        inner_hash,
        [
            (f"{outer}({inner}, {form})", fn, outer_hash)
            for form, fn in FORMS.items()
            for outer, outer_hash in ALGOS.items()
        ],
    )
    for inner, inner_hash in ALGOS.items()
]
HASHES_PER_CANDIDATE = sum(1 + len(chained) for _, _, chained in CHAINS)

_targets: frozenset[bytes] = frozenset()


def _init_worker(targets: frozenset[bytes]) -> None:
    global _targets
    _targets = targets


def _search(batch: tuple[str, ...]) -> tuple[int, list[tuple[str, str, str]]]:
    """Hash a batch of candidates in plain sha1/md5 and chained outer(inner(x)); return the matches."""
    hits = []
    for candidate in batch:
        data = candidate.encode(
            "ascii"
        )  # strict: fail loudly rather than hash something else
        for name, inner, chained in CHAINS:
            first = inner(data).digest()
            if first in _targets:
                hits.append((candidate, name, first.hex().upper()))
            for chain_name, form, outer in chained:
                digest = outer(form(first)).digest()
                if digest in _targets:
                    hits.append((candidate, chain_name, digest.hex().upper()))
    return len(batch), hits


class Hasher:
    def __init__(self, hashes_file: Path = HASHES_FILE) -> None:
        # hashes.txt holds the 5 NIST seeds plus their decrements, as uppercase hex, one per line
        seeds = {bytes.fromhex(line) for line in hashes_file.read_text().split()}
        # MD5 is 16 bytes, a seed is 20: also match 16-byte digests against the first or last 16 bytes of a seed
        targets = frozenset(seeds | {s[:16] for s in seeds} | {s[-16:] for s in seeds})
        self.pool = Pool(
            initializer=_init_worker, initargs=(targets,)
        )  # one worker per core

    def hash(self, to_test: Iterable[str]) -> Iterator[tuple[str, str, str]]:
        """Yield (candidate, algorithm, hex digest) for every candidate that matches hashes.txt."""
        start = last_print = time.monotonic()
        count = 0
        for n, hits in self.pool.imap_unordered(_search, batched(to_test, BATCH)):
            yield from hits
            count += n
            if (now := time.monotonic()) - last_print >= 1:
                last_print = now
                print(self._rate(count, now - start), end="\r", flush=True)
        print(self._rate(count, time.monotonic() - start))

    @staticmethod
    def _rate(count: int, seconds: float) -> str:
        rate = count / max(seconds, 1e-9)
        return f"{count:,} candidates, {rate:,.0f} candidates/s, {rate * HASHES_PER_CANDIDATE:,.0f} hashes/s"
