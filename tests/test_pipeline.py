"""Run with: uv run python -m unittest. Targets are planted in temporary files, hashes.txt is never touched."""

import contextlib
import io
import tempfile
import unittest
from hashlib import md5, sha1
from pathlib import Path

from src.fuzzer import Fuzzer
from src.hasher import HASHES_FILE, Hasher

BOUNTY_SEEDS = {
    "3045AE6FC8422F64ED579528D38120EAE12196D5",
    "BD71344799D5C7FCDC45B59FA3B9AB8F6A948BC5",
    "C49D360886E704936A6678E1139D26B7819F7E90",
    "A335926AA319A27A1D00896A6773A4827ACDAC73",
    "D09E8800291CB85396CC6717393284AAA0DA64BA",
}
PHRASE = "Jerry deserves a raise"


def b(text: str) -> bytes:
    return text.encode()


class WithHasher(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            contextlib.redirect_stdout(io.StringIO())
        )  # hide the hashes/s lines

    def hasher(self, *targets: bytes | str) -> Hasher:
        """A Hasher whose targets file holds only the given digests (bytes or hex)."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        file = Path(tmp.name) / "hashes.txt"
        file.write_text(
            "\n".join(t.hex() if isinstance(t, bytes) else t for t in targets)
        )
        hasher = Hasher(file)
        self.addCleanup(hasher.pool.terminate)
        return hasher


class HasherTest(WithHasher):
    def test_plain_sha1(self):
        h = self.hasher(sha1(b(PHRASE)).hexdigest())
        self.assertEqual(
            list(h.hash(["nope", PHRASE])),
            [(PHRASE, "sha1", sha1(b(PHRASE)).hexdigest().upper())],
        )

    def test_md5_matches_first_16_bytes_of_a_target(self):
        h = self.hasher(md5(b(PHRASE)).digest() + bytes(4))
        self.assertEqual([m[:2] for m in h.hash([PHRASE])], [(PHRASE, "md5")])

    def test_md5_matches_last_16_bytes_of_a_target(self):
        h = self.hasher(bytes(4) + md5(b(PHRASE)).digest())
        self.assertEqual([m[:2] for m in h.hash([PHRASE])], [(PHRASE, "md5")])

    def test_sha1_of_md5_hex(self):
        target = sha1(md5(b(PHRASE)).hexdigest().encode()).digest()
        self.assertEqual(
            [m[:2] for m in self.hasher(target).hash([PHRASE])],
            [(PHRASE, "sha1(md5, hex)")],
        )

    def test_sha1_of_sha1_raw(self):
        target = sha1(sha1(b(PHRASE)).digest()).digest()
        self.assertEqual(
            [m[:2] for m in self.hasher(target).hash([PHRASE])],
            [(PHRASE, "sha1(sha1, raw)")],
        )

    def test_md5_of_sha1_upper_hex(self):
        target = md5(sha1(b(PHRASE)).hexdigest().upper().encode()).digest() + bytes(4)
        self.assertEqual(
            [m[:2] for m in self.hasher(target).hash([PHRASE])],
            [(PHRASE, "md5(sha1, HEX)")],
        )

    def test_no_false_positives(self):
        h = self.hasher(sha1(b"something else").hexdigest())
        self.assertEqual(list(h.hash([PHRASE, "x", ""])), [])

    def test_non_ascii_fails_loudly(self):
        with self.assertRaises(UnicodeEncodeError):
            list(self.hasher(bytes(20)).hash(["caf\u00e9"]))

    def test_matches_across_many_batches(self):
        h = self.hasher(sha1(b"phrase 123456").hexdigest())
        candidates = (
            f"phrase {i}" for i in range(200_000)
        )  # several batches, several workers
        self.assertEqual([m[0] for m in h.hash(candidates)], ["phrase 123456"])


class FuzzerTest(unittest.TestCase):
    def test_case_variants_are_unique(self):
        variants = list(
            Fuzzer().get_case_variants(PHRASE)
        )  # capitalize() == the input here
        self.assertEqual(len(variants), len(set(variants)))
        self.assertEqual(variants.count(PHRASE), 1)

    def test_candidates_are_unique(self):
        candidates = list(Fuzzer().fuzz(PHRASE))
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_expected_variants_are_generated(self):
        candidates = set(Fuzzer().fuzz("Give Bob and Jerry a raise"))
        for expected in (
            "Give Bob and Jerry a raise",
            "GIVE BOB AND JERRY A RAISE.",
            "give bob and jerry a raise of $1999!",
            "Give Bob And Jerry A Raise Worth 7 Bucks...",
            "Give Bob and Jerry a raise 2499 dollars?",
        ):
            self.assertIn(expected, candidates)


class EndToEndTest(WithHasher):
    def test_fuzzed_variant_is_found(self):
        wanted = "Give Bob And Jerry A Raise Of 7 Bucks."  # case variant of a number + punctuation variant
        h = self.hasher(sha1(b(wanted)).hexdigest())
        self.assertEqual(
            [m[0] for m in h.hash(Fuzzer().fuzz("Give Bob and Jerry a raise"))],
            [wanted],
        )


class HashesFileTest(unittest.TestCase):
    def test_contains_the_five_bounty_seeds(self):
        self.assertLessEqual(BOUNTY_SEEDS, set(HASHES_FILE.read_text().split()))

    def test_every_line_is_a_sha1_hex_digest(self):
        for line in HASHES_FILE.read_text().split():
            self.assertEqual(len(bytes.fromhex(line)), 20, line)


if __name__ == "__main__":
    unittest.main()
