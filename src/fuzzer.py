from collections.abc import Generator, Iterable
from itertools import chain
from string import capwords

# Raises are usually round numbers, so also try a few big ones. No negatives: "a raise of -5" makes no sense.
NUMBERS = tuple(
    chain(
        range(2500),
        (5000, 10000, 15000, 20000, 25000, 50000, 100000, 1000000, 1000000000),
    )
)
# Currency signs go before the number, words after it. Only 1997-era ASCII: no euros (introduced in 1999), no € or £.
AMOUNT_FORMATS = ("{}", "${}", "{} dollars", "{} bucks")
# "of" / "worth" only make sense after one of these nouns, e.g. "a raise of 5", but not "Pay Bob and Jerry of 5"
CONNECTOR_NOUNS = ("raise", "raises", "bonus", "promotion", "increase", "money", "payout", "payday")
CONNECTORS = ("", "of ", "worth ")  # "" means the amount directly follows the phrase
PUNCTUATION = (".", "!", "?", "...")


class Fuzzer:
    def __init__(self):
        pass

    # Unique case variants, so e.g. an already capitalized input is not hashed twice.
    # count_variants() is therefore an upper bound.
    # capwords instead of str.title(), which breaks apostrophes ("Don'T") and digits ("5Dollars")
    def get_case_variants(self, input: str) -> Iterable[str]:
        return dict.fromkeys(
            (input, input.upper(), input.lower(), capwords(input), input.capitalize())
        )

    def append_numbers(self, input: str) -> Generator[str]:
        # If the LLM outputs Jerry deserves a raise, we also want to try Jerry deserves a raise of x
        # A phrase that already ends in "of" / "worth" gets neither connector
        connectors = CONNECTORS if input.lower().endswith(CONNECTOR_NOUNS) else ("",)
        for connector in connectors:
            for amount in AMOUNT_FORMATS:
                for i in NUMBERS:
                    yield f"{input} {connector}{amount.format(i)}"

    def append_punctuation(self, input: str) -> Generator[str]:
        for p in PUNCTUATION:
            yield f"{input}{p}"

    # The string itself followed by every punctuation variant of it
    def with_punctuation(self, input: str) -> Generator[str]:
        yield input
        yield from self.append_punctuation(input)

    # The phrase on its own, then with every amount appended, each with punctuation
    def build_candidates(self, input: str) -> Generator[str]:
        yield from self.with_punctuation(input)
        for numbered in self.append_numbers(input):
            yield from self.with_punctuation(numbered)

    # Hashing is cheap, so we want to test different variants of the input phrase
    def fuzz(self, input: str) -> Generator[str]:
        for candidate in self.build_candidates(input):
            yield from self.get_case_variants(candidate)
