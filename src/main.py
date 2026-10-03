from pathlib import Path

from .fuzzer import Fuzzer
from .phrase_generator import MODEL, PhraseGenerator
from .hasher import Hasher

PHRASES_DIR = Path(__file__).resolve().parent.parent / "phrases"
# e.g. phrases/gemma-4-e4b-it-OptiQ-4bit_phrases_tested.txt
TESTED_FILE = PHRASES_DIR / f"{MODEL.split('/')[-1]}_phrases_tested.txt"


def main() -> None:
    llm = PhraseGenerator()
    fuzzer = Fuzzer()
    hasher = Hasher()
    PHRASES_DIR.mkdir(exist_ok=True)
    # Phrases from earlier runs (and repeats within this one) are not hashed again
    tested = set(TESTED_FILE.read_text().splitlines()) if TESTED_FILE.exists() else set()
    for phrase in llm.stream_phrases():
        if phrase in tested:
            print("SKIP:", phrase)
            continue
        tested.add(phrase)
        print(phrase)
        for candidate, algorithm, digest in hasher.hash(fuzzer.fuzz(phrase)):
            print("FOUND:", candidate, algorithm, digest)
            with open(PHRASES_DIR / f"{digest}.txt", "w+") as f:
                f.write(f"{candidate}\n{algorithm}\n{digest}\n")
        # Only recorded once every variant of the phrase has been hashed
        with open(TESTED_FILE, "a") as f:
            f.write(f"{phrase}\n")



if __name__ == "__main__":
    main()
