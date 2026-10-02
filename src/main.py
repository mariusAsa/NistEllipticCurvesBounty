from .fuzzer import Fuzzer
from .phrase_generator import PhraseGenerator


def main() -> None:
    llm = PhraseGenerator()
    fuzzer = Fuzzer()
    for phrase in llm.stream_phrases():
        variants = f"{round(fuzzer.count_variants(phrase)/1000000, 2)}"
        print(f"{variants}M variants of {phrase}")


if __name__ == "__main__":
    main()
