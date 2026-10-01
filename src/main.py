from .phrase_generator import PROJECT_ROOT, PhraseGenerator

OUTPUT_PATH = PROJECT_ROOT / "phrases" / "generated.txt"


def generate_phrases():
    llm = PhraseGenerator()
    thinking = True
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        print("\033[94m", end="", flush=True)  # Blue
        for resp in llm.stream():
            if resp.token == llm.tokenizer.think_end_id:
                thinking = False
                print(
                    "\n"
                    "\033[92m"  # Green
                    f"Thinking complete with {round(resp.generation_tps, 2)} tps, {round(resp.peak_memory, 2)} GB peak memory"
                    "\033[0m"
                )
                continue
            if thinking:
                print(resp.text, end="", flush=True)
            else:
                output_file.write(resp.text)
                output_file.flush()
                print(resp.text, end="", flush=True)
    print()


if __name__ == "__main__":
    generate_phrases()
