import random
from collections.abc import Generator
from pathlib import Path

import mlx.core as mx
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODEL = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
THINKING = False
THINKING_LEVEL = "low"
MAX_TOKENS = 8192
TEMPERATURE = 0.9
TOP_P = 0.95

SYSTEM_PROMPT = "You job is to help recover a lost passphrase. You do so by writing short English sentences in plain ASCII, one per line, with no numbering, quotes or commentary."

USER_PROMPT = """In 1997 two NSA mathematicians at Fort Meade, Jerry Solinas and Bob Reiter, generated the seeds for the NIST elliptic curves. Bob wrote the program, which hashed the ASCII text of a humorous message with SHA-1 and added a counter until the curve came out right. Both of them later forgot the exact message.

  What people remember about the message:
  - Jerry, in a 2015 email: "The message was along the lines of 'Give Bob and Jerry a raise' or 'Bob and Jerry rule' or something like that."
  - Someone who knew Jerry remembered it as something like "Jerry will get a raise".
  - A friend remembered Jerry saying it was something like "Jerry deserves a raise."
  - Another person remembered it having two names in it, like "Give Alice and Bob a raise."
  Jerry tried every phrase he could think of along these lines and none matched.

  About the two men: Jerome A. "Jerry" Solinas (1957-2023) had a math PhD from the University of Michigan and worked on elliptic curves at the NSA for decades. Robert W. "Bob" Reiter was his collaborator; they shared an NSA patent from 1998. Jerry had a dry, self-deprecating sense of humor: he signed an email "Greetings from evil Jerry", joked that he could "cross 'become an evil Internet meme' off my bucket list", and said the lost files were "sitting on a hard drive near the Ark of the Covenant". Other NSA colleagues of his in the 1990s included Laurie Law and Susan Sabett.

  Write as many new candidates for the lost message as you can, one per line. Picture Jerry or Bob typing a quick joke into a program in 1997. Vary:
  - The names: Bob and Jerry, Jerry and Bob, Bob & Jerry, Jerry Solinas, Solinas, Jerome, Dr. Solinas, Bob Reiter, Robert Reiter, Reiter, Dr. Reiter, and me, us, we or Bob and I, as if one of them were writing. Most lines should name both Bob and Jerry, some only one of them, and a few could add a third colleague.
  - The joke: a raise, a big raise, raises, a bonus, a promotion, more money, a pay increase; or bragging that they rule, rock, are geniuses, are the best, kick butt.
  - The grammar: orders (Give Bob and Jerry a raise), claims (Bob and Jerry deserve a raise), predictions (Jerry will get a raise), requests (Please give Jerry a raise), and who should pay up (the NSA, the agency, the boss, Uncle Sam, the government).
  - The amount: the program automatically appends a number after each line, including as a dollar amount like $1000, so also write lines that stop exactly where an amount would go, such as "Give Bob and Jerry a raise of", "Pay Bob and Jerry", "Bob and Jerry are worth" or "Give Bob and Jerry a bonus of". Never type the number yourself.
  Mix these freely, and also try other short, natural ways a person might phrase the same joke.

  Rules:
  - Every line must be a joke message about Jerry and/or Bob. Never drift into other topics, even after hundreds of lines.
  - Keep lines short, like something typed as a constant in a program: usually 3 to 10 words.
  - Plain ASCII only: straight apostrophes ('), no curly quotes, dashes, emojis or underscores.
  - Use only words and slang that were common in 1997, and spell every word correctly.
  - Do not end lines with punctuation and do not write any number yourself. The program automatically tries every ending, capitalization, and counter, including plain numbers and dollar amounts, at the end of each line, so never add one and never repeat a line with only its number, capitalization or punctuation changed.
  - No numbering, quotes, explanations or headings. Just the messages.
  - Think about the question a bit but then, output the phrases in the response channel not the thinking channel.
"""


class PhraseGenerator:
    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")  # HF_TOKEN for faster Hugging Face downloads
        mx.random.seed(random.randint(0, 2**32 - 1))
        model_path = snapshot_download(
            MODEL,
            cache_dir=PROJECT_ROOT / ".hf-cache",
        )
        self.model, self.tokenizer = load(model_path)[:2]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
        self.prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=THINKING,
            thinking_level=THINKING_LEVEL,
            return_dict=False
        )

    def stream_phrases(self) -> Generator[str]:
        """Yield phrases forever, starting a new generation whenever the model stops."""
        while True:
            yield from self._stream_once()

    def _stream_once(self) -> Generator[str]:
        sentence_buffer = ""
        thinking_phase = THINKING
        if thinking_phase:
            print("\033[96m", end="")
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            self.prompt,
            MAX_TOKENS,
            sampler=make_sampler(temp=TEMPERATURE, top_p=TOP_P),
        ):
            if not thinking_phase:
                # yield sentences to make it easier to process them later on
                sentence_buffer += resp.text
                sentences = sentence_buffer.split("\n")
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        # the model sometimes separates lines with blank ones
                        if sentence.strip():
                            yield sentence.strip()
                    sentence_buffer = sentences[-1]
            else:
                if resp.token == self.tokenizer.think_end_id:
                    thinking_phase = False
                    print(
                        f"\n\033[92mThinking complete with {round(resp.generation_tps, 2)} tps, {round(resp.peak_memory, 2)} GB peak memory.\033[0m",
                        flush=True,
                    )
                    continue
                print(resp.text, end="", flush=True)
        # the model may stop without a trailing newline, so flush what is left
        if sentence_buffer.strip():
            yield sentence_buffer.strip()
