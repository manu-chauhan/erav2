import regex as re

import utilities
from .base import Tokenizer, get_stats, merge

# from .BasicTokenizer import BasicTokenizer

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

"""
Basic Devanagari: \u0900 to \u097F
Vedic Extensions: \u1CD0 to \u1CFF
Extended Devanagari: \uA8E0 to \uA8FF
"""
# ignore case in compile below
SIMPLE_HINDI_PATTERN = r"""[\t\n\r\f\v]?|[^\r\n\p{Devanagari}\p{N}]?+\p{Devanagari}+|\\p{N}{1,}| ?[^\s\p{Devanagari}+\p{N}]++[\r\n]*|\s*[\r\n]*|\s+(?!\S)|\s+"""
EXTENDED_HINDI_PATTERN = r"""[\t\n\r\f\v]?|[^\r\n\p{Devanagari}\uA8E0-\uA8FF\u1CD0-\u1CFF\p{N}]?+[\p{Devanagari}\uA8E0-\uA8FF\u1CD0-\u1CFF]+|\p{N}{1,}| ?[^\s\p{Devanagari}+\p{N}\uA8E0-\uA8FF\u1CD0-\u1CFF]++[\r\n]*|\s*[\r\n]*|\s+(?!\S)|\s+"""


class HindiTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = SIMPLE_HINDI_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE, re.UNICODE)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.merges = None
        self.vocab = None

    @utilities.log_to_file("HindiTokenizer-train.log")
    def train(self, text, vocab_size, verbose=False):
        print("\nTraining...for HindiTokenizer")
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks if len(ch) > 1]

        # iteratively merge the MOST COMMON pair from the text
        # use same merge dict if exists
        merges = {} if self.vocab is None else self.merges  # to hold all merges (int, int) -> int

        # Use same vocab for this Tokenizer object if it exists
        # Tokenizer vocab:  int -> bytes
        vocab = {idx: bytes([idx]) for idx in range(256)} if self.vocab is None else self.vocab

        # run merging iteratively
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    @utilities.log_to_file("HindiTokenizer-decode.log")
    def decode(self, ids):
        print("\nDecoding...for HindiTokenizer")
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    @utilities.log_to_file("HindiTokenizer-encode.log")
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids


if __name__ == "__main__":
    custom_text = """
    <|endoftext|>ूज रहा है जहाँ चकित हो जन-जन देख अकाज
सात वर्ष हो गये राह में, अटका कहाँ स्वराज?

अटका कहाँ स्वराज? बोल दिल्ली! तू क्या कहती है?
तू रानी बन गयी वेदना जनता क्यों सहती है?
सबके भाग्य दबा रखे हैं किसने अपने कर में?
उतरी थी जो विभा, हुई बंदिनी बता किस घर में

समर शेष है, यह प्रकाश बंदीगृह से छूटेगा
और नहीं तो तुझ पर पापिनी! महावज्र टूटेगा

समर शेष है, उस स्वराज को सत्य बनाना होगा
जिसका है ये न्यास उसे सत्वर पहुँचाना होगा
धारा के मग में अनेक जो पर्वत खडे हुए हैं
गंगा का पथ रोक इन्द्र के गज जो अडे हुए हैं

कह दो उनसे झुके अगर तो जग मे यश पाएंगे
अड़े रहे अगर तो ऐरावत पत्तों से बह जाऐंगे<|fim_prefix|><|endofprompt|>
    """.strip()
    special_tokens = {
        '<|endoftext|>': 100257,
        '<|fim_prefix|>': 100258,
        '<|fim_middle|>': 100259,
        '<|fim_suffix|>': 100260,
        '<|endofprompt|>': 100276
    }
    text = custom_text
    # create a Tokenizer and do 64 merges
    tokenizer = HindiTokenizer()
    tokenizer.train(text, 256 + 2, verbose=True)
    tokenizer.register_special_tokens(special_tokens)
    # verify that decode(encode(x)) == x
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
