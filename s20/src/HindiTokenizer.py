from textwrap import dedent

import regex as re
import unicodedata

import utilities
from src.base import Tokenizer, get_stats, merge

whitespace = ' \t\n\r\v\f'
ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ascii_letters = ascii_lowercase + ascii_uppercase
digits = '0123456789'
hexdigits = digits + 'abcdef' + 'ABCDEF'
octdigits = '01234567'
punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

ascii_printable = whitespace + ascii_letters + hexdigits + punctuation

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


def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class HindiTokenizer:
    def __init__(self, pattern=None, encoding="utf-8"):
        self.pattern = SIMPLE_HINDI_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE, re.UNICODE)
        self.inverse_special_tokens = {}
        self.merges = None
        self.vocab = None
        self.encoding = encoding
        self.hindi_varnmala_and_key_units = dedent("""
                    अ आ इ ई उ ऊ ए ऐ ओ औ अं अः ऋ ॠ
                    ा ि ी ु ू ृॄ ॅॆ े ैॉ ॊ ो ौ                     
                    क ख ग घ ङ क़ ख़ ग़ घ़ ङ़
                    च छ ज झ ञ ज़ झ़ ञ़
                    ट ठ ड ढ ण ड़ ढ़ ण़
                    त थ द ध न त़ थ़ द़ ध़ ऩ
                    प फ ब भ म प़ फ़ ब़ म़
                    य र ल ळ व य़ ऱ ल़ ऴ व़
                    श ष ॺ स ह श़ ष़ स़ ह़
                    ० १ २ ३ ४ ५ ६ ७ ८ ९ 
                    ॥
                    """)

        super().__init__()

    def _build_vocab(self):
        '''add other important ASCII units except English letters'''

        print("\n====================================\n\n"
              "Building initial Hindi vocabulary with basic Hindi letters and key tokens")
        self.vocab = {}
        ascii_letters_encoded = ascii_letters.encode(
            encoding="utf-8")  # was using this to ignore ASCII English letters, revisit/todo, hindi usage with English or day to day usage and chats may include english letter and what to fill with those blank idxes?
        for idx in range(256):
            self.vocab[idx] = bytes([idx])

        max_idx = max(self.vocab.keys()) + 1

        basic_hindi_alphabet = self.hindi_varnmala_and_key_units.strip().split()

        for idx in range(len(basic_hindi_alphabet)):
            encoded_char = basic_hindi_alphabet[idx].encode(encoding=self.encoding)

            new_idx = idx + max_idx
            self.vocab[new_idx] = encoded_char

        print("\n=================\nVocab initialisation done...")

    @utilities.log_to_file("HindiTokenizer-train.log")
    def train(self, text, vocab_size, verbose=False, default_initial_vocab_size=256, encoding="utf-8"):
        if self.vocab is None:
            self._build_vocab()

        print("\n`Training`...for HindiTokenizer")

        assert vocab_size >= default_initial_vocab_size
        num_merges = vocab_size - default_initial_vocab_size

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks if len(ch) > 1]

        # iteratively merge the MOST COMMON pair from the text
        # use same merge dict if exists
        self.merges = {} if self.merges is None else self.merges  # to hold all merges (int, int) -> int

        # run merging iteratively
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)

            while pair in self.merges:
                # pair was previously merged ... use this first to update IDS
                # No need to add to merges and vocab, use previously stored token
                already_merged_idx = self.merges[pair]

                # just replace already merged pairs in ids and get new ids and no need to again add to merges and vocab
                ids = merge(ids, pair, already_merged_idx)

                stats = get_stats(ids)  # get updated stats now

                # just avoiding merging when ids become less than 2
                if stats and len(ids) >= 2:
                    pair = max(stats, key=stats.get)
                else:
                    # no new merges found in this incoming data batch
                    print(f"\n\nstopping merges as no new byte pair found in the current batch")
                    break

            # mint a new token as the pair was already not in merges: assign it the next available id
            idx = len(self.vocab) + 1

            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            # save the merge
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")

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

    # directly from BPE repo
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        print("Saving tokenizer...")
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    #

# if __name__ == "__main__":
#     custom_text = """
#     <|endoftext|>ूज रहा है जहाँ चकित हो जन-जन देख अकाज
# सात वर्ष हो गये राह में, अटका कहाँ स्वराज?
#
# अटका कहाँ स्वराज? बोल दिल्ली! तू क्या कहती है?
# तू रानी बन गयी वेदना जनता क्यों सहती है?
# सबके भाग्य दबा रखे हैं किसने अपने कर में?
# उतरी थी जो विभा, हुई बंदिनी बता किस घर में
#
# समर शेष है, यह प्रकाश बंदीगृह से छूटेगा
# और नहीं तो तुझ पर पापिनी! महावज्र टूटेगा
#
# समर शेष है, उस स्वराज को सत्य बनाना होगा
# जिसका है ये न्यास उसे सत्वर पहुँचाना होगा
# धारा के मग में अनेक जो पर्वत खडे हुए हैं
# गंगा का पथ रोक इन्द्र के गज जो अडे हुए हैं
#
# कह दो उनसे झुके अगर तो जग मे यश पाएंगे
# अड़े रहे अगर तो ऐरावत पत्तों से बह जाऐंगे<|fim_prefix|><|endofprompt|>
#     """.strip()
#     special_tokens = {
#         '<|endoftext|>': 100257,
#         '<|fim_prefix|>': 100258,
#         '<|fim_middle|>': 100259,
#         '<|fim_suffix|>': 100260,
#         '<|endofprompt|>': 100276
#     }
#     text = custom_text
#     # create a Tokenizer and do 64 merges
#     tokenizer = HindiTokenizer()
#     tokenizer.train(text, 256 + 2, verbose=True)
#     tokenizer.register_special_tokens(special_tokens)
#     # verify that decode(encode(x)) == x
#     assert tokenizer.decode(tokenizer.encode(text, "all")) == text
