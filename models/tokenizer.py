
from __future__ import annotations

import html
import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import regex as re
import torch

_PATTERN = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
    re.IGNORECASE,)


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def get_pairs(word: Tuple[str, ...]) -> set:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def basic_clean(text: str) -> str:
    text = html.unescape(html.unescape(text))
    return text.strip()


class TokenizerOutput:
    __slots__ = ("input_ids", "attention_mask")

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, key):
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}[key]

    def keys(self):
        return ("input_ids", "attention_mask")


class CLIPTokenizer:
    """BPE tokenizer của CLIP."""

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        model_max_length: int = 77,
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: Optional[str] = None,
        html_unescape: bool = True,
    ):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder: Dict[str, int] = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        if lines and lines[0].startswith("#version"):
            lines = lines[1:]
        merges = [tuple(line.split()) for line in lines if len(line.split()) == 2]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {bos_token: bos_token, eos_token: eos_token}

        self.model_max_length = model_max_length
        self.bos_token, self.eos_token = bos_token, eos_token
        self.bos_token_id = self.encoder[bos_token]
        self.eos_token_id = self.encoder[eos_token]
        self.pad_token_id = 0
        self.unk_token_id = self.eos_token_id
        self.html_unescape = html_unescape

    @classmethod
    def from_pretrained(cls, path: str, subfolder: str = "tokenizer") -> "CLIPTokenizer":
        folder = os.path.join(path, subfolder) if subfolder else path
        if not os.path.isdir(folder):
            folder = path
        vocab_file = os.path.join(folder, "vocab.json")
        merges_file = os.path.join(folder, "merges.txt")
        for f in (vocab_file, merges_file):
            if not os.path.isfile(f):
                raise FileNotFoundError(
                    f"Không thấy {f}. Cần tải kèm thư mục 'tokenizer' của checkpoint."
                )
        model_max_length = 77
        cfg_path = os.path.join(folder, "tokenizer_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            model_max_length = int(cfg.get("model_max_length", 77) or 77)
            if model_max_length > 1e6:
                model_max_length = 77
        return cls(vocab_file, merges_file, model_max_length=model_max_length)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
            
        # [QUAN TRỌNG] Chỉ ngoại lệ đúng dấu "!" để khớp chuẩn của CLIPTokenizerFast
        if token == "!" and "!" in self.encoder:
            self.cache[token] = "!"
            return "!"

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def tokenize(self, text: str) -> List[str]:
        if self.html_unescape:
            text = basic_clean(text)
        text = whitespace_clean(text).lower()
        tokens: List[str] = []
        for token in _PATTERN.findall(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            tokens.extend(self.bpe(token).split(" "))
        return tokens

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        return [self.encoder.get(t, self.unk_token_id) for t in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        text = "".join(self.decoder.get(int(i), "") for i in ids)
        byte_array = bytearray(self.byte_decoder.get(c, 32) for c in text)
        return byte_array.decode("utf-8", errors="replace").replace("</w>", " ").strip()

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        padding: Union[bool, str] = "max_length",
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        **ignored: Any,
    ) -> TokenizerOutput:
        texts = [text] if isinstance(text, str) else list(text)
        max_length = max_length or self.model_max_length

        batch_ids, batch_mask = [], []
        for t in texts:
            ids = self.convert_tokens_to_ids(self.tokenize(t))
            if truncation and len(ids) > max_length - 2:
                ids = ids[: max_length - 2]
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            mask = [1] * len(ids)
            if padding in (True, "max_length") and len(ids) < max_length:
                pad = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad
                mask = mask + [0] * pad
            batch_ids.append(ids)
            batch_mask.append(mask)

        if return_tensors == "pt":
            return TokenizerOutput(torch.tensor(batch_ids, dtype=torch.long),
                                   torch.tensor(batch_mask, dtype=torch.long))
        return TokenizerOutput(batch_ids, batch_mask)

    def __len__(self) -> int:
        return len(self.encoder)

