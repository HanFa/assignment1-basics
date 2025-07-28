import regex as re

from collections import defaultdict
from typing import Protocol, Iterator, Iterable

pretokenization_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize_count_in_boundary(input_path, start, end,
                                   special_tokens) -> dict[tuple, int]:
    """
    Run single thread pretokenization given an input path to a file, with specified boundary between (start, end)
    Args:
        input_path: the path to file for pre-tokenization
        start: boundary start index
        end: boundary end index
        special_tokens: a list of special token

    Returns:
        a list of pre-tokens to count of each of them
    """

    result: dict[tuple, int] = defaultdict(int)
    from tqdm import tqdm
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # split the chunk according to the special token
        pattern = "|".join(re.escape(token) for token in special_tokens)
        documents = re.split(pattern, chunk)

        for document in tqdm(documents, desc=f"pre_tokenization: start={start}, end={end}"):
            # perform pre tokenization
            for match in re.finditer(pretokenization_pat, document):
                pretoken = match.group()
                pretoken_bytes = pretoken.encode('utf-8')
                pretoken_bytes_tuple = tuple(bytes([byte]) for byte in pretoken_bytes)
                result[pretoken_bytes_tuple] += 1

        return result


def pre_tokenize_text(text: str, special_tokens: list[str]) -> list[tuple]:
    """ Convert text from string to list of bytes tuples with special tokens stripped out. """
    result: list[tuple] = list()

    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
    documents = re.split(f'({pattern})', text)

    for document in documents:
        if not document:
            continue

        if document in special_tokens:
            special_token_bytes = document.encode('utf-8')
            result.append(tuple([special_token_bytes]))
        else:
            for match in re.finditer(pretokenization_pat, document):
                pretoken = match.group()
                pretoken_bytes = pretoken.encode('utf-8')
                pretoken_bytes_tuple = tuple(bytes([byte]) for byte in pretoken_bytes)
                result.append(pretoken_bytes_tuple)

    return result


class Tokenizer(Protocol):
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """
        Initialize a Tokenizer, given a vocabulary and a list of merges
        Args:
            vocab: given vocabulary from ID to bytes
            merges: a list of merge-pairs of bytes and bytes
            special_tokens: a list of special tokens in string
        """
        self.vocab = vocab
        self.merges: set[tuple[bytes, bytes]] = set(merges)
        self.merges_priorities: dict[tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }
        self.special_tokens = special_tokens if special_tokens else ["<|endoftext|>"]

        self.bytes_to_idx: dict[bytes, int] = defaultdict(int)

        for key, val in self.vocab.items():
            self.bytes_to_idx[val] = key

    @staticmethod
    def from_files(vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and
        list of merges
        Args:
            vocab_filepath: path to the vocab pickle
            merges_filepath: path to the list of merges pickle
            special_tokens:  a list of special tokens

        Returns: a Tokenizer object
        """

        import pickle

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Tokenizer encodes a string into a list of IDs
        Args:
            text: string to encode

        Returns: a list of encoded IDs
        """
        results: list[int] = list()
        # pre-tokenize
        pre_tokens: list[tuple] = pre_tokenize_text(text, self.special_tokens)

        # encode each pre-token
        for pre_token in pre_tokens:
            assert len(pre_token) > 0

            if len(pre_token) == 1:
                results.append(self.bytes_to_idx[pre_token[0]])
                continue

            current_merges = list(pre_token)

            while True:
                pairs_found: list[tuple[int, tuple]] = list()

                for idx in range(len(current_merges) - 1):
                    pair = (current_merges[idx], current_merges[idx + 1])

                    if pair in self.merges:
                        pairs_found.append((idx, pair))

                if not pairs_found:
                    break

                best_merge = min(pairs_found, key=lambda x: self.merges_priorities[x[1]])
                merge_idx, (left, right) = best_merge
                current_merges = current_merges[:merge_idx] + [left + right] + current_merges[merge_idx + 2:]

            for merge in current_merges:
                results.append(self.bytes_to_idx[merge])

        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """ Returns a generator that lazily yields token IDs given a strings iterable. """
        for text in iterable:
            for ids in self.encode(text):
                yield ids

    def decode(self, ids: list[int]) -> str:
        """ Decode from IDs into text string. """
        results = b''.join(self.vocab[idx] for idx in ids)
        return results.decode('utf-8', errors='replace')
