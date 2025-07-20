import os
from collections import defaultdict
from typing import BinaryIO


def run_train_bpe_with_pretokenization_dict(
        pretoken_to_count: dict[tuple[bytes], int],
        vocab: dict[int, bytes],
        vocab_set: set[bytes],
        vocab_size: int
):
    """
    Runs BPE training after pre-tokenization results `pretoken_to_count` dictionary is ready.

    Args:
        pretoken_to_count: pre-tokenization results, each word to its occurrence
        vocab: Vocabulary from indices to bytes
        vocab_set: Vocabulary set
        vocab_size: Capping vocabulary size
    Returns:
        vocab: it updates vocab in-place
        merges:
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
    """
    # find most frequent bytes pair and merge
    merges = []
    while len(vocab) < vocab_size:
        merge_to_count: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for pretoken, count in pretoken_to_count.items():
            for first_bytes, second_bytes in zip(pretoken[:-1], pretoken[1:]):
                merge_to_count[(first_bytes, second_bytes)] += count

        if not merge_to_count:
            break

        pair: tuple[bytes, bytes] = max(merge_to_count, key=lambda x: (merge_to_count[x], x))
        first_bytes, second_bytes = pair

        if first_bytes + second_bytes not in vocab_set:
            vocab[len(vocab_set)] = first_bytes + second_bytes
            merges.append((first_bytes, second_bytes))
            vocab_set.add((first_bytes + second_bytes))

        # update pre-tokenization result with merged pair
        pretoken_to_count = merge_pretoken_to_count(pretoken_to_count, pair)

    return vocab, vocab_set, merges


def merge_pretoken_to_count(pretoken_to_count: dict[tuple[bytes], int], pair: tuple[bytes, bytes]):
    """
    Update dictionary `pretoken_to_count` after giving the most frequent bytes pair

    Args:
        pretoken_to_count: a map from pretoken (basically a tuple of bytes) to the occurrence count
            An example is {(l, o, w): 5, (l, o, w, e, s, t): 2, (w, i, d, e, s, t): 3, (n, e, w, e, s, t): 6}
        pair:
            Pair of bytes (A and B) to be merged. An example is (s, t)
    Returns:
        Returns a new version of pretoken_to_count:
            Example result is {(l, o, w): 5, (l, o, w, e, st): 2, (w, i, d, e, st): 3, (n, e, w, e, st): 6}
    """

    first_bytes, second_bytes = pair
    merged_bytes = first_bytes + second_bytes

    new_pretoken_to_count: dict[tuple[bytes], int] = defaultdict(int)
    for pretoken, count in pretoken_to_count.items():
        idx = 0
        new_pretoken = []
        while idx < len(pretoken):

            if idx + 1 < len(pretoken) and pretoken[idx] == first_bytes and pretoken[
                idx + 1] == second_bytes:
                new_pretoken.append(merged_bytes)
                idx += 2
            else:
                new_pretoken.append(pretoken[idx])
                idx += 1

        new_pretoken_to_count[tuple(new_pretoken)] = count

    return new_pretoken_to_count


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
