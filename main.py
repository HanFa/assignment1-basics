import os.path
import pickle


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


merges_pickle_fn = os.path.join("pickles", "merges_owt.pickle")
vocab_pickle_fn = os.path.join("pickles", "vocab_owt.pickle")

if __name__ == '__main__':
    with open(merges_pickle_fn, "rb") as f:
        merges = pickle.load(f)
        print(merges)

    with open(vocab_pickle_fn, "rb") as f:
        vocab = pickle.load(f)
        print(vocab)

        max_entry = max(vocab.items(), key=lambda x: len(x[1]))
        print(max_entry)
