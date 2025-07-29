import os.path
import time
import unittest
import pickle
import random

from tests.adapters import get_tokenizer
from tests.common import FIXTURES_PATH


class TestTokenizerExperiments(unittest.TestCase):

    @staticmethod
    def load_x_samples_from(document_path: str, x: int = 10):
        with open(f"{FIXTURES_PATH}/{document_path}", "r") as f:
            text = f.read()

            documents = text.split('<|endoftext|>')
            documents = [doc.strip() for doc in documents if doc.strip()]
            random.seed(43)

            if len(documents) >= x:
                sampled_documents = random.sample(documents, x)
            else:
                print(f"Warning: Only {len(documents)} documents available, using all of them")
                sampled_documents = documents

            sampled_text = '<|endoftext|>'.join(sampled_documents)
            return sampled_text

    @staticmethod
    def load_10_samples_from_tinystories_sample_5M():
        return TestTokenizerExperiments.load_x_samples_from("tinystories_sample_5M.txt")

    @staticmethod
    def load_10_samples_from_owt_valid():
        return TestTokenizerExperiments.load_x_samples_from("../../data/owt_valid.txt")

    def test_10k_tiny_stories_tokenizer(self):
        vocab_path_fn = os.path.join("pickles", "vocab.pickle")
        merge_path_fn = os.path.join("pickles", "merges.pickle")

        with open(merge_path_fn, "rb") as f:
            merges = pickle.load(f)

        with open(vocab_path_fn, "rb") as f:
            vocab = pickle.load(f)

        tokenizer = get_tokenizer(vocab, merges, special_tokens=None)

        tiny_sampled_text, owt_sampled_text = TestTokenizerExperiments.load_10_samples_from_tinystories_sample_5M(), TestTokenizerExperiments.load_10_samples_from_owt_valid()

        for t, sampled_text in [("tiny stories", tiny_sampled_text), ("owt", owt_sampled_text)]:
            start = time.time()
            encoded_ids = tokenizer.encode(sampled_text)
            duration = time.time() - start
            print(f"[{t}] text length: {len(sampled_text)}, bytes length: {len(sampled_text.encode('utf-8'))}, "
                  f"encoded ids num: {len(encoded_ids)}, compress rate: {len(sampled_text.encode('utf-8')) / len(encoded_ids)} bytes/token, "
                  f"tput: {len(sampled_text.encode('utf-8')) / duration} bytes/sec")

    def test_32k_owt_tokenizer(self):
        vocab_path_fn = os.path.join("pickles", "vocab_owt.pickle")
        merge_path_fn = os.path.join("pickles", "merges_owt.pickle")

        with open(merge_path_fn, "rb") as f:
            merges = pickle.load(f)

        with open(vocab_path_fn, "rb") as f:
            vocab = pickle.load(f)

        tokenizer = get_tokenizer(vocab, merges, special_tokens=None)

        tiny_sampled_text, owt_sampled_text = TestTokenizerExperiments.load_10_samples_from_tinystories_sample_5M(), TestTokenizerExperiments.load_10_samples_from_owt_valid()

        for t, sampled_text in [("tiny stories", tiny_sampled_text), ("owt", owt_sampled_text)]:
            start = time.time()
            encoded_ids = tokenizer.encode(sampled_text)
            duration = time.time() - start
            print(f"[{t}] text length: {len(sampled_text)}, bytes length: {len(sampled_text.encode('utf-8'))}, "
                  f"encoded ids num: {len(encoded_ids)}, compress rate: {len(sampled_text.encode('utf-8')) / len(encoded_ids)} bytes/token, "
                  f"tput: {len(sampled_text.encode('utf-8')) / duration} bytes/sec")