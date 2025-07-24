import unittest
import pathlib
import cProfile
import pstats
import pickle
from viztracer import VizTracer

from tests.adapters import run_train_bpe

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data"


class TestTrainBPETinyStories(unittest.TestCase):

    def setUp(self):
        self.tracer = VizTracer()
        self.profiler = cProfile.Profile()

    def tearDown(self):
        pass

    def test_train_bpe_tiny_stories(self):
        self.tracer.start()

        try:
            input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
            vocab, merges = run_train_bpe(
                input_path=input_path,
                vocab_size=10000,
                special_tokens=["<|endoftext|>"],
                multiprocess_num=16
            )

        finally:
            self.tracer.stop()
            self.tracer.save()

    def test_train_bpe_tiny_stories_c_profiler(self):
        self.profiler.enable()

        try:
            input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
            vocab, merges = run_train_bpe(
                input_path=input_path,
                vocab_size=10000,
                special_tokens=["<|endoftext|>"],
                multiprocess_num=16
            )

        finally:
            self.profiler.disable()

            stats = pstats.Stats(self.profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)

    def test_train_bpe_tiny_stories_no_profilers(self):

        input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
            multiprocess_num=16
        )

        with open("vocab.pickle", 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("merges.pickle", 'wb') as handle:
            pickle.dump(merges, handle, protocol=pickle.HIGHEST_PROTOCOL)
