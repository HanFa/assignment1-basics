import unittest
import pathlib
import cProfile
import pstats
import pickle
from viztracer import VizTracer

from tests.adapters import run_train_bpe

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data"


class TestTrainBPEOpenWebText(unittest.TestCase):

    def setUp(self):
        self.tracer = VizTracer()
        self.profiler = cProfile.Profile()

    def tearDown(self):
        pass

    def test_train_bpe_owt_c_profiler(self):
        self.profiler.enable()

        try:
            input_path = DATA_PATH / "owt_train.txt"
            vocab, merges = run_train_bpe(
                input_path=input_path,
                vocab_size=32000,
                special_tokens=["<|endoftext|>"],
                multiprocess_num=24
            )

            with open("vocab_owt.pickle", 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open("merges_owt.pickle", 'wb') as handle:
                pickle.dump(merges, handle, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.profiler.disable()

            stats = pstats.Stats(self.profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)
