import unittest

from cs336_basics.pretokenization import merge_pretoken_to_count, run_train_bpe_with_pretokenization_dict


class TestMergePretokenToCount(unittest.TestCase):

    def test_example_from_docstring(self):
        """Test the exact example provided in the docstring"""
        # Input from docstring example
        pretoken_to_count = {
            (b'l', b'o', b'w'): 5,
            (b'l', b'o', b'w', b'e', b's', b't'): 2,
            (b'w', b'i', b'd', b'e', b's', b't'): 3,
            (b'n', b'e', b'w', b'e', b's', b't'): 6
        }

        pair = (b's', b't')

        # Expected output from docstring
        expected = {
            (b'l', b'o', b'w'): 5,
            (b'l', b'o', b'w', b'e', b'st'): 2,
            (b'w', b'i', b'd', b'e', b'st'): 3,
            (b'n', b'e', b'w', b'e', b'st'): 6
        }

        result = merge_pretoken_to_count(pretoken_to_count, pair)
        self.assertEqual(result, expected)


class TestRunBPE(unittest.TestCase):

    def test_run_train_bpe_with_pretokenization_dict_one_merge(self):
        initial_pretoken_to_count = {
            (b'l', b'o', b'w'): 5,
            (b'l', b'o', b'w', b'e', b'r'): 2,
            (b'w', b'i', b'd', b'e', b's', b't'): 3,
            (b'n', b'e', b'w', b'e', b's', b't'): 6
        }

        vocab = {
            0: b'l', 1: b'o', 2: b'w', 3: b'e', 4: b'r',
            5: b'i', 6: b'd', 7: b's', 8: b't', 9: b'n'
        }
        vocab_set = set(vocab.values())
        vocab_size = 11  # Allow one merge

        # Act
        vocab, vocab_set, merges = run_train_bpe_with_pretokenization_dict(
            initial_pretoken_to_count, vocab, vocab_set, vocab_size
        )

        # Assert
        assert len(merges) == 1
        assert merges[0] == (b's', b't')
        assert len(vocab_set) == vocab_size

    def test_run_train_bpe_with_pretokenization_dict_six_merges(self):
        initial_pretoken_to_count = {
            (b'l', b'o', b'w'): 5,
            (b'l', b'o', b'w', b'e', b'r'): 2,
            (b'w', b'i', b'd', b'e', b's', b't'): 3,
            (b'n', b'e', b'w', b'e', b's', b't'): 6
        }

        vocab = {
            0: b'l', 1: b'o', 2: b'w', 3: b'e', 4: b'r',
            5: b'i', 6: b'd', 7: b's', 8: b't', 9: b'n'
        }
        vocab_set = set(vocab.values())
        vocab_size = 16  # Allow six merges

        # Act
        vocab, vocab_set, merges = run_train_bpe_with_pretokenization_dict(
            initial_pretoken_to_count, vocab, vocab_set, vocab_size
        )

        # Assert
        assert len(merges) == 6
        assert merges == [(b's', b't'), (b'e', b'st'), (b'o', b'w'), (b'l', b'ow'), (b'w', b'est'), (b'n', b'e')]
        assert len(vocab_set) == vocab_size
