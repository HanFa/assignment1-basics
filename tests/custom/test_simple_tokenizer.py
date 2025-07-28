import unittest

from cs336_basics.tokenizer import Tokenizer


class TestSimpleTokenizer(unittest.TestCase):

    def test_simple_tokenizer_encode(self):
        # Vocabulary from the example
        vocab = {
            0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at'
        }

        # Merges from the example (note: corrected the typo in merge 3)
        merges = [
            (b't', b'h'),  # merge 1: t + h -> th
            (b' ', b'c'),  # merge 2: space + c -> " c"
            (b' ', b'a'),  # merge 3: space + a -> " a" (fixed typo from example)
            (b'th', b'e'),  # merge 4: th + e -> the
            (b' a', b't')  # merge 5: " a" + t -> " at"
        ]

        # Create tokenizer instance
        tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=None)

        # Test the example: 'the cat ate'
        text = 'the cat ate'
        result = tokenizer.encode(text)

        # Expected result from the example: [9, 7, 1, 5, 10, 3]
        # Breaking down:
        # - 'the' -> [9] (token for b'the')
        # - ' cat' -> [7, 1, 5] (tokens for b' c', b'a', b't')
        # - ' ate' -> [10, 3] (tokens for b' at', b'e')
        expected = [9, 7, 1, 5, 10, 3]

        self.assertEqual(expected, result,
                         f"Expected {expected}, but got {result}")

    def test_simple_tokenizer_decode(self):
        # Vocabulary from the example
        vocab = {
            0: b' ',
            1: b'a',
            2: b'c',
            3: b'e',
            4: b'h',
            5: b't',
            6: b'th',
            7: b' c',
            8: b' a',
            9: b'the',
            10: b' at'
        }

        # Merges from the example (note: corrected the typo in merge 3)
        merges = [
            (b't', b'h'),  # merge 1: t + h -> th
            (b' ', b'c'),  # merge 2: space + c -> " c"
            (b' ', b'a'),  # merge 3: space + a -> " a" (fixed typo from example)
            (b'th', b'e'),  # merge 4: th + e -> the
            (b' a', b't')  # merge 5: " a" + t -> " at"
        ]

        # Create tokenizer instance
        tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=None)

        # Act
        result = tokenizer.decode([9, 7, 1, 5, 10, 3])

        # Assert
        expected = 'the cat ate'
        self.assertEqual(expected, result,
                         f"Expected {expected}, but got {result}")
