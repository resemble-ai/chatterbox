import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TextChunkTests(unittest.TestCase):
    def split(self, text, limit):
        path = ROOT / 'src/chatterbox/text_chunks.py'
        self.assertTrue(path.exists(), 'Automatic text splitting is missing')
        spec = importlib.util.spec_from_file_location('text_chunks', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.split_text(text, limit)

    def test_sentences_keep_order_and_punctuation(self):
        self.assertEqual(self.split('First sentence. Second sentence! Third?', 30),
                         ['First sentence.', 'Second sentence! Third?'])

    def test_long_sentence_preserves_every_word(self):
        text = ' '.join(['development'] * 90)
        chunks = self.split(text, 100)
        self.assertEqual(' '.join(chunks), text)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            self.split('   ', 100)

    def test_long_word_rejected_instead_of_discarded(self):
        with self.assertRaises(ValueError):
            self.split('x' * 101, 100)


if __name__ == '__main__':
    unittest.main()
