import unittest
from unittest.mock import Mock

import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.text_chunks import split_text


class TextChunkTests(unittest.TestCase):
    split = staticmethod(split_text)

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

    def test_invalid_chunk_sizes_rejected(self):
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.split('Hello.', limit)


class LongGenerationTests(unittest.TestCase):
    def setUp(self):
        # Neural inference is expensive; test orchestration with deterministic audio.
        self.model = object.__new__(ChatterboxTurboTTS)
        self.model.sr = 1000
        self.model.model_label = 'Turbo'
        self.model.conds = object()
        self.model.prepare_conditionals = Mock()
        self.model.generate = Mock(side_effect=[
            torch.tensor([[1., 1.]]), torch.tensor([[2., 2.]]), torch.tensor([[3., 3.]])
        ])

    def test_audio_order_pause_and_reference_reuse(self):
        result = self.model.generate_long(
            'First sentence. Second sentence! Third?',
            audio_prompt_path='reference.wav', max_chunk_chars=16, chunk_pause_ms=2,
            max_gen_len=1500, temperature=0.6,
        )
        self.assertEqual(result.tolist(), [[1, 1, 0, 0, 2, 2, 0, 0, 3, 3]])
        self.model.prepare_conditionals.assert_called_once_with(
            'reference.wav', exaggeration=0.0, norm_loudness=True)
        calls = self.model.generate.call_args_list
        self.assertEqual([call.args[0] for call in calls],
                         ['First sentence.', 'Second sentence!', 'Third?'])
        for call in calls:
            self.assertEqual(call.kwargs['max_gen_len'], 1500)
            self.assertEqual(call.kwargs['temperature'], 0.6)
            self.assertNotIn('audio_prompt_path', call.kwargs)

    def test_zero_pause_and_cached_voice(self):
        result = self.model.generate_long('First sentence. Second sentence!',
                                          max_chunk_chars=16, chunk_pause_ms=0)
        self.assertEqual(result.tolist(), [[1, 1, 2, 2]])
        self.model.prepare_conditionals.assert_not_called()

    def test_invalid_options_fail_before_preparing_reference(self):
        for options in ({'chunk_pause_ms': -1}, {'chunk_pause_ms': float('nan')},
                        {'max_gen_len': 0}, {'max_gen_len': True}, {'max_chunk_chars': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model.generate_long('Hello.', audio_prompt_path='reference.wav', **options)
        self.model.prepare_conditionals.assert_not_called()
        self.model.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
