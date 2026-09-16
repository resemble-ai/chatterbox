"""Run with: python -m unittest discover -s tests -p test_turbo_alignment.py."""

import unittest

import torch
from transformers import GPT2Config, GPT2Model

from chatterbox.models.t3.inference.turbo_alignment import TurboAlignmentCapture, word_end_samples


class AlignmentCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT2Model(GPT2Config(
            n_layer=2, n_head=2, n_embd=16, n_positions=64, vocab_size=8,
            attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
            attn_implementation="sdpa",
        )).eval()
        self.prefill = torch.randn(1, 7, 16)  # 3 conditioning + 3 text + 1 BOS
        self.step = torch.randn(1, 1, 16)

    @torch.inference_mode()
    def generate(self):
        first = self.model(inputs_embeds=self.prefill, use_cache=True)
        second = self.model(inputs_embeds=self.step, past_key_values=first.past_key_values, use_cache=True)
        return first.last_hidden_state.clone(), second.last_hidden_state.clone()

    def test_capture_preserves_outputs_and_matches_projected_qk(self):
        before = self.generate()
        config = self.model.config._attn_implementation
        seen = []
        handle = self.model.h[0].attn.c_attn.register_forward_hook(lambda m, a, o: seen.append(o.detach().clone()))
        try:
            with TurboAlignmentCapture(self.model, 3, [(0, 1), (1, 0), (0, 0)]) as capture:
                after = self.generate()
        finally:
            handle.remove()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
        self.assertEqual(self.model.config._attn_implementation, config)
        weights = capture.text_attention()
        self.assertEqual(tuple(weights.shape), (3, 2, 3))
        q = seen[0][0, -1, :16].reshape(2, 8)[1]
        k = seen[0][0, 3:6, 16:32].reshape(3, 2, 8)[:, 1]
        expected = (q @ k.T / 8**0.5).softmax(-1)
        torch.testing.assert_close(weights[0, 0], expected)
        self.assertFalse(any(block.attn.c_attn._forward_hooks for block in self.model.h))

    def test_exception_and_reuse_remove_hooks(self):
        capture = TurboAlignmentCapture(self.model, 3, [(0, 0)], max_steps=1)
        with self.assertRaisesRegex(ValueError, 'max_steps'), capture:
            self.generate()
        self.assertFalse(any(block.attn.c_attn._forward_hooks for block in self.model.h))
        with capture, torch.inference_mode():
            self.model(inputs_embeds=self.prefill, use_cache=True)
        self.assertEqual(tuple(capture.text_attention().shape), (1, 1, 3))

    def test_invalid_inputs_fail_before_hooks(self):
        for count, heads in [(0, [(0, 0)]), (3, []), (3, [(2, 0)]), (3, [(0, 2)]), (3, [(0, 0), (0, 0)])]:
            with self.assertRaises(ValueError):
                TurboAlignmentCapture(self.model, count, heads)
        with self.assertRaisesRegex(ValueError, 'BOS'), TurboAlignmentCapture(self.model, 8, [(0, 0)]):
            self.generate()
        self.assertFalse(any(block.attn.c_attn._forward_hooks for block in self.model.h))


class WordTimingTests(unittest.TestCase):
    def test_repeated_words_backtracking_invalid_tokens_and_eos(self):
        attention = torch.nn.functional.one_hot(torch.tensor([2, 0, 1, 2, 2, 1, 3]), 4).float()
        # One invalid speech token is filtered out before S3Gen; final row is EOS.
        mask = torch.tensor([True, True, False, True, True, True])
        words = word_end_samples(attention, "boom then boom.", [[0, 4], [5, 9], [10, 14], [14, 15]],
                                 mask, 24000, 8 * 960)
        self.assertEqual(words, [[0, 4, 1920], [10, 14, 3840]])
        self.assertEqual(word_end_samples(attention, "boom", [[0, 4]], mask, 24000, 1), [])


if __name__ == '__main__':
    unittest.main()
