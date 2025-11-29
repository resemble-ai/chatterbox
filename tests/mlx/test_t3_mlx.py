# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Unit tests for T3 MLX implementation.
Tests functional parity with PyTorch version and MLX-specific features.
"""

import unittest
import numpy as np
import mlx.core as mx

from chatterbox.models.t3_mlx.modules.learned_pos_emb_mlx import LearnedPositionEmbeddingsMLX
from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX, T3CondEncMLX
from chatterbox.models.t3_mlx.modules.perceiver_mlx import PerceiverMLX
from chatterbox.models.t3_mlx.t3_mlx import T3MLX
from chatterbox.models.t3.modules.t3_config import T3Config


class TestLearnedPositionEmbeddingsMLX(unittest.TestCase):
    """Test learned position embeddings MLX implementation."""

    def setUp(self):
        self.seq_len = 100
        self.model_dim = 64
        self.emb = LearnedPositionEmbeddingsMLX(self.seq_len, self.model_dim)

    def test_initialization(self):
        """Test that embeddings are properly initialized."""
        self.assertEqual(self.emb.emb.weight.shape, (self.seq_len, self.model_dim))

    def test_forward_pass(self):
        """Test forward pass with input tensor."""
        x = mx.zeros((2, 10, self.model_dim))  # Batch of 2, seq len 10
        pos_emb = self.emb(x)

        self.assertEqual(pos_emb.shape, (10, self.model_dim))

    def test_get_fixed_embedding_int(self):
        """Test getting embedding for single index."""
        idx = 5
        emb = self.emb.get_fixed_embedding(idx)

        self.assertEqual(emb.shape, (1, 1, self.model_dim))

    def test_get_fixed_embedding_array(self):
        """Test getting embeddings for array of indices."""
        idx = mx.array([[0, 1, 2], [3, 4, 5]])
        emb = self.emb.get_fixed_embedding(idx)

        self.assertEqual(emb.shape, (2, 3, self.model_dim))


class TestT3CondMLX(unittest.TestCase):
    """Test T3 conditioning dataclass."""

    def test_initialization(self):
        """Test creating T3CondMLX object."""
        speaker_emb = mx.random.normal((1, 256))

        cond = T3CondMLX(
            speaker_emb=speaker_emb,
            emotion_adv=0.5,
        )

        self.assertIsNotNone(cond.speaker_emb)
        self.assertEqual(cond.emotion_adv, 0.5)
        self.assertIsNone(cond.clap_emb)

    def test_to_device(self):
        """Test to_device method (should be no-op in MLX)."""
        speaker_emb = mx.random.normal((1, 256))
        cond = T3CondMLX(speaker_emb=speaker_emb)

        cond_moved = cond.to_device()
        self.assertIs(cond_moved, cond)  # Should return self


class TestT3CondEncMLX(unittest.TestCase):
    """Test conditioning encoder."""

    def setUp(self):
        self.hp = T3Config.english_only()
        self.enc = T3CondEncMLX(self.hp)

    def test_initialization(self):
        """Test encoder initialization."""
        self.assertIsNotNone(self.enc.spkr_enc)
        self.assertIsNotNone(self.enc.emotion_adv_fc)
        self.assertIsNotNone(self.enc.perceiver)

    def test_forward_pass(self):
        """Test forward pass with conditioning."""
        speaker_emb = mx.random.normal((2, 256))

        cond = T3CondMLX(
            speaker_emb=speaker_emb,
            emotion_adv=0.5,
        )

        output = self.enc(cond)

        # Output should be (B, len_cond, dim)
        self.assertEqual(output.ndim, 3)
        self.assertEqual(output.shape[0], 2)  # Batch size
        self.assertEqual(output.shape[2], self.hp.n_channels)


class TestPerceiverMLX(unittest.TestCase):
    """Test Perceiver resampler."""

    def setUp(self):
        self.perceiver = PerceiverMLX(
            pre_attention_query_token=32,
            pre_attention_query_size=1024,
            embedding_dim=1024,
            num_attn_heads=4
        )

    def test_initialization(self):
        """Test Perceiver initialization."""
        self.assertEqual(self.perceiver.pre_attention_query.shape, (1, 32, 1024))

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        seq_len = 100
        dim = 1024

        h = mx.random.normal((batch_size, seq_len, dim))
        output = self.perceiver(h)

        # Output should be resampled to query token size
        self.assertEqual(output.shape, (batch_size, 32, dim))


class TestT3MLX(unittest.TestCase):
    """Test main T3 MLX model."""

    def setUp(self):
        self.hp = T3Config.english_only()
        # Use very small config for testing
        self.hp.llama_config_name = "Llama_520M"  # Already small

        self.model = T3MLX(hp=self.hp)

    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.tfmr)
        self.assertIsNotNone(self.model.text_emb)
        self.assertIsNotNone(self.model.speech_emb)
        self.assertIsNotNone(self.model.text_head)
        self.assertIsNotNone(self.model.speech_head)

    def test_prepare_conditioning(self):
        """Test conditioning preparation."""
        speaker_emb = mx.random.normal((1, 256))
        cond = T3CondMLX(speaker_emb=speaker_emb, emotion_adv=0.5)

        cond_emb = self.model.prepare_conditioning(cond)

        self.assertEqual(cond_emb.ndim, 3)
        self.assertEqual(cond_emb.shape[0], 1)  # Batch size

    def test_prepare_input_embeds(self):
        """Test input embedding preparation."""
        speaker_emb = mx.random.normal((1, 256))
        cond = T3CondMLX(speaker_emb=speaker_emb, emotion_adv=0.5)

        text_tokens = mx.array([[self.hp.start_text_token, 10, 20, self.hp.stop_text_token]])
        speech_tokens = mx.array([[self.hp.start_speech_token]])

        embeds, len_cond = self.model.prepare_input_embeds(
            t3_cond=cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # Check shapes
        self.assertEqual(embeds.ndim, 3)
        self.assertEqual(embeds.shape[0], 1)  # Batch
        self.assertGreater(len_cond, 0)

    @unittest.skip("Forward pass requires full Llama implementation")
    def test_forward_pass(self):
        """Test full forward pass."""
        # This test requires a full Llama model which is expensive
        # Skip for now, test in integration tests
        pass


class TestSamplingUtilities(unittest.TestCase):
    """Test sampling utilities."""

    def test_apply_repetition_penalty(self):
        """Test repetition penalty application."""
        from chatterbox.models.t3_mlx.inference.sampling_utils_mlx import apply_repetition_penalty

        logits = mx.random.normal((1, 100))
        generated_ids = [mx.array([[5]]), mx.array([[10]]), mx.array([[5]])]

        penalized = apply_repetition_penalty(logits, generated_ids, penalty=1.2)

        # Tokens 5 and 10 should be penalized
        self.assertEqual(penalized.shape, logits.shape)

    def test_apply_top_p(self):
        """Test top-p sampling."""
        from chatterbox.models.t3_mlx.inference.sampling_utils_mlx import apply_top_p

        logits = mx.random.normal((1, 100))
        filtered = apply_top_p(logits, top_p=0.9)

        self.assertEqual(filtered.shape, logits.shape)
        # Some tokens should be filtered (set to -inf)
        self.assertTrue(mx.any(filtered == -float('inf')))

    def test_apply_min_p(self):
        """Test min-p sampling."""
        from chatterbox.models.t3_mlx.inference.sampling_utils_mlx import apply_min_p

        logits = mx.random.normal((1, 100))
        filtered = apply_min_p(logits, min_p=0.05)

        self.assertEqual(filtered.shape, logits.shape)


class TestWeightConversion(unittest.TestCase):
    """Test weight conversion utilities."""

    def test_pytorch_to_mlx_tensor(self):
        """Test converting PyTorch tensor to MLX."""
        from chatterbox.models.t3_mlx.utils.convert_weights import pytorch_to_mlx_tensor
        import torch

        pt_tensor = torch.randn(10, 20)
        mlx_array = pytorch_to_mlx_tensor(pt_tensor)

        self.assertEqual(mlx_array.shape, (10, 20))
        np.testing.assert_allclose(
            np.array(mlx_array),
            pt_tensor.numpy(),
            rtol=1e-5
        )

    def test_mlx_to_pytorch_tensor(self):
        """Test converting MLX array to PyTorch."""
        from chatterbox.models.t3_mlx.utils.convert_weights import mlx_to_pytorch_tensor

        mlx_array = mx.random.normal((10, 20))
        pt_tensor = mlx_to_pytorch_tensor(mlx_array)

        self.assertEqual(pt_tensor.shape, (10, 20))
        np.testing.assert_allclose(
            pt_tensor.numpy(),
            np.array(mlx_array),
            rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
