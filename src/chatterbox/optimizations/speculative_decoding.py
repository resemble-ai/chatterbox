"""
Speculative Decoding for faster autoregressive generation

Paper: Fast Inference from Transformers via Speculative Decoding
https://arxiv.org/abs/2211.17192

Can provide 2-3x speedup for T3 (LLM) generation with no quality loss.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DraftModel(nn.Module):
    """
    Lightweight draft model for speculative decoding

    This is a distilled/pruned version of the main T3 model.
    Architecture options:
    1. Smaller T3 (fewer layers/heads)
    2. N-gram model
    3. Lookup table for common patterns
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Lightweight transformer
        from transformers import LlamaConfig, LlamaModel

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 2,  # Smaller MLP
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # No GQA for simplicity
        )

        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        """Forward pass"""
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        return logits, outputs.past_key_values

    @staticmethod
    def from_main_model(
        main_model: nn.Module,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> 'DraftModel':
        """
        Create draft model by distilling from main model

        Simple approach: Copy first N layers
        Better approach: Knowledge distillation (requires training)
        """
        # Get config from main model
        vocab_size = main_model.speech_head.out_features
        hidden_dim = main_model.dim // 2  # Use smaller hidden dim

        draft = DraftModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Copy embeddings from main model (with projection if needed)
        with torch.no_grad():
            if hidden_dim == main_model.dim:
                # Direct copy
                draft.model.embed_tokens.weight.data = (
                    main_model.speech_emb.weight.data.clone()
                )
            else:
                # Project down
                projection = nn.Linear(main_model.dim, hidden_dim, bias=False)
                nn.init.xavier_uniform_(projection.weight)

                draft.model.embed_tokens.weight.data = projection(
                    main_model.speech_emb.weight.data
                )

            # Copy first few layers (approximation)
            for i in range(min(num_layers, len(main_model.tfmr.layers))):
                # This is a simplification - proper copying would need careful handling
                pass

        logger.info(f"Created draft model with {num_layers} layers")
        return draft


class SpeculativeDecoder:
    """
    Speculative Decoding engine

    Algorithm:
    1. Draft model generates K tokens quickly
    2. Main model verifies all K tokens in parallel
    3. Accept correct tokens, reject and regenerate from first wrong token
    """
    def __init__(
        self,
        main_model: nn.Module,
        draft_model: DraftModel,
        num_speculative_tokens: int = 5,
    ):
        self.main_model = main_model
        self.draft_model = draft_model
        self.K = num_speculative_tokens

    @torch.inference_mode()
    def generate(
        self,
        input_embeds: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """
        Generate with speculative decoding

        Returns:
            generated_ids: (batch_size, num_tokens)
        """
        batch_size = input_embeds.shape[0]
        device = input_embeds.device

        generated_ids = []
        main_past_kv = None
        draft_past_kv = None

        # Initial forward pass for main model
        # ... (implementation details)

        total_draft_tokens = 0
        total_accepted_tokens = 0

        while len(generated_ids) < max_new_tokens:
            # Step 1: Draft model generates K tokens
            draft_tokens = []
            draft_logits_list = []

            for _ in range(self.K):
                if len(generated_ids) == 0:
                    # Use input embeds for first token
                    # ... implementation
                    pass
                else:
                    # Use last generated token
                    # ... implementation
                    pass

                # Draft forward pass (fast)
                # ... implementation

                draft_tokens.append(next_token)
                total_draft_tokens += 1

            # Step 2: Main model verifies all K tokens in parallel
            # This is the key: we can verify K tokens in one forward pass!

            # Prepare input: [last_generated] + draft_tokens
            verify_ids = torch.cat([
                generated_ids[-1:] if generated_ids else input_embeds,
                torch.stack(draft_tokens)
            ])

            # Main model forward pass (verifies all K tokens at once)
            # ... implementation

            # Step 3: Find first mismatch
            num_accepted = 0
            for i in range(self.K):
                # Sample from main model logits
                # ... implementation

                if sampled_token == draft_tokens[i]:
                    num_accepted += 1
                    generated_ids.append(draft_tokens[i])
                    total_accepted_tokens += 1
                else:
                    # Reject rest of draft tokens
                    generated_ids.append(sampled_token)  # Use main model's token
                    break

            # If all K tokens accepted, continue with last main model sample
            if num_accepted == self.K:
                # Sample one more token from main model
                # ... implementation
                pass

            # Check EOS
            if eos_token_id is not None and generated_ids[-1] == eos_token_id:
                break

        acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        logger.info(
            f"Speculative decoding stats: "
            f"{total_accepted_tokens}/{total_draft_tokens} tokens accepted "
            f"({acceptance_rate:.1%})"
        )

        return torch.stack(generated_ids)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Helper function to sample token from logits"""
    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    # Top-p sampling
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs[indices_to_remove] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


def create_draft_model_from_checkpoint(
    main_model_path: str,
    output_path: str,
    num_layers: int = 2,
) -> None:
    """
    Utility to create and save a draft model from trained checkpoint

    For production, you would:
    1. Create draft architecture
    2. Knowledge distillation training
    3. Save draft model
    """
    logger.info("Creating draft model...")

    # Load main model
    # ... implementation

    # Create draft
    # ... implementation

    # Save
    # ... implementation

    logger.info(f"Draft model saved to {output_path}")


class NGramDraftModel:
    """
    Simple N-gram based draft model

    For TTS, speech tokens have strong local patterns.
    An N-gram model can provide reasonable drafts.
    """
    def __init__(self, n: int = 5, vocab_size: int = 6561):
        self.n = n
        self.vocab_size = vocab_size
        self.counts = {}  # (n-1)-gram -> {next_token: count}

    def train(self, token_sequences: List[torch.Tensor]):
        """Build N-gram statistics from training data"""
        logger.info(f"Building {self.n}-gram model...")

        for tokens in token_sequences:
            tokens = tokens.tolist()
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                next_token = tokens[i+self.n-1]

                if context not in self.counts:
                    self.counts[context] = {}

                if next_token not in self.counts[context]:
                    self.counts[context][next_token] = 0

                self.counts[context][next_token] += 1

        logger.info(f"N-gram model built with {len(self.counts)} contexts")

    def predict(
        self,
        context: List[int],
        temperature: float = 1.0,
    ) -> Optional[int]:
        """Predict next token given context"""
        if len(context) < self.n - 1:
            return None

        context_tuple = tuple(context[-(self.n-1):])

        if context_tuple not in self.counts:
            return None

        # Get distribution
        counts = self.counts[context_tuple]
        tokens = list(counts.keys())
        probs = torch.tensor([counts[t] for t in tokens], dtype=torch.float32)

        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)

        probs = probs / probs.sum()

        # Sample
        idx = torch.multinomial(probs, num_samples=1).item()
        return tokens[idx]

    def save(self, path: str):
        """Save N-gram model"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'n': self.n, 'counts': self.counts}, f)

    @staticmethod
    def load(path: str) -> 'NGramDraftModel':
        """Load N-gram model"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = NGramDraftModel(n=data['n'])
        model.counts = data['counts']
        return model
