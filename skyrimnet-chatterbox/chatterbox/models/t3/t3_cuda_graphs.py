import torch
from loguru import logger
from typing import Optional, Tuple, Callable, Any, Dict, Union

TOKEN_LIMIT = 1500


def get_next_bucket(
    seq_len: int, bucket_size: int = 250, max_bucket: int = TOKEN_LIMIT
) -> int:
    bucket = ((seq_len - 1) // bucket_size + 1) * bucket_size
    return min(bucket, max_bucket)


class T3StepCUDAGraphWrapper:
    """
    A wrapper class that automatically captures and replays CUDA graphs for optimized inference
    with support for bucketing to handle dynamic max_position values.

    Maintains separate graphs for different bucket sizes while sharing kv_cache and memory.
    """

    def __init__(
        self,
        generate_token: Callable,
        patched_model: Any,
        kv_cache: Any,
        repetition_penalty_processor: Any,
        min_p_warper: Any,
        top_p_warper: Any,
    ):
        """
        Initialize the CUDA graph wrapper with bucketing support.

        Args:
            generate_token: The function to wrap with CUDA graph
            patched_model: The model instance
            kv_cache: The key-value cache (shared across all buckets)
            repetition_penalty_processor: Repetition penalty processor
            min_p_warper: Min-p warper
            top_p_warper: Top-p warper
        """
        self.generate_token = generate_token
        self.patched_model = patched_model
        self.kv_cache = kv_cache  # Shared across all buckets
        self.repetition_penalty_processor = repetition_penalty_processor
        self.min_p_warper = min_p_warper
        self.top_p_warper = top_p_warper

        # Dictionary to store graphs and static tensors for each bucket
        self._bucket_graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._bucket_static_tensors: Dict[Tuple[int, int], dict] = {}

        # Track which buckets have been captured
        self._captured_buckets = set()

        self.dtype = patched_model.dtype
        self._current_cfg_mode = None  # Track current CFG mode for validation

    def guard(self, cfg_weight: float = 0.0):
        """
        Validate critical parameters have not changed, such as effective batch size or dtype.
        Reset all graphs if CFG mode changes since KV cache batch dimensions are incompatible.
        """
        assert self.patched_model.dtype == self.dtype
        
        current_cfg_mode = 1 if cfg_weight > 0.0 else 0
        if self._current_cfg_mode is not None and self._current_cfg_mode != current_cfg_mode:
            logger.debug(f"CFG mode changed from {self._current_cfg_mode} to {current_cfg_mode}, resetting all CUDA graphs")
            self.reset()
            self.kv_cache.reset()
        
        self._current_cfg_mode = current_cfg_mode

    def _capture_graph_for_bucket(
        self,
        bucket_key: Tuple[int, int],
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        stride_length: int,
        max_position: Optional[int] = None,
    ) -> None:
        """Capture the CUDA graph for a specific bucket."""
        logger.debug(f"Capturing CUDA graph for bucket {bucket_key} (max_position: {max_position})")

        # Create new graph for this bucket
        self._bucket_graphs[bucket_key] = torch.cuda.CUDAGraph()

        # Initialize static tensors dictionary for this bucket
        static_tensors = {}

        # Clone static tensors for this bucket
        static_tensors["speech_embedding_cache"] = speech_embedding_cache.clone()
        static_tensors["output_logits"] = output_logits.clone()
        static_tensors["i_tensor"] = i_tensor.clone()
        static_tensors["batch_idx"] = batch_idx.clone()
        static_tensors["speech_pos_embedding_cache"] = (
            speech_pos_embedding_cache.clone()
        )
        static_tensors["generated_ids"] = generated_ids
        static_tensors["cfg_weight"] = cfg_weight
        static_tensors["temperature"] = temperature
        static_tensors["stride_length"] = stride_length
        static_tensors["max_position"] = bucket_key[0]  # Use base bucket (sequence length)

        # Capture the graph
        with torch.inference_mode():
            with torch.cuda.graph(self._bucket_graphs[bucket_key]):
                static_tensors["out_1"], static_tensors["out_2"] = self.generate_token(
                    static_tensors["speech_embedding_cache"],
                    static_tensors["output_logits"],
                    static_tensors["i_tensor"],
                    static_tensors["batch_idx"],
                    static_tensors["speech_pos_embedding_cache"],
                    static_tensors["generated_ids"],
                    static_tensors["cfg_weight"],
                    static_tensors["temperature"],
                    self.repetition_penalty_processor,
                    self.min_p_warper,
                    self.top_p_warper,
                    self.patched_model,
                    self.kv_cache,  # Shared kv_cache
                    static_tensors["stride_length"],
                    static_tensors["max_position"],
                )

        # Store static tensors for this bucket
        self._bucket_static_tensors[bucket_key] = static_tensors
        self._captured_buckets.add(bucket_key)

    def __call__(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        repetition_penalty_processor: Any = None,
        min_p_warper: Any = None,
        top_p_warper: Any = None,
        patched_model: Any = None,
        kv_cache: Any = None,
        stride_length: int = 1,
        max_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_bucket = max_position or TOKEN_LIMIT
        cfg_mode = 1 if cfg_weight > 0.0 else 0
        bucket_key = (base_bucket, cfg_mode)

        # Check if we need to capture a graph for this bucket
        if bucket_key not in self._captured_buckets:
            self._capture_graph_for_bucket(
                bucket_key,
                speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                speech_pos_embedding_cache,
                generated_ids,
                cfg_weight,
                temperature,
                stride_length,
                max_position,
            )
        else:
            # Update static tensors for this bucket and replay
            static_tensors = self._bucket_static_tensors[bucket_key]

            static_tensors["speech_embedding_cache"].copy_(speech_embedding_cache)
            static_tensors["output_logits"].copy_(output_logits)
            static_tensors["i_tensor"].copy_(i_tensor)
            static_tensors["batch_idx"].copy_(batch_idx)
            static_tensors["speech_pos_embedding_cache"].copy_(
                speech_pos_embedding_cache
            )
            static_tensors["generated_ids"].copy_(generated_ids)
            static_tensors["cfg_weight"] = cfg_weight
            static_tensors["temperature"] = temperature
            static_tensors["stride_length"] = stride_length
            static_tensors["max_position"] = max_position

            # Replay the graph for this bucket
            self._bucket_graphs[bucket_key].replay()

        # Return outputs from the appropriate bucket
        static_tensors = self._bucket_static_tensors[bucket_key]
        return (
            static_tensors["out_1"],
            static_tensors["out_2"],
            static_tensors["generated_ids"],
        )

    def reset(self, bucket_key: Optional[Union[int, Tuple[int, int]]] = None) -> None:
        if bucket_key is not None:
            # Reset specific bucket
            if bucket_key in self._bucket_graphs:
                del self._bucket_graphs[bucket_key]
            if bucket_key in self._bucket_static_tensors:
                del self._bucket_static_tensors[bucket_key]
            self._captured_buckets.discard(bucket_key)
            print(f"Reset bucket {bucket_key}")
        else:
            # Reset all buckets
            self._bucket_graphs.clear()
            self._bucket_static_tensors.clear()
            self._captured_buckets.clear()
            print("Reset all buckets")

    def mark_new_generation(self):
        self.kv_cache.reset()
