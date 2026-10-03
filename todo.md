# Source of Truth for Rust Translation

Instructions for agents:
Translate the python files below to rust, **one python file at a time**.
Use **no external libraries**; use **first principle math first**.
Check against the python file for parity, then mark that file for the next agent to know by checking the box.

## Files:

- [x] `example_for_mac.py`
- [x] `example_tts.py`
- [ ] `example_tts_turbo.py`
- [ ] `example_vc.py`
- [ ] `gradio_tts_app.py`
- [ ] `gradio_tts_turbo_app.py`
- [ ] `gradio_vc_app.py`
- [ ] `multilingual_app.py`
- [ ] `src/chatterbox/__init__.py`
- [ ] `src/chatterbox/models/__init__.py`
- [ ] `src/chatterbox/models/s3gen/__init__.py`
- [ ] `src/chatterbox/models/s3gen/configs.py`
- [ ] `src/chatterbox/models/s3gen/const.py`
- [x] `src/chatterbox/models/s3gen/decoder.py`
- [x] `src/chatterbox/models/s3gen/f0_predictor.py`
- [x] `src/chatterbox/models/s3gen/flow.py`
- [x] `src/chatterbox/models/s3gen/flow_matching.py`
- [x] `src/chatterbox/models/s3gen/hifigan.py`
- [x] `src/chatterbox/models/s3gen/matcha/decoder.py`
- [ ] `src/chatterbox/models/s3gen/matcha/flow_matching.py`
- [ ] `src/chatterbox/models/s3gen/matcha/text_encoder.py`
- [ ] `src/chatterbox/models/s3gen/matcha/transformer.py`
- [ ] `src/chatterbox/models/s3gen/s3gen.py`
- [ ] `src/chatterbox/models/s3gen/transformer/__init__.py`
- [x] `src/chatterbox/models/s3gen/transformer/activation.py`
- [x] `src/chatterbox/models/s3gen/transformer/attention.py`
- [x] `src/chatterbox/models/s3gen/transformer/convolution.py`
- [x] `src/chatterbox/models/s3gen/transformer/embedding.py`
- [x] `src/chatterbox/models/s3gen/transformer/encoder_layer.py`
- [x] `src/chatterbox/models/s3gen/transformer/positionwise_feed_forward.py`
- [x] `src/chatterbox/models/s3gen/transformer/subsampling.py`
- [x] `src/chatterbox/models/s3gen/transformer/upsample_encoder.py`
- [x] `src/chatterbox/models/s3gen/utils/class_utils.py`
- [x] `src/chatterbox/models/s3gen/utils/intmeanflow.py`
- [x] `src/chatterbox/models/s3gen/utils/mask.py`
- [x] `src/chatterbox/models/s3gen/utils/mel.py`
- [x] `src/chatterbox/models/s3gen/xvector.py`
- [ ] `src/chatterbox/models/s3tokenizer/__init__.py`
- [x] `src/chatterbox/models/s3tokenizer/s3tokenizer.py`
- [ ] `src/chatterbox/models/t3/__init__.py`
- [ ] `src/chatterbox/models/t3/inference/t3_hf_backend.py`
- [ ] `src/chatterbox/models/t3/llama_configs.py`
- [ ] `src/chatterbox/models/t3/modules/cond_enc.py`
- [ ] `src/chatterbox/models/t3/modules/learned_pos_emb.py`
- [ ] `src/chatterbox/models/t3/modules/perceiver.py`
- [ ] `src/chatterbox/models/t3/modules/t3_config.py`
- [ ] `src/chatterbox/models/t3/t3.py`
- [ ] `src/chatterbox/models/tokenizers/__init__.py`
- [ ] `src/chatterbox/models/tokenizers/tokenizer.py`
- [ ] `src/chatterbox/models/utils.py`
- [x] `src/chatterbox/models/voice_encoder/__init__.py`
- [x] `src/chatterbox/models/voice_encoder/config.py`
- [x] `src/chatterbox/models/voice_encoder/melspec.py`
- [x] `src/chatterbox/models/voice_encoder/voice_encoder.py`
- [ ] `src/chatterbox/mtl_tts.py`
- [ ] `src/chatterbox/tts.py`
- [ ] `src/chatterbox/tts_turbo.py`
- [ ] `src/chatterbox/vc.py`
