#chatterbox/src/chatterbox/models/s3gen/const.py

S3GEN_SR = 24000

# The ratio of audio samples to speech tokens.
# 1 token -> 2 mel frames (in CausalMaskedDiffWithXvec)
# 1 mel frame -> 480 audio samples (hop_size in hifigan)
# Total: 2 * 480 = 960
TOKEN_TO_WAV_RATIO = 960