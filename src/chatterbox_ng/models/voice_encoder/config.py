class VoiceEncConfig:
    num_mels = 40
    sample_rate = 16000
    speaker_embed_size = 256
    ve_hidden_size = 256
    flatten_lstm_params = False
    n_fft = 400
    hop_size = 160
    win_size = 400
    fmax = 8000
    fmin = 0
    preemphasis = 0.
    mel_power = 2.0
    mel_type = "amp"
    normalized_mels = False
    ve_partial_frames = 160
    ve_final_relu = True
    stft_magnitude_min = 1e-4
