from itertools import groupby

import numpy as np
import webrtcvad as _webrtcvad

from transforms.vad.vad_stream import VADStream
from transforms.wav_encoding import encode_pcm16

# The sample rate the algo can operate at
_WEBRTC_SAMPLE_RATES = np.array([8000, 16000, 32000, 48000])
# The algo operates with window sizes of 10, 20 and 30ms
_WEBRTC_WINDOW_SIZES_MS = (10, 20, 30)
# Greatest common divisor and lowest common multiple of the above
_WEBRTC_WINDOW_SIZES_MS_GCD = 10
_WEBRTC_WINDOW_SIZES_MS_LCM = 60


class WebRTCVADStream(VADStream):
    def __init__(self, sample_rate: int, aggressiveness=2, dilation_ms=40, min_voiced_region_ms=125):
        """
        :param sample_rate: sample rate of the wavs that will be passed
        :param aggressiveness: parameter for controlling the aggressiveness of the VAD algo. Possible values are 1,
        2 and 3. Higher means less regions will be detected as voiced.
        :param dilation_ms: pass a value greater than 0 to include regions directly preceding or succeeding voiced
        regions. Voiced regions will be expanded left and right by this value, in milliseconds.
        N.B.: this is a best effort parameter. When the output is requested as fast as the input is produced,
        it's impossible to foresee an upcoming voiced region. In that case, the dilation on the left of that region
        may not appear.
        :param min_voiced_region_ms: to exclude regions detected as speech that are considered too short, pass a value
        greater than 0. Voiced regions shorter than this value (prior to dilation) will be set as unvoiced.
        N.B.: this is also a best effort parameter. A region may be too short, but because VAD has not finished
        being computed at the end of that region, it won't be removed as it could potentially be large enough.
        """
        webrtc_sr = int(_WEBRTC_SAMPLE_RATES[np.argmin(np.abs(_WEBRTC_SAMPLE_RATES - sample_rate))])
        lcm_win_size = (_WEBRTC_WINDOW_SIZES_MS_LCM * webrtc_sr) // 1000
        self._gcd_win_size = (_WEBRTC_WINDOW_SIZES_MS_GCD * webrtc_sr) // 1000

        # webrtcvad.Vad is stateful, predictions will be impacted if a new instance is created halfway through an
        # audio. This is why we create them now.
        self._detectors = {win_size: _webrtcvad.Vad(mode=aggressiveness) for win_size in _WEBRTC_WINDOW_SIZES_MS}

        super().__init__(sample_rate, webrtc_sr, lcm_win_size, dilation_ms, min_voiced_region_ms)

    def _wav_vad(self, wav: np.ndarray) -> np.ndarray:
        pcm = encode_pcm16(wav)

        # Perform the VAD by ensembling the different window sizes
        win_vad = np.zeros(len(wav) // self._gcd_win_size, dtype=np.int32)
        for sub_win_size_ms in _WEBRTC_WINDOW_SIZES_MS:
            detector = self._detectors[sub_win_size_ms]
            sub_win_size_pcm = (2 * sub_win_size_ms * self.vad_sr) // 1000
            factor = sub_win_size_ms // _WEBRTC_WINDOW_SIZES_MS_GCD

            for i, win_start in enumerate(range(0, len(pcm), sub_win_size_pcm)):
                win_i_vad = detector.is_speech(pcm[win_start:win_start + sub_win_size_pcm], self.vad_sr)
                win_vad[i * factor:(i + 1) * factor] += win_i_vad
        win_vad = win_vad > (len(_WEBRTC_WINDOW_SIZES_MS) // 2)

        # Convert the output to regions
        regions = np.diff(win_vad, prepend=0, append=0).nonzero()[0].reshape(-1, 2)
        regions = regions * (len(wav) // len(win_vad))

        return regions


def webrtc_vad(wav: np.ndarray, source_sr: int, aggressiveness=2, dilation_ms=40, min_voiced_region_ms=125):
    """
    Peforms Voice Activation Detection on a single audio. See WebrtcVADStream for more details.

    :return vad: a boolean numpy array of length equal to <wav>
    """
    vad_stream = WebRTCVADStream(source_sr, aggressiveness, dilation_ms, min_voiced_region_ms)
    vad_stream.feed(wav, close_input=True)
    if vad_stream.can_step():
        return vad_stream.step(len(wav))
    else:
        return np.zeros_like(wav, dtype=bool)


def split_on_silence(
    wav, sr, vad, thresholds_ms=[500, 300, 200, 100, 50], min_dur_s=1.5, max_split_dur_s=20, max_dur_s=30,
):
    """
    Split a wav into chunks, splitting on silence when the length of the silence exceeds a threshold.
    Args:
        wav: 1d-array
        sr: sample rate
        thresholds_ms: min length of silence to split on, clips are recursively split using values from this list until
            the resulting chunks are all within the min / max duration bounds
        min_dur_s: minimum duration of a chunk in seconds
        max_split_dur_s: segments above this length are continue to be split down with smaller thesholds
        max_dur_s: maximum duration of a chunk in seconds
    """
    assert isinstance(wav, np.ndarray) and wav.ndim == 1

    # unpack silence length thresholds
    thresh_ms, next_thresh_ms = (thresholds_ms + [0, 0])[:2]
    if thresh_ms <= 0:
        return [wav]

    # convert thresholds to samples
    max_split_dur_s = min(max_split_dur_s, max_dur_s)
    thresh = int(thresh_ms * sr / 1000)
    min_len = int(min_dur_s * sr)
    max_split_len = int(max_split_dur_s * sr)
    max_len = int(max_dur_s * sr)
    wav_len = len(wav)

    # detect regions of silence using groupby
    sil_regions = []
    for is_voiced, idxs in groupby(range(wav_len), key=vad.__getitem__):
        idxs = list(idxs)
        i = idxs[0]
        j = idxs[-1]
        j += 1
        n = j - i
        mid = (i + j) // 2

        # record split point if this is a long silence region
        if (not is_voiced) and n > thresh:
            sil_regions += [(
                min(mid, i + (0 if i == 0 else thresh // 2)),
                max(mid, j - (0 if j == wav_len else thresh // 2)),
            )]

    # invert silence regions to get voiced regions
    ptr = 0
    voiced_regions = []
    for i, j in sil_regions:
        if i > 0:
            voiced_regions += [(ptr, i)]
        ptr = j
    if ptr < wav_len:
        voiced_regions += [(ptr, wav_len)]

    # split the waveform into chunks using the detected content bounds and silence split points
    chunks = []
    for i, j in voiced_regions:
        chunk = wav[i:j]
        chunklen = len(chunk)

        # chunk is within bounds
        if chunklen < max_split_len:
            chunks += [chunk]

        # chunk is too long, attempt to split it recursively using threshold list
        elif next_thresh_ms > 0:
            chunks += split_on_silence(
                chunk, sr, vad[i:j], thresholds_ms=thresholds_ms[1:],
                min_dur_s=min_dur_s, max_dur_s=max_dur_s,
            )

        # NOTE: keeping chunks longer than `max_len` here, filtering is done below
        else:
            chunks += [chunk]

    # merge short chunks
    merged_chunks = []
    for chunk in chunks:
        chunklen = len(chunk)

        # chunk is too short, add it to the previous chunk if possible
        if chunklen == 0:
            continue

        elif chunklen < min_len:
            # NOTE: ignore the edge case where this would make the previous chunk too long, by just dropping this chunk
            if len(merged_chunks) > 0 and len(merged_chunks[-1]) + chunklen < max_len:
                merged_chunks[-1] = np.concatenate([merged_chunks[-1], chunk])

        elif chunklen < max_len:
            merged_chunks += [chunk]

        else:
            # TODO: keep long chunks as well? one benefit is to keep the adjascent ordering of chunks, for
            #   building paragraph-level datasets. However, this should rarely drop any clips, so it's probably okay.
            # merged_chunks += [chunk]
            pass
    chunks = merged_chunks

    return chunks
