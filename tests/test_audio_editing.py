import numpy as np
from chatterbox.audio_editing import (
    splice_audios,
    trim_audio,
    insert_audio,
    delete_segment,
    crossfade,
)


def test_splice_and_trim():
    a = np.ones(1000)
    b = np.zeros(1000)
    joined = splice_audios([a, b])
    assert joined.size == 2000
    trimmed = trim_audio(joined, start_sec=0, end_sec=0.01, sr=100000)
    assert trimmed.size == 1000


def test_insert_and_delete():
    base = np.zeros(1000)
    ins = np.ones(100)
    inserted = insert_audio(base, ins, 0.5, sr=1000)
    assert inserted[500:600].sum() == 100
    deleted = delete_segment(inserted, 0.5, 0.6, sr=1000)
    assert deleted.size == 1000


def test_crossfade():
    a1 = np.zeros(100)
    a2 = np.ones(100)
    out = crossfade(a1, a2, duration_sec=0.01, sr=10000)
    expected = len(a1) + len(a2) - int(0.01 * 10000)
    assert out.size == expected

