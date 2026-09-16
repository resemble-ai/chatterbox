# Experimental Turbo alignment capture

`TurboAlignmentCapture` observes the existing GPT-2 Q/K projections during one
Turbo generation. It does not switch attention kernels, request full attention
matrices, modify logits, or run another neural model. It installs temporary
hooks only on explicitly selected layers and removes them on normal or failed
exit. Default `generate()` behavior and return values remain unchanged.

```sh
python example_turbo_alignment.py --text 'First came a boom, then another boom, and finally silence.' --audio-prompt /path/to/reference.wav --output /tmp/turbo-trace
python -m unittest discover -s tests -p test_turbo_alignment.py
```

The example's heads `(4, 6)`, `(5, 14)`, `(8, 4)` showed ordered text progression
in a small Turbo exploration. They are diagnostic examples, not a calibrated
or universal head selection. Nano and multilingual models were not validated.

## Output and limits

- The tensor has axes `[selected head, decoding step, normalized text token]`.
  Subword pieces and repeated words retain distinct token positions.
- Softmax is over the text keys only. Scores are **not** probabilities over the
  full context and cannot establish how much attention was paid to text versus
  the conditioning/audio context.
- The first row is the speech-BOS query used to predict the first speech token;
  subsequent rows observe successive decode queries. An EOS prediction may be
  present. Existing generation then filters invalid tokens and adds trailing
  silence before converting speech tokens to a waveform. These rows are not
  automatically waveform positions.
- Word-boundary accuracy, multilingual behavior, paralinguistic tags, long
  inputs, and failure/confidence thresholds remain unvalidated. In particular,
  do not use an attention argmax as a guaranteed audible word-end timestamp.
- Use one utterance and one active generation on the model. Match the exact
  normalized/truncated text-token count; the API does not infer it. Capture
  memory grows with the selected heads and decoding steps. `max_steps` bounds
  recording and raises on overflow; it defaults to the current Turbo limit plus
  its prefill prediction. Read the trace after leaving the context.

## Validation

CPU tests use a tiny randomly initialized GPT-2: exact hidden-output parity,
projected-QK numerical agreement, unchanged attention configuration, invalid
arguments, bounded recording, cleanup after exceptions and recorder reuse.
They require no checkpoint downloads.

A separate offline GPU check used the existing cached Turbo checkpoint on an
RTX 3060, PyTorch 2.6.0+cu124 and Transformers 5.2.0. After one warmup, three
seed-matched baseline/capture pairs produced **bit-identical waveform tensors**
(maximum absolute difference zero) and left zero hooks installed:

| Input | Audio seconds | Baseline seconds | Capture + extraction seconds |
| --- | ---: | ---: | ---: |
| First came a boom, then another boom, and finally silence. | 4.28 | 1.622 | 1.614 |
| The red boat passed the blue boat before sunset. | 2.96 | 1.168 | 1.169 |
| Hello! Really? Yes, really. | 2.32 | 0.945 | 0.935 |

These single short runs establish feasibility, not a speedup, statistical
performance bound, general output-parity guarantee or word-timing accuracy.
No reference voice, generated audio, checkpoint, credential or machine-specific
configuration is distributed with this change. No live TTS service was changed.


## Real playback follow-up

An opt-in reproducible checkpoint check now covers three sentences with seeds
968 and 969, including adjacent repeated words:

```sh
PYTHONPATH=src python scripts/validate_turbo_alignment.py --audio-prompt /path/to/reference.wav --output /tmp/turbo-validation
```

This performs six baseline/capture pairs using the same model and reference.
It asserts bit-identical waveform and vocoder tokens, hook cleanup, and the
observed speech-token/frame relationship. It saves audio, attention, tokenizer
offsets and diagnostic candidate times. It never runs a transcription model.
Output audio/reference assets should remain private unless their owner chooses
to publish them. This opt-in check requires the real checkpoint; ordinary CPU
unit tests still do not download one.

The six saved utterances were played through a browser with separate speech
and sound-effect sources scheduled on the same AudioContext clock. Effects
were placed at the second target word using head (4, 6)'s last argmax row and
40 ms per emitted speech token. Dedicated browser-stream loopback found all six
effect starts at the scheduled offsets, with reference correlations above
0.999994. All six baseline/capture waveform and vocoder-token pairs were
identical. The operator listened and explicitly accepted the audible timing
as the desired behavior for that soundboard application.

[Sanitized results](turbo-alignment-playback.json) preserve those measurements.
This is application-specific listener acceptance, not a universal word-boundary
accuracy claim. Approximate independent spectrogram inspection still showed
lead/lag variation (one candidate about 70–150 ms before the estimated acoustic
end). The browser probe has not yet been integrated into the live application.
No private voice, generated audio, or checkpoint is included in this repository.


## Optional timing return

```python
waveform, timing = model.generate(text, return_alignment=True)
```

The opt-in return contains `version: 1`, normalized `text`, `sample_rate`, and
`words` rows `[character_start, character_end, end_sample]`. It uses the tested
Turbo head (4, 6), ignores brief backward attention spikes, maps only speech
tokens retained by S3Gen, and excludes EOS/trailing silence from word timing.
Unexpected frame rates or missing text evidence omit timings; they do not
invent timestamps from text length. This remains an attention-derived timing
estimate with the listener-tested behavior described above. Only English Turbo
has been exercised. Existing callers still receive the waveform tensor alone.

The integrated option was also checked against default generation with seed
969: identical waveform, both repeated-word positions preserved (1.00 s and
2.08 s). A local HTTP wrapper carried metadata in an ancillary WAV RIFF chunk
while retaining PCM samples. No second model or extra synthesis is used to
compute the timing.
