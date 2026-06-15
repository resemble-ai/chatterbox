import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

import torch

from chatterbox.streaming import audio_to_pcm_s16le, write_chunks_to_wav
from chatterbox.tts_turbo import ChatterboxTurboTTS


def player_command(sample_rate: int, latency_msec: int) -> list[str] | None:
    pw_cat = shutil.which("pw-cat")
    if pw_cat:
        return [
            pw_cat,
            "--playback",
            "--raw",
            "--format",
            "s16",
            "--rate",
            str(sample_rate),
            "--channels",
            "1",
            "--latency",
            f"{latency_msec}ms",
            "-",
        ]

    paplay = shutil.which("paplay") or shutil.which("pacat")
    if paplay:
        return [
            paplay,
            "--raw",
            "--format=s16le",
            f"--rate={sample_rate}",
            "--channels=1",
            f"--latency-msec={latency_msec}",
            "--stream-name=chatterbox-turbo-stream",
        ]

    ffplay = shutil.which("ffplay")
    if ffplay:
        return [
            ffplay,
            "-autoexit",
            "-nodisp",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-",
        ]

    return None


def stop_player(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
    except BrokenPipeError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Chatterbox Turbo TTS chunks.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", default="Hi, this is Chatterbox Turbo streaming audio chunks in real time.")
    parser.add_argument("--audio-prompt-path", default=None)
    parser.add_argument("--out", default="test-turbo-stream.wav")
    parser.add_argument("--chunk-tokens", type=int, default=24)
    parser.add_argument("--crossfade-ms", type=float, default=12.0)
    parser.add_argument("--play", action="store_true", help="Play chunks while generating.")
    parser.add_argument("--latency-msec", type=int, default=120)
    args = parser.parse_args()

    model = ChatterboxTurboTTS.from_pretrained(device=args.device)

    chunks = []
    player = None
    cmd = None

    try:
        stream = model.stream(
            args.text,
            audio_prompt_path=args.audio_prompt_path,
            chunk_tokens=args.chunk_tokens,
            crossfade_ms=args.crossfade_ms,
        )

        for chunk in stream:
            chunks.append(chunk)
            print(
                f"chunk={chunk.index} final={chunk.is_final} "
                f"samples={chunk.end_sample - chunk.start_sample} "
                f"duration={chunk.duration_seconds:.3f}s tokens={chunk.generated_tokens}",
                flush=True,
            )

            if args.play and player is None:
                cmd = player_command(chunk.sample_rate, args.latency_msec)
                if cmd is None:
                    print("No supported player found; continuing without live playback.", flush=True)
                else:
                    print(f"player={shlex.join(cmd)}", flush=True)
                    player = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            if player is not None and player.stdin is not None:
                try:
                    player.stdin.write(audio_to_pcm_s16le(chunk.audio))
                    player.stdin.flush()
                except BrokenPipeError:
                    print("Player stopped accepting audio; continuing generation.", flush=True)
                    player = None
    finally:
        stop_player(player)

    if not chunks:
        raise RuntimeError("no audio chunks generated")

    out = write_chunks_to_wav(Path(args.out), chunks)
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
