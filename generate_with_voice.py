import random
from typing import List, Union, Optional, Tuple
import numpy as np
import torch
import torchaudio as ta
import ffmpeg
from src.chatterbox.tts import ChatterboxTTS
import os
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = ChatterboxTTS.from_pretrained(DEVICE)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_and_save(text: str, audio_prompt_path: str, output_path: str, exaggeration: float, temperature: float, seed_num: int, cfgw: float) -> None:    
    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    
    ta.save(output_path, wav, model.sr)

def concatenate_wav_files(
    wav_files: List[str], 
    output_path: str,
    sample_rate: str,
    format: str = "mp3",
    bitrate: str = "128k",
    trim_silence: bool = True,
    silence_threshold: str = "-50dB",
    silence_duration: str = "0.1",
    gap_duration: float = 0.5
):
    """
    Concatenate multiple WAV files into a single MP3 or M4A file with silence trimming and gaps.
    
    Args:
        wav_files: List of paths to WAV files to concatenate
        output_path: Path for the output file (MP3 or M4A)
        format: Output format - "mp3" or "m4a" (default: "mp3")
        bitrate: Audio bitrate (default: "128k")
        sample_rate: Resample to this rate (optional, keeps original if None)
        trim_silence: Whether to trim silence from end of files (default: True)
        silence_threshold: Volume threshold for silence detection (default: "-50dB")
        silence_duration: Minimum duration of silence to trim (default: "0.1" seconds)
        gap_duration: Seconds of silence to insert between files (default: 0.5)
    
    Requirements:
        pip install ffmpeg-python
        
    Example:
        wav_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
        concatenate_wav_files(wav_files, "combined.mp3", gap_duration=1.0)
    """
    
    # Validate inputs
    if not wav_files:
        raise ValueError("No WAV files provided")
    
    for wav_file in wav_files:
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")
    
    # Set codec based on format
    if format.lower() == "mp3":
        codec = "libmp3lame"
    elif format.lower() == "m4a":
        codec = "aac"
    else:
        raise ValueError("Format must be 'mp3' or 'm4a'")
    
    try:
        processed_inputs = []
        
        for i, wav_file in enumerate(wav_files):
            # Start with the input file
            input_stream = ffmpeg.input(str(wav_file))
            
            # Apply silence trimming if requested
            if trim_silence:
                # Use silenceremove filter to trim silence from the end
                input_stream = ffmpeg.filter(
                    input_stream, 
                    'silenceremove',
                    stop_periods=1,
                    stop_threshold=silence_threshold,
                    stop_silence=silence_duration
                )
            
            processed_inputs.append(input_stream)
            
            # Add gap between files (except after the last file)
            if i < len(wav_files) - 1 and gap_duration > 0:
                # Generate silence
                silence = ffmpeg.input(
                    f'anullsrc=channel_layout=mono:sample_rate=44100',
                    f='lavfi',
                    t=gap_duration
                )
                processed_inputs.append(silence)
        
        # Concatenate all inputs (audio files + silence gaps)
        joined = ffmpeg.concat(*processed_inputs, v=0, a=1)
        
        # Build output with codec and bitrate
        output_args = {
            'acodec': codec,
            'audio_bitrate': bitrate
        }
        
        # Add sample rate if specified
        if sample_rate:
            output_args['ar'] = sample_rate
        
        # Create output stream
        out = ffmpeg.output(joined, str(output_path), **output_args)
        
        # Run the concatenation
        ffmpeg.run(out, overwrite_output=True, quiet=True)
        
        print(f"Successfully concatenated {len(wav_files)} files to {output_path}")
        if trim_silence:
            print(f"Trimmed silence below {silence_threshold}")
        if gap_duration > 0:
            print(f"Added {gap_duration}s gaps between files")
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg error during concatenation: {error_msg}")


# Advanced function with more granular silence control
def concatenate_with_advanced_silence_control(
    wav_files: List[str],
    output_path: str,
    trim_start: bool = False,
    trim_end: bool = True,
    start_threshold: str = "-50dB",
    end_threshold: str = "-50dB",
    gap_duration: float = 0.5,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    debug: bool = False
):
    """
    Advanced concatenation with separate start/end silence trimming and fades.
    
    Args:
        trim_start: Trim silence from beginning of files
        trim_end: Trim silence from end of files  
        start_threshold: Threshold for start trimming
        end_threshold: Threshold for end trimming
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds
        debug: Print debug information
    """
    
    if debug:
        print(f"Processing {len(wav_files)} files...")
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                # Get file info
                probe = ffmpeg.probe(wav_file)
                duration = float(probe['streams'][0]['duration'])
                print(f"  {wav_file}: {duration:.2f} seconds")
            else:
                print(f"  {wav_file}: FILE NOT FOUND")
    
    processed_inputs = []
    
    for i, wav_file in enumerate(wav_files):
        input_stream = ffmpeg.input(str(wav_file))
        
        # Trim silence from start if requested
        if trim_start:
            input_stream = ffmpeg.filter(
                input_stream,
                'silenceremove',
                start_periods=1,
                start_threshold=start_threshold,
                start_silence=0.1
            )
        
        # Trim silence from end if requested  
        if trim_end:
            input_stream = ffmpeg.filter(
                input_stream,
                'silenceremove', 
                stop_periods=1,
                stop_threshold=end_threshold,
                stop_silence=0.1
            )
        
        # Add fades if requested
        if fade_in > 0:
            input_stream = ffmpeg.filter(input_stream, 'afade', type='in', duration=fade_in)
        if fade_out > 0:
            input_stream = ffmpeg.filter(input_stream, 'afade', type='out', duration=fade_out)
            
        processed_inputs.append(input_stream)
        
        # Add gap between files (not after last file)
        if i < len(wav_files) - 1 and gap_duration > 0:
            # Get sample rate from the current file to match
            probe = ffmpeg.probe(wav_file)
            sample_rate = int(probe['streams'][0]['sample_rate'])
            
            silence = ffmpeg.input(
                f'anullsrc=channel_layout=mono:sample_rate={sample_rate}',
                f='lavfi', 
                t=gap_duration
            )
            processed_inputs.append(silence)
    
    # Concatenate and output
    joined = ffmpeg.concat(*processed_inputs, v=0, a=1)
    
    # Run with error output for debugging
    out = ffmpeg.output(joined, output_path, acodec='libmp3lame', audio_bitrate='192k')
    
    if debug:
        print("Running FFmpeg...")
        ffmpeg.run(out, overwrite_output=True)  # Don't suppress output for debugging
    else:
        ffmpeg.run(out, overwrite_output=True, quiet=True)

def concatenate_wav_files_safe(
    wav_files: List[str],
    output_path: str,
    gap_duration: float = 0.5
):
    """
    Safe concatenation without any silence trimming - just join files with gaps.
    Use this to test if the issue is with silence detection.
    """
    processed_inputs = []
    
    for i, wav_file in enumerate(wav_files):
        # Just add the file as-is, no processing
        input_stream = ffmpeg.input(str(wav_file))
        processed_inputs.append(input_stream)
        
        # Add gap
        if i < len(wav_files) - 1 and gap_duration > 0:
            # Get sample rate to match
            probe = ffmpeg.probe(wav_file)
            sample_rate = int(probe['streams'][0]['sample_rate'])
            
            silence = ffmpeg.input(
                f'anullsrc=channel_layout=mono:sample_rate={sample_rate}',
                f='lavfi', 
                t=gap_duration
            )
            processed_inputs.append(silence)
    
    # Concatenate
    joined = ffmpeg.concat(*processed_inputs, v=0, a=1)
    out = ffmpeg.output(joined, output_path, acodec='libmp3lame', audio_bitrate='192k')
    ffmpeg.run(out, overwrite_output=True, quiet=True)
    
    print(f"Safe concatenation complete: {output_path}")



# Debug function to test individual files
def debug_wav_files(wav_files: List[str]):
    """
    Debug function to check your WAV files and detect potential issues.
    """
    print("=== WAV File Analysis ===")
    total_duration = 0
    all_volumes = []
    
    for i, wav_file in enumerate(wav_files):
        if not os.path.exists(wav_file):
            print(f"❌ File {i+1}: {wav_file} - NOT FOUND")
            continue
            
        try:
            probe = ffmpeg.probe(wav_file)
            stream = probe['streams'][0]
            duration = float(stream['duration'])
            sample_rate = int(stream['sample_rate'])
            channels = int(stream['channels'])
            
            total_duration += duration
            
            print(f"\n✅ File {i+1}: {os.path.basename(wav_file)}")
            print(f"   Duration: {duration:.2f}s ({duration/60:.1f} min)")
            print(f"   Sample Rate: {sample_rate} Hz")
            print(f"   Channels: {channels}")
            
            # Get detailed volume analysis
            result = (
                ffmpeg
                .input(wav_file)
                .filter('volumedetect')
                .output('pipe:', format='null')
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Parse volume information from stderr
            stderr = result[1].decode() if result[1] else ""
            
            mean_vol = None
            max_vol = None
            
            for line in stderr.split('\n'):
                if 'mean_volume:' in line:
                    try:
                        mean_vol = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                        print(f"   📊 Mean Volume: {mean_vol:.1f} dB")
                    except:
                        pass
                elif 'max_volume:' in line:
                    try:
                        max_vol = float(line.split('max_volume:')[1].split('dB')[0].strip())
                        print(f"   📊 Max Volume: {max_vol:.1f} dB")
                    except:
                        pass
            
            if mean_vol is not None and max_vol is not None:
                all_volumes.append({'file': wav_file, 'mean': mean_vol, 'max': max_vol})
                
                # Give recommendations based on volume levels
                if mean_vol < -50:
                    print(f"   ⚠️  Very quiet audio (mean: {mean_vol:.1f}dB) - may be detected as silence")
                elif mean_vol < -30:
                    print(f"   ℹ️  Quiet audio (mean: {mean_vol:.1f}dB)")
                else:
                    print(f"   ✅ Good volume levels")
            
        except Exception as e:
            print(f"❌ File {i+1}: {wav_file} - ERROR: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total expected duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    if all_volumes:
        mean_volumes = [v['mean'] for v in all_volumes]
        max_volumes = [v['max'] for v in all_volumes]
        
        overall_mean = sum(mean_volumes) / len(mean_volumes)
        overall_max = max(max_volumes)
        quietest_mean = min(mean_volumes)
        
        print(f"\n📈 VOLUME ANALYSIS:")
        print(f"   Overall average volume: {overall_mean:.1f} dB")
        print(f"   Loudest peak across all files: {overall_max:.1f} dB")
        print(f"   Quietest file average: {quietest_mean:.1f} dB")
        
        print(f"\n💡 SILENCE THRESHOLD RECOMMENDATIONS:")
        if quietest_mean < -60:
            recommended_threshold = quietest_mean - 10
            print(f"   Use threshold: {recommended_threshold:.0f}dB or lower")
            print(f"   Very quiet audio detected - use gentle silence detection")
        elif quietest_mean < -40:
            recommended_threshold = -70
            print(f"   Use threshold: {recommended_threshold}dB")
            print(f"   Moderately quiet audio - standard settings should work")
        else:
            recommended_threshold = -50
            print(f"   Use threshold: {recommended_threshold}dB")
            print(f"   Good volume levels - default settings should work")
        
        print(f"\n🔧 SUGGESTED COMMAND:")
        print(f"concatenate_wav_files(")
        print(f"    your_files,")
        print(f"    'output.mp3',")
        print(f"    trim_silence=True,")
        print(f"    silence_threshold='{recommended_threshold}dB',")
        print(f"    silence_duration='0.5',  # Adjust as needed")
        print(f"    gap_duration=0.5")
        print(f")")
    
    return total_duration