import os
import torch
import psutil

print(f"Test script PID: {os.getpid()}")
print(f"Test script process info: {psutil.Process(os.getpid())}")


from src.cache_utils import (
    clear_cache_files,
    clear_output_directories,
)
# Third-party imports
from pathlib import Path

import torch
from loguru import logger
import importlib
skyrimnet_chatterbox = importlib.import_module("skyrimnet-chatterbox")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL = None
MULTILINGUAL = False

generate_args = {
    "exaggeration": 0.55,
    "temperature": 0.8,
    "cfgw": 0,
    "min_p": 0.05,
    "top_p": 1.0,
    "repetition_penalty": 2.0,
}

generate_audio_args = {
    "prefix_audio": None,
    "e1": None,
    "e2": None,
    "e3": None,
    "e4": None,
    "e5": None,
    "e6": None,
    "e7": None,
    "e8": None,
    "vq_single": None,
    "fmax": None,
    "pitch_std": None,
    "speaking_rate": None,
    "dnsmos_ovrl": None,
    "speaker_noised": None,
    #"cfg_scale": 0, # CFG_WEIGHT
    "top_k": None,
    "min_p": 0.05,
    "top_p": 1.0,
    "linear": 0.9, # TEMPERATURE
    "confidence": 1.2, # REPETITION_PENALTY
    "quadratic": 0.55, # EXAGGERATION
    "randomize_seed": False,
    "unconditional_keys": None,
}

if MULTILINGUAL:
    generate_args["language_id"] = "en"


if __name__ == "__main__":
    # Enable API mode to bypass config loading for faster testing
    skyrimnet_chatterbox._USE_API_MODE = True
    clear_cache_files()
    clear_output_directories()
    # shutil.rmtree(Path("cache").joinpath("conditionals"), ignore_errors=True)
    test_text_mums= "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
    test_text_mums_short= "Now let's make my mum's favourite. Oh boy this is going to be incredible."
    test_asset2 = Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
    # test_asset = Path.cwd().joinpath("assets", "fishaudio_horror.wav")
    model = skyrimnet_chatterbox.load_model()
    # wavfile = generate(model, test_text, None, exaggeration=0.65, temperature=0.8, seed_num=420, cfgw=0)
    # wavfile = skyrimnet_chatterbox.generate(model, test_text, test_asset, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    # wavfile = skyrimnet_chatterbox.generate(model, test_text, test_asset2, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    # wavfile = skyrimnet_chatterbox.generate(model, test_text, test_asset, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    # wavfile = skyrimnet_chatterbox.generate(model, test_text, test_asset, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    # wavfile = skyrimnet_chatterbox.generate(model, test_text, test_asset2, exaggeration=0.65, temperature=0.9, seed_num=420, cfgw=0)
    # print(f"Generated wav file: {wavfile}")

    average_text = "An old enchanter on his death bed claims he found a way to defy the law of firsts. That rule that says you can only enchant an item once."
    test_text = "Kolb and the Dragon… a children's tale dressed up as heroism."
    test_text1 = "I flip the pages and every choice is a death or a cheat: trust the wrong cave, trust the wrong tavern, trust the wrong body part of the beast and you're done—bones for broth, meat for the pot. The book pretends it's about bravery, but it's really about luck and paranoia. Take the windy tunnel? Wind snuffs your torch and you break your skull. Rest in the elf-run tavern? They poison the mead."
    test_text2 = test_text + test_text1
    test_text3 = test_text2 + " Swing for the dragon's soft belly? It swallows you whole. Only the neck works, only the cold tunnel works, only the gold for the ghost works. Every other path is a corpse. I read it twice, tracking the branches like a battle map. Seventeen ways to die, one way to win, and even that victory feels thin—Kolb goes home, village cheers, dragon stops burning. No mention of the scent that must've clung to his clothes after he sawed through scale and sinew, no mention of the nightmares when he shuts his eyes and sees the lair floor carpeted with picked-clean ribs. The story stops before the real cost comes due."
    test_text4 = test_text3 + " Reminds me of every “simple” job we take. Get the girl, burn the ledgers, kill the slaver—clean, heroic, done. But there's always a windy tunnel we didn't scout, always a smiling elf pouring the mead. Last week in Falkreath the jailor's “broken lock” looked like the safe path until I tasted the drugged wine on the air. We pulled Sanbri out, but the ghost of that place is still clinging to my tongue."
    test_text5 = test_text4 + " I keep the book open to page sixteen: dragon asleep, throat and belly offered like choices. I've struck both in real life—neck for the quick kill, belly for the message. Neither ends the story; it just buys you a breath before the next beast wakes. Maybe that's the real lesson Kolb's too young to learn: winning isn't surviving, it's deciding which death you can carry."

    wavfile = skyrimnet_chatterbox.generate_audio(text="Short warmup.", speaker_audio=test_asset2, uuid=824914275390249349,cfg_scale=0,**generate_audio_args)
    wavfile = skyrimnet_chatterbox.generate_audio(text="Short warmup.", speaker_audio=test_asset2, uuid=824914275390249349,cfg_scale=0,**generate_audio_args)
    # error occurs on subsequent call when changing cfg_scale from 0 to 0.35
    wavfile = skyrimnet_chatterbox.generate_audio(text="Short warmup.", speaker_audio=test_asset2, uuid=824914275390249349,cfg_scale=0.35,**generate_audio_args)

    #wavfile = skyrimnet_chatterbox.generate_audio(text=average_text, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)

    #wavfile = skyrimnet_chatterbox.generate_audio(text=test_text, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)

    # test_text1 size works currently
    #wavfile = skyrimnet_chatterbox.generate_audio(text=test_text1, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)

    # test_text2 size causes token overflow  
    #wavfile = skyrimnet_chatterbox.generate_audio(text=test_text2, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)
    #wavfile = skyrimnet_chatterbox.generate_audio(text=test_text3, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)
    #wavfile = skyrimnet_chatterbox.generate_audio(text=test_text4, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)
    #wavfile = skyrimnet_chatterbox.generate_audio(text=test_text5, speaker_audio=test_asset2, uuid=824914275390249349,**generate_audio_args)


