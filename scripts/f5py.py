import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)

# ── USER CONFIG ────────────────────────────────────────────────────────────────
config_path    = "infer/examples/basic/basic.toml"
model          = "F5TTS_v1_Base"
model_cfg_path = None  # e.g. "path/to/your/model.yaml", or leave None to use default from config
ckpt_file      = ""    # leave blank to pull from HF cache
vocab_file     = ""    # leave blank to use default
ref_audio      = "data/15sec.wav"
ref_text       = (
    "Fuck your phone. Stop texting all the time. "
    "Look up from your phone and breathe. Release yourself."
)
gen_text       = (
    "I am not feeling it. This is it. There is no reconceptualizing."
)
gen_file       = ""    # if set, will override gen_text by loading from this file
output_dir     = "tests"
output_file    = f"infer_cli_{datetime.now():%Y%m%d_%H%M%S}.wav"
save_chunk     = False
remove_silence = False
load_vocoder_from_local = False
vocoder_name   = None  # "vocos" or "bigvgan" or None to use default from config
# ────────────────────────────────────────────────────────────────────────────────

# load config
config = tomli.load(open(config_path, "rb"))

# resolve parameters (fall back to config defaults where applicable)
model_cfg_path = model_cfg_path or config.get("model_cfg", None)
ckpt_file      = ckpt_file      or config.get("ckpt_file", "")
vocab_file     = vocab_file     or config.get("vocab_file", "")
gen_file       = gen_file       or config.get("gen_file", "")
save_chunk     = save_chunk     or config.get("save_chunk", False)
remove_silence = remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name   = vocoder_name   or config.get("vocoder_name", mel_spec_type)
target_rms     = config.get("target_rms", target_rms)
cross_fade_duration = config.get("cross_fade_duration", cross_fade_duration)
nfe_step       = config.get("nfe_step", nfe_step)
cfg_strength   = config.get("cfg_strength", cfg_strength)
sway_sampling_coef = config.get("sway_sampling_coef", sway_sampling_coef)
speed          = config.get("speed", speed)
fix_duration   = config.get("fix_duration", fix_duration)
device         = config.get("device", device)

# if user pointed at example paths inside the package, fix them
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(ref_audio))
if gen_file and "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(gen_file))
if "voices" in config:
    for v in config["voices"].values():
        if "infer/examples/" in v.get("ref_audio", ""):
            v["ref_audio"] = str(files("f5_tts").joinpath(v["ref_audio"]))

# if using a gen_file, load its text
if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()

# prepare output paths
wave_path = Path(output_dir) / output_file
if save_chunk:
    chunk_dir = Path(output_dir) / f"{wave_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

# load vocoder
if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
else:
    vocoder_local_path = None

vocoder = load_vocoder(
    vocoder_name=vocoder_name,
    is_local=load_vocoder_from_local,
    local_path=vocoder_local_path,
    device=device,
)

# load TTS model
model_cfg = OmegaConf.load(
    model_cfg_path
    or str(files("f5_tts").joinpath(f"configs/{model}.yaml"))
)
ModelClass = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    else:
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name, ckpt_step = "E2-TTS", 1200000

if not ckpt_file:
    ckpt_file = str(
        cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}")
    )

print(f"Loading model {model} checkpoint…")
ema_model = load_model(
    ModelClass,
    model_cfg.model.arch,
    ckpt_file,
    mel_spec_type=vocoder_name,
    vocab_file=vocab_file,
    device=device,
)


def generate_tts(input_text, output_dir="tests", output_file=None, ref_audio=ref_audio, ref_text=None):
    """
    Generate text-to-speech audio from input text.
    
    Args:
        input_text (str): Text to convert to speech
        output_dir (str): Directory to save the output file (default: "tests")
        output_file (str): Output filename (default: auto-generated based on timestamp)
        ref_audio (str): Reference audio file (default: "15sec.wav")
        ref_text (str): Reference text (default: predefined text)
    
    Returns:
        str: Path to the generated audio file
    """
    if ref_text is None:
        ref_text = (
            "Fuck your phone. Stop texting all the time. "
            "Look up from your phone and breathe. Release yourself."
        )
    
    gen_text = input_text
    
    if output_file is None:
        output_file = f"infer_cli_{datetime.now():%Y%m%d_%H%M%S}.wav"
    
    # assemble voices dict
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    if "voices" in config:
        voices.update(config["voices"])
        voices["main"] = main_voice

    # preprocess all references
    for name, v in voices.items():
        v["ref_audio"], v["ref_text"] = preprocess_ref_audio_text(
            v["ref_audio"], v["ref_text"]
        )

    # break text into per‑voice chunks
    reg1 = r"(?=\[\w+\])"
    reg2 = r"\[(\w+)\]"
    chunks = re.split(reg1, gen_text)

    segments = []
    for chunk in chunks:
        txt = chunk.strip()
        if not txt:
            continue
        m = re.match(reg2, txt)
        if m:
            voice = m.group(1)
            txt = re.sub(reg2, "", txt).strip()
        else:
            voice = "main"

        if voice not in voices:
            print(f"Unknown voice '{voice}', using main.")
            voice = "main"

        seg, sr, _ = infer_process(
            voices[voice]["ref_audio"],
            voices[voice]["ref_text"],
            txt,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
        segments.append(seg)

        if save_chunk:
            name = txt[:200].replace(" ", "_")
            sf.write(str(chunk_dir / f"{len(segments)-1}_{name}.wav"), seg, sr)

    # concatenate and write
    final = np.concatenate(segments) if segments else np.array([], dtype=np.float32)
    os.makedirs(output_dir, exist_ok=True)
    wave_path = Path(output_dir) / output_file
    sf.write(str(wave_path), final, sr)
    if remove_silence:
        remove_silence_for_generated_wav(str(wave_path))
    print(f"Written output to {wave_path}")
    return str(wave_path)

if __name__ == "__main__":
    test_text = "This is a test of the TTS system."
    generated_file = generate_tts(test_text)
    print(f"Generated file: {generated_file}")
