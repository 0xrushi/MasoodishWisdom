import tomli
import pickle
import os
from importlib.resources import files
from f5_tts.infer.utils_infer import preprocess_ref_audio_text

def load_ref_weights(pkl_path="ref_weights.pkl"):
    """
    Load and return a dict of voices -> {"ref_audio", "ref_text"}.

    Args:
        pkl_path (str): Path to the pickle file.

    Returns:
        dict: Mapping voice names to preprocessed refs.
    """
    
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Ref weights pickle not found at {pkl_path}. Please run ref_utils.py first.")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def build_ref_weights(config_path="infer/examples/basic/basic.toml", output_pkl="data/ref_weights.pkl"):
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    def fix_path(path):
        if "infer/examples/" in path:
            return str(files("f5_tts").joinpath(path))
        return path

    ref_audio = fix_path(config.get("ref_audio", "data/15sec.wav"))
    ref_text = config.get("ref_text")
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    if "voices" in config:
        for name, v in config["voices"].items():
            voices[name] = {
                "ref_audio": fix_path(v.get("ref_audio")),
                "ref_text": v.get("ref_text"),
            }

    for v in voices.values():
        v["ref_audio"], v["ref_text"] = preprocess_ref_audio_text(
            v["ref_audio"], v["ref_text"]
        )

    with open(output_pkl, "wb") as f:
        pickle.dump(voices, f)
    print(f"Saved {output_pkl}")


if __name__ == "__main__":
    build_ref_weights()
