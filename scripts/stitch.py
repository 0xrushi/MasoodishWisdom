from pydub import AudioSegment
import random

def create_music_speech_mix(speech_path, music_path="data/instrumental.wav", output_path="data/output.wav"):
    """
    Combine speech audio with background music at random position.
    
    Args:
        speech_path (str): Path to speech WAV file
        music_path (str): Path to music WAV file (default: data/instrumental.wav)
        output_path (str): Path for output WAV file (default: output.wav)
    
    Returns:
        tuple: (start_time_seconds, end_time_seconds)
    """
    speech = AudioSegment.from_wav(speech_path)
    music = AudioSegment.from_wav(music_path)

    # Durations (in milliseconds)
    speech_len = len(speech)
    music_len = len(music)

    if speech_len > music_len:
        raise ValueError("Speech audio is longer than background music!")

    # Choose a random start point
    max_start = music_len - speech_len
    start_ms = random.randint(0, max_start)

    # Extract the music segment
    music_segment = music[start_ms : start_ms + speech_len]

    # Lower volume by 10db
    music_segment = music_segment - 10

    # Overlay speech on music
    combined = music_segment.overlay(speech)

    combined.export(output_path, format="wav")
    
    return output_path

if __name__ == "__main__":
    output_path = create_music_speech_mix("tests/infer_cli_basic.wav")
    print(f"Created {output_path} using music")
