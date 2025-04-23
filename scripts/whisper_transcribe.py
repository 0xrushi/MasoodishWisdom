from faster_whisper import WhisperModel
import json

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

file = "something.wav"
segments, info = model.transcribe(file, beam_size=5)

transcription_data = {
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": []
}

for segment in segments:
    transcription_data["segments"].append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })

output_file = file.rsplit(".", 1)[0] + "_transcription.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(transcription_data, f, indent=4, ensure_ascii=False)

print(f"Transcription saved to: {output_file}")