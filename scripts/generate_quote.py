from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from f5py import generate_tts
from stitch import create_music_speech_mix

adapter_path = "./checkpoints/epoch-11"
base_model = "mistralai/Mistral-7B-Instruct-v0.3"

# Load base model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Generate a Masood Boomgaard style quote:"

output = generator(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    num_return_sequences=1
)

print("Masood Boomgaard wisdom incoming:")
text = output[0]["generated_text"].replace(prompt, "")
print(f"Generated quote: {text}")
output_path = generate_tts(input_text=text)
final_audio_path = create_music_speech_mix(speech_path=output_path)

print(f"Final audio here: {final_audio_path}")
