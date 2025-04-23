import json
import os
import requests
from time import sleep
from tqdm import tqdm

API_KEY = ""  # Add your Gemini API key here
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

with open("cleaned_output2_corrected.json", "r") as f:
    data = json.load(f)

output_path = "cleaned_output2_gemini_corrected.json"
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        processed_data = json.load(f)
    start_index = len(processed_data)
    print(f"Resuming from index {start_index}")
else:
    processed_data = []
    start_index = 0

def correct_grammar(text: str) -> str:
    """
    Sends the given text to Gemini API with a grammar correction prompt.
    Returns the corrected text.
    """
    prompt = (
        "You are a grammar correction assistant. "
        "Please correct the grammar of the following text, preserving the original line breaks and formatting:\n\n"
        f"{text}"
    )
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    url = f"{API_URL}?key={API_KEY}"
    
    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    
    result = response.json()
    corrected = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    print(f"Original: {text}\nCorrected: {corrected}\n")
    return corrected

total_items = len(data)
print(f"Processing {total_items} items...")

for i, entry in enumerate(tqdm(data[start_index:], desc="Correcting grammar", initial=start_index, total=total_items)):
    original_text = entry["text"]
    corrected_text = correct_grammar(original_text)
    entry["text"] = corrected_text
    processed_data.append(entry)
    
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    sleep(10)

print(f"Completed! Saved corrected JSON to {output_path}")
