import json
import re

def clean_json_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully cleaned {input_file} and saved to {output_file}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_file = "cleaned_output2_gemini_corrected.json" 
    output_file = "cleaned_final.json"
    clean_json_file(input_file, output_file)