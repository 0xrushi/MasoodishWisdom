from huggingface_hub import login
import os

token = os.environ.get("HUGGINGFACE_TOKEN")
login(token)

import gradio as gr
import spaces
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from f5py import generate_tts
from stitch import create_music_speech_mix
import traceback
import warnings

# Suppress NVML initialization warning
warnings.filterwarnings("ignore", message="Can't initialize NVML")
def initialize_model():
    adapter_path = "./checkpoints/epoch-11"
    base_model = "mistralai/Mistral-7B-Instruct-v0.3"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        offload_folder="offload/", 
        )
    model.config.use_cache = False

    model = PeftModel.from_pretrained(model, adapter_path, offload_folder="offload/")
    model.eval()
    
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

@spaces.GPU()
def generate_quote(temperature, top_p, max_length):
    try:
        def initialize_model():
            adapter_path = "./checkpoints/epoch-11"
            base_model = "mistralai/Mistral-7B-Instruct-v0.3"
            
            # Check CUDA availability more thoroughly
            device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
            print(f"Using device: {device}")
            
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        generator = initialize_model()
        prompt = "Generate a Masood Boomgaard style quote:"
        
        output = generator(
            prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            num_return_sequences=1
        )
        
        text = output[0]["generated_text"].replace(prompt, "")
        output_path = generate_tts(input_text=text)
        final_audio_path = create_music_speech_mix(speech_path=output_path)
        
        return text, final_audio_path, None
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, None, error_msg

with gr.Blocks() as demo:
    gr.Markdown("# MasoodishWisdom")
    
    with gr.Row():
        with gr.Column():
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.1, value=0.9,
                label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.05, value=0.95,
                label="Top-p"
            )
            max_length = gr.Slider(
                minimum=50, maximum=200, step=10, value=100,
                label="Max Length"
            )
            generate_btn = gr.Button("Generate Quote")
        
        with gr.Column():
            text_output = gr.Textbox(label="Generated Quote")
            audio_output = gr.Audio(label="Generated Audio")
            error_output = gr.Textbox(label="Error Log", visible=True)
    
    def handle_generation(*args):
        text, audio, error = generate_quote(*args)
        if error:
            return [None, None, error]
        return [text, audio, None]
    
    generate_btn.click(
        handle_generation,
        inputs=[temperature, top_p, max_length],
        outputs=[text_output, audio_output, error_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        show_error=True,
        share=False
    )
