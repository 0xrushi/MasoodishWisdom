from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from datasets import Dataset
import json, os
from transformers import TrainerCallback, pipeline

with open("cleaned_final.json", "r") as f:
    raw_data = json.load(f)

formatted_data = [
    {"text": f"Generate a Masood Boomgaard style quote:\n{item['text']}"}
    for item in raw_data
]

dataset = Dataset.from_list(formatted_data)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./mistral-boomgaard",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)


class QuoteLoggerCallback(TrainerCallback):
    def __init__(self, peft_model, tokenizer, num_samples=3, prompt="Generate a Masood Boomgaard style quote:", output_dir="./checkpoints"):
        self.text_gen = pipeline("text-generation", model=peft_model, tokenizer=tokenizer)
        self.peft_model = peft_model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.prompt = prompt
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print("Epoch complete — Boomgaard wisdom incoming:")
        epoch = int(state.epoch)

        generations = []
        for i in range(self.num_samples):
            output = self.text_gen(self.prompt, max_new_tokens=60, do_sample=True, temperature=1.2)[0]['generated_text']
            generations.append(output)
            print(f"Sample {i+1}: {output}")

        checkpoint_path = os.path.join(self.output_dir, f"epoch-{epoch}")
        os.makedirs(checkpoint_path, exist_ok=True)
        self.peft_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        with open(os.path.join(checkpoint_path, f"quotes_epoch_{epoch}.txt"), "w", encoding="utf-8") as f:
            for i, quote in enumerate(generations, 1):
                f.write(f"Sample {i}:\n{quote}\n\n")

quote_callback = QuoteLoggerCallback(trainer.model, tokenizer)
trainer.callback_handler.callbacks.append(quote_callback)

trainer.train()
