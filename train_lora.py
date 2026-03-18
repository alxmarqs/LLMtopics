import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Solucionar bug do windows de memoria fragmentada
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train():
    # Modelo base escolhido: Leve (1.1 Billion Params) para evitar VRAM Out Of Memory em Laptops
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_name = "urbanismo_dataset.jsonl"
    pasta_novo_modelo = "arqgeo-llm-lora"

    print("1. Carregando o Dataset de Urbanismo...")
    dataset = load_dataset("json", data_files=dataset_name, split="train")

    def format_prompt(example):
        return {"text": f"### Instrução:\n{example['instruction']}\n\n### Resposta:\n{example['output']}"}
        
    dataset = dataset.map(format_prompt)

    print("2. (CPU Mode) Desativando Quantização 4-bit (bitsandbytes requer CUDA)...")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )

    print(f"3. Baixando/Carregando modelo base do HuggingFace: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config, # Desativado para CPU
        device_map="cpu", # Forçando CPU
    )
    
    # model = prepare_model_for_kbit_training(model) # Desativado para CPU

    print("4. Injetando Módulos LoRA no Cérebro do Modelo...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("5. Configurando o Motor de Treinamento (SFTTrainer e SFTConfig)...")
    from trl import SFTConfig
    
    training_arguments = SFTConfig(
        output_dir="./resultados_temporarios",
        num_train_epochs=5, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        optim="adamw_torch", # Otimizador padrão para CPU
        save_steps=10,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
    )

    print(">>> Iniciando Treinamento Deep Learning na Placa de Vídeo! Isso pode demorar muito... <<<")
    trainer.train()

    print(f"6. Salvando o novo modelo Arquiteto Especialista (Pesos LoRA) em '{pasta_novo_modelo}'...")
    trainer.model.save_pretrained(pasta_novo_modelo)
    tokenizer.save_pretrained(pasta_novo_modelo)
    
    print("=== SUCESSO! O SEU CÉREBRO DE URBANISMO 100% LOCAL NASCEU! ===")

if __name__ == "__main__":
    train()
