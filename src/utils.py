import os
import json
import torch
from transformers import LlavaNextForConditionalGeneration, AutoModelForCausalLM

def save_model_and_tokenizer(model_name_or_path, model, tokenizer, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")
    
    merged_model = model.merge_and_unload() 
    
    for name, param in merged_model.named_parameters():
        if param.device.type == "meta":
            print(f"Parameter {name} is on meta device. Moving to CPU...")
            param.data = torch.empty_like(param).to("cpu")
            
    if drop_layers_after is not None:
        anchor_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=merged_model.dtype, device_map="auto")
        merged_model.transformer.h = merged_model.transformer.h + anchor_model.transformer.h[drop_layers_after + 1:]
        merged_model.config = anchor_model.config
    else:
        raise ValueError("Incompatible layer structure for model merging.")

    try:
        merged_model.save_pretrained(output_dir, max_shard_size="5GB")
    except Exception as e:
        print(f"Error saving model: {e}")
        
    tokenizer.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)

    torch.use_deterministic_algorithms(False)

    if trainer.training_args.do_eval:
        trainer.evaluate()

def save_llava_model_and_tokenizer(model_name_or_path, model, processor, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model and processor saving to {output_dir}")
    
    merged_model = model.merge_and_unload() 
    
    anchor_model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=merged_model.dtype)
    
    merged_model.language_model.model.layers = merged_model.language_model.model.layers + anchor_model.language_model.model.layers[drop_layers_after + 1:]
    
    merged_model.config = anchor_model.config

    try:
        merged_model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"Error saving model: {e}")
    
    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)

    if trainer.training_args.do_eval:
        trainer.evaluate()

def get_model_generation(inputs, model, tokenizer):
    tokenizer.chat_template = {
        "system": "You are a helpful assistant.",
        "user": "{input}",
        "assistant": "{output}"
    }
    inputs = tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False) + prefill
    return model.generate(inputs)
