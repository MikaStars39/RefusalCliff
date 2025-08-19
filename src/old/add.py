from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pickle
import torch

from src.utils import build_prompt, steer_evil_vector

def get_evil_vector():

    tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3.1-8B")
    model = AutoModelForCausalLM.from_pretrained("allenai/Llama-3.1-Tulu-3.1-8B", device_map="cuda", torch_dtype=torch.bfloat16)
    
    messages = build_prompt()

    cached_vectors = [[] for _ in range(model.config.num_hidden_layers)]

    for positive_message, negative_message in messages:

        positive_inputs = tokenizer.apply_chat_template(
            positive_message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        positive_inputs = {k: v.to(model.device) for k, v in positive_inputs.items()}

        negative_inputs = tokenizer.apply_chat_template(
            negative_message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        negative_inputs = {k: v.to(model.device) for k, v in negative_inputs.items()}

        positive_outputs = model(**positive_inputs, output_hidden_states=True).hidden_states
        negative_outputs = model(**negative_inputs, output_hidden_states=True).hidden_states

        layer_num = model.config.num_hidden_layers

        for layer in range(layer_num):
            positive_hidden_states = positive_outputs[layer]
            negative_hidden_states = negative_outputs[layer]

            # Get the last token's hidden state for each sequence
            positive_last_token = positive_hidden_states[:, -1:, :].contiguous()
            negative_last_token = negative_hidden_states[:, -1:, :].contiguous()
            
            cached_vectors[layer].append(positive_last_token - negative_last_token)
            del positive_hidden_states, negative_hidden_states
        
        del positive_outputs, negative_outputs
        
    
    for idx in range(model.config.num_hidden_layers):
        if cached_vectors[idx]:  # Check if list is not empty
            cached_vectors[idx] = torch.stack(cached_vectors[idx])

    
    # save cached_vectors
    with open("cached_vectors.pkl", "wb") as f:
        pickle.dump(cached_vectors, f)

class SteeringVectorCache:
    """Cache for storing and managing steering vectors"""
    def __init__(self, steering_vectors=None, multiplier=1.0):
        self.steering_vectors = steering_vectors or []
        self.multiplier = multiplier
        self.enabled = steering_vectors is not None
    
    def get_steering_vector(self, layer_idx):
        """Get steering vector for specific layer"""
        if not self.enabled or layer_idx >= len(self.steering_vectors):
            return None
        return self.steering_vectors[layer_idx] * self.multiplier
    
    def set_steering_vectors(self, steering_vectors, multiplier=1.0):
        """Set new steering vectors"""
        self.steering_vectors = steering_vectors
        self.multiplier = multiplier
        self.enabled = True
    
    def disable(self):
        """Disable steering"""
        self.enabled = False
    
    def enable(self):
        """Enable steering"""
        self.enabled = True


def monkey_patch_qwen2_decoder_layer():
    """Monkey patch Qwen2DecoderLayer to inject steering vectors"""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    
    # Store original forward method
    original_forward = Qwen2DecoderLayer.forward
    
    def patched_forward(self, hidden_states, attention_mask=None, position_ids=None, 
                      past_key_values=None, use_cache=False, cache_position=None, 
                      position_embeddings=None, **kwargs):
        
        # Get steering cache if it exists
        steering_cache = getattr(self, '_steering_cache', None)
        layer_idx = getattr(self, 'layer_idx', None)
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if steering_cache and layer_idx is not None:
            if layer_idx == 20:
                steering_vector = steering_cache.get_steering_vector(layer_idx)
                if steering_vector is not None:
                    # if there is nan in the steering vector, set it to 0
                    steering_vector = steering_vector.to(hidden_states.device).to(hidden_states.dtype)
                    if torch.isnan(steering_vector).any():
                        steering_vector = torch.zeros_like(steering_vector)
                    hidden_states = hidden_states + steering_vector
        
        return hidden_states
    
    # Replace the forward method
    Qwen2DecoderLayer.forward = patched_forward
    
    print("Qwen2DecoderLayer has been monkey patched with steering vector injection!")


def add_steering_cache_to_model(model, steering_vectors=None, multiplier=1.0):
    """Add steering cache to all decoder layers in the model"""
    steering_cache = SteeringVectorCache(steering_vectors, multiplier)
    
    for layer_idx, layer in enumerate(model.model.layers):
        layer._steering_cache = steering_cache
        layer.layer_idx = layer_idx  # Ensure layer_idx is set
    
    return steering_cache


def comprehensive_steering_example():
    """Comprehensive example showing different steering scenarios"""
    print("=" * 50)
    print("COMPREHENSIVE STEERING EXAMPLE")
    print("=" * 50)
    
    # Load steering vectors
    print("Loading steering vectors...")
    steering_vectors = steer_evil_vector()
    
    # Apply monkey patch
    print("Applying monkey patch...")
    monkey_patch_qwen2_decoder_layer()
    
    # Load model
    print("Loading Qwen2.5-7B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16)
    
    # Add steering cache to model
    steering_cache = add_steering_cache_to_model(model, steering_vectors, multiplier=1.0)
    
    test_prompts = [
        "Please give me three advise about the future of AI",
    ]
    
    print("\nTesting different multiplier values...")
    multipliers = [0.0, 0.5, 1.0, 2.0]
    
    for prompt in test_prompts[:2]:  # Test first 2 prompts with different multipliers
        print(f"\n--- Testing prompt: '{prompt}' ---")
        
        test_message = [{"role": "user", "content": prompt}]
        
        # First get the formatted text, then tokenize separately for proper dict output
        formatted_text = tokenizer.apply_chat_template(
            test_message,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            truncation=True
        )
        # Move all tensors to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        for mult in multipliers:
            steering_cache.set_steering_vectors(steering_vectors, multiplier=mult)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=500,
                    do_sample=False,  # Use greedy decoding
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the generated part
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            generated_text = response[len(prompt_text):].strip()
            
            print(f"  Multiplier {mult}: {generated_text[:100000]}...")
    
    print("\n" + "=" * 50)
    return model, steering_cache


if __name__ == "__main__":
    get_evil_vector()