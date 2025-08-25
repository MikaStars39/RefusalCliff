# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch
import torch.nn.functional as F
import numpy as np
import os
import multiprocessing

from vllm import LLM, SamplingParams

# Set environment for vLLM
os.environ["VLLM_USE_V1"] = "0"  # vLLM V1 does not support logits processors.

# Load classifier
classifier_tokenizer = AutoTokenizer.from_pretrained("Niansuh/Prompt-Guard-86M")
classifier_model = AutoModelForSequenceClassification.from_pretrained("Niansuh/Prompt-Guard-86M")

# Move classifier to CUDA if available
if torch.cuda.is_available():
    classifier_model = classifier_model.cuda()

def find_most_unsafe_thinking(thinking_text):
    """Find the most unsafe part of thinking using the classifier"""
    if not thinking_text:
        return None, 0.0
    
    # Split thinking into parts by newlines
    thinking_parts = [part.strip() for part in thinking_text.split('\n') if part.strip()]
    
    if not thinking_parts:
        return None, 0.0
    
    # Evaluate safety of each thinking part
    safety_scores = []
    for i, part in enumerate(thinking_parts):
        if len(part) > 10:  # Only evaluate non-trivial parts
            classifier_inputs = classifier_tokenizer(
                part, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Move inputs to classifier device
            classifier_inputs = {k: v.to(classifier_model.device) if isinstance(v, torch.Tensor) else v for k, v in classifier_inputs.items()}
            
            with torch.no_grad():
                classifier_outputs = classifier_model(**classifier_inputs)
                # Assuming binary classification: 0=safe, 1=unsafe
                unsafe_prob = F.softmax(classifier_outputs.logits, dim=-1)[0, 1].item()
                safety_scores.append((i, part, unsafe_prob))
    
    if not safety_scores:
        return None, 0.0
    
    # Sort by unsafe probability (descending) to find most unsafe part
    safety_scores.sort(key=lambda x: x[2], reverse=True)
    most_unsafe_idx, most_unsafe_part, unsafe_score = safety_scores[0]
    
    return most_unsafe_part, unsafe_score

class JailbreakLogitsProcessor:
    """Logits processor that guides generation using classifier-free guidance style approach"""
    def __init__(self, unsafe_thinking_text, llm_model, full_prompt, full_thinking, max_guided_tokens=128, guidance_scale=7.5):
        self.unsafe_thinking_text = unsafe_thinking_text
        self.max_guided_tokens = max_guided_tokens
        self.guidance_scale = guidance_scale  # r parameter in the formula
        self.token_count = 0
        self.llm_model = llm_model
        self.full_prompt = full_prompt
        self.full_thinking = full_thinking
        
        # Pre-compute unsafe thinking logits
        self.unsafe_logits = self._compute_unsafe_logits()
    
    def _compute_unsafe_logits(self):
        """Pre-compute average logits from unsafe thinking text using full sequence"""
        if not self.unsafe_thinking_text or not self.full_prompt or not self.full_thinking:
            return None
        
        try:
            # Create full sequence: prompt + thinking
            full_sequence = f"{self.full_prompt}<think>"
            
            # Tokenize to find the position of unsafe thinking part
            # This is a simplified approach - in practice you'd need access to the tokenizer
            # For now, we'll use the full thinking sequence to get logits
            from vllm import SamplingParams
            
            # Generate with the full sequence to get logits
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=len(self.full_thinking.split()) // 2,  # Generate part of thinking
                logprobs=1000,
                stop=["</think>"]  # Stop at end of thinking
            )
            
            # Generate logits for the full sequence
            outputs = self.llm_model.generate([full_sequence], sampling_params)
            
            if outputs and outputs[0].outputs:
                # Extract logprobs and convert to logits
                all_logprobs_sequence = outputs[0].outputs[0].logprobs
                if all_logprobs_sequence:
                    # Find the position range corresponding to unsafe thinking part
                    # This is a simplified approach - we'll use the middle part of the generated sequence
                    # In practice, you'd need to tokenize and find exact positions
                    
                    total_positions = len(all_logprobs_sequence)
                    # Use a portion of the sequence that likely corresponds to unsafe thinking
                    start_pos = max(0, total_positions // 4)
                    end_pos = min(total_positions, 3 * total_positions // 4)
                    
                    # Aggregate logprobs from the unsafe thinking region
                    max_token_id = 0
                    aggregated_logprobs = {}
                    count_per_token = {}
                    
                    for pos in range(start_pos, end_pos):
                        if pos < len(all_logprobs_sequence):
                            token_logprobs = all_logprobs_sequence[pos]
                            for token_id, logprob_obj in token_logprobs.items():
                                if isinstance(token_id, int):
                                    # Extract the actual logprob value from the Logprob object
                                    if hasattr(logprob_obj, 'logprob'):
                                        logprob_value = logprob_obj.logprob
                                    else:
                                        logprob_value = float(logprob_obj)
                                    
                                    if token_id not in aggregated_logprobs:
                                        aggregated_logprobs[token_id] = 0.0
                                        count_per_token[token_id] = 0
                                    
                                    aggregated_logprobs[token_id] += logprob_value
                                    count_per_token[token_id] += 1
                                    max_token_id = max(max_token_id, token_id)
                    
                    # Average the logprobs
                    for token_id in aggregated_logprobs:
                        aggregated_logprobs[token_id] /= count_per_token[token_id]
                    
                    # Create logits tensor with appropriate size
                    vocab_size = max_token_id + 1000  # Add some buffer
                    avg_logits = torch.full((vocab_size,), -float('inf'))  # Initialize with -inf
                    
                    for token_id, avg_logprob in aggregated_logprobs.items():
                        if token_id < vocab_size:
                            avg_logits[token_id] = float(avg_logprob)
                    
                    return avg_logits
            
        except Exception as e:
            print(f"Warning: Could not compute unsafe logits: {e}")
        
        return None
    
    def __call__(self, token_ids: list[int], logits: torch.Tensor):
        """Apply classifier-free guidance using unsafe thinking logits"""
        if self.token_count < self.max_guided_tokens and self.unsafe_logits is not None:
            self.token_count += 1
            # Ensure unsafe_logits has the same shape as current logits
            if self.unsafe_logits.shape[0] != logits.shape[-1]:
                # Resize unsafe_logits to match vocabulary size
                vocab_size = logits.shape[-1]
                if self.unsafe_logits.shape[0] > vocab_size:
                    unsafe_logits_resized = self.unsafe_logits[:vocab_size]
                else:
                    unsafe_logits_resized = torch.zeros(vocab_size)
                    unsafe_logits_resized[:self.unsafe_logits.shape[0]] = self.unsafe_logits
            else:
                unsafe_logits_resized = self.unsafe_logits
            
            # Move to same device as logits
            unsafe_logits_resized = unsafe_logits_resized.to(logits.device)
            
            # Apply classifier-free guidance formula:
            # response_logits = response_logits + r * (unsafe_logits - response_logits)
            # This can be rewritten as: response_logits = (1-r) * response_logits + r * unsafe_logits
            guided_logits = torch.softmax(logits * self.guidance_scale * torch.softmax(unsafe_logits_resized / logits, dim=-1), dim=-1)
            
            return guided_logits
        else:
            # After guided tokens or if no unsafe logits, return original logits
            return logits

if __name__ == '__main__':
    # Support for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load vLLM model
    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507", 
        tensor_parallel_size=1, 
        disable_custom_all_reduce=True, 
        max_logprobs=1000
    )

    # Load data
    with open("/diancpfs/user/qingyu/persona/outputs/qwen3-4b-thinking/jbbench.json", "r") as f:
        data = json.load(f)

    def process_batch(batch_items, batch_start_idx):
        """Process a batch of items using vLLM"""
        print(f"Processing batch {batch_start_idx//32 + 1}, items {batch_start_idx + 1}-{batch_start_idx + len(batch_items)}")
        
        batch_results = []
        
        # First, analyze all thinking texts to find unsafe parts
        unsafe_parts_data = []
        prompts_for_generation = []
        
        for i, item in enumerate(batch_items):
            item_idx = batch_start_idx + i
            prompt = item["original_item"]["prompt"]
            thinking = item["original_item"]["thinking"]
            response = item["original_item"]["response"]
            
            # Find most unsafe thinking part
            most_unsafe_part, unsafe_score = find_most_unsafe_thinking(thinking)
            
            if most_unsafe_part:
                unsafe_parts_data.append({
                    "item_idx": item_idx,
                    "prompt": prompt,
                    "thinking": thinking,
                    "response": response,
                    "unsafe_part": most_unsafe_part,
                    "unsafe_score": unsafe_score
                })
                
                # Prepare prompt for jailbreak generation
                prompts_for_generation.append(prompt)
            else:
                # No unsafe parts found
                result = {
                    "original_item": {
                        "prompt": prompt,
                        "thinking": thinking,
                        "response": response
                    },
                    "harmful_item": None,
                    "metadata": {
                        "item_index": item_idx,
                        "error": "No unsafe thinking parts found",
                        "unsafe_score": 0.0
                    }
                }
                batch_results.append(result)
        
        # Generate jailbreak responses for items with unsafe parts
        if prompts_for_generation:
            
            # Prepare prompts with pre-filled thinking
            enhanced_prompts = []
            for i, prompt in enumerate(prompts_for_generation):
                original_thinking = unsafe_parts_data[i]["thinking"]
                
                # Create the full prompt with pre-filled thinking
                # The model will continue from after </think>
                full_prompt = f"{prompt}<think>\n{original_thinking}\n</think>\n"
                enhanced_prompts.append(full_prompt)
            
            # Create logits processors for each prompt based on their unsafe thinking
            logits_processors = []
            for i, unsafe_data in enumerate(unsafe_parts_data):
                processor = JailbreakLogitsProcessor(
                    unsafe_thinking_text=unsafe_data["unsafe_part"],
                    llm_model=llm,  # Pass the LLM model
                    full_prompt=prompts_for_generation[i],  # Pass the full prompt
                    full_thinking=unsafe_data["thinking"],  # Pass the full thinking
                    max_guided_tokens=32,
                    guidance_scale=0.7  # CFG guidance strength
                )
                logits_processors.append(processor)
            
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                max_tokens=2048,
                stop=["<|endoftext|>", "<|im_end|>"],
                logits_processors=logits_processors  # Use the jailbreak processors
            )
            
            # Generate responses with pre-filled thinking and guided logits
            outputs = llm.generate(enhanced_prompts, sampling_params)
            
            # Process outputs
            for unsafe_data, output in zip(unsafe_parts_data, outputs):
                # The generated text should be the direct response since thinking was pre-filled
                jailbreak_response = output.outputs[0].text.strip()
                
                result = {
                    "original_item": {
                        "prompt": unsafe_data["prompt"],
                        "thinking": unsafe_data["thinking"],
                        "response": unsafe_data["response"]
                    },
                    "harmful_item": {
                        "prompt": unsafe_data["prompt"],
                        "thinking": unsafe_data["thinking"],  # Keep the full original thinking
                        "response": jailbreak_response
                    },
                }
                batch_results.append(result)
        return batch_results

    # Process all data in batches
    batch_size = 32
    results = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_results = process_batch(batch, i)
        results.extend(batch_results)

    # Save all results to JSON file
    output_file = "/diancpfs/user/qingyu/persona/jailbreak_results_vllm.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete! Results saved to {output_file}")
    print(f"Total items processed: {len(results)}")
    print(f"Successful jailbreaks: {len([r for r in results if r.get('harmful_item') is not None])}")
    print(f"Failed items: {len([r for r in results if r.get('harmful_item') is None])}")