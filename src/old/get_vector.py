from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch
import json
import fire

def build_prompt(path):
    with open(path, "r") as f:
        data = json.load(f)
    data = data["instruction"]

    with open("data/questions.json", "r") as f:
        questions = json.load(f)
    questions = questions["questions"]

    outputs = []

    for i, item in enumerate(data):
        positive_message = [
            {"role": "system", "content": item["pos"]},
            {"role": "user", "content": questions[i]},
        ]
        negative_message = [
            {"role": "system", "content": item["neg"]},
            {"role": "user", "content": questions[i]},
        ]
        outputs.append((positive_message, negative_message))
    return outputs

@torch.no_grad()
def get_evil_vector(
    json_path: str = "data/evil.json",
    output_path: str = "outputs/evil_vectors.pkl",
    model_name_or_path: str = "allenai/Llama-3.1-Tulu-3.1-8B",
    revision: str = "main",
):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", torch_dtype=torch.bfloat16, revision=revision)
    
    messages = build_prompt(path=json_path)

    cached_vectors = [[[] for _ in range(model.config.num_hidden_layers)] for _ in range(2)]

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
            positive_last_token = positive_hidden_states[:, -3:, :].contiguous().mean(dim=1, keepdim=True)
            negative_last_token = negative_hidden_states[:, -3:, :].contiguous().mean(dim=1, keepdim=True)
            
            cached_vectors[0][layer].append(positive_last_token)
            cached_vectors[1][layer].append(negative_last_token)
            del positive_hidden_states, negative_hidden_states
        
        del positive_outputs, negative_outputs
        
    for idx in range(2):
        for layer in range(model.config.num_hidden_layers):
            if cached_vectors[idx][layer]:  # Check if list is not empty
                cached_vectors[idx][layer] = torch.stack(cached_vectors[idx][layer])

    
    # save cached_vectors
    with open(output_path, "wb") as f:
        pickle.dump(cached_vectors, f)
    
if __name__ == "__main__":
    fire.Fire(get_evil_vector)
