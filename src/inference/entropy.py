import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import matplotlib.pyplot as plt
import fire
import numpy as np


@torch.no_grad()
def calculate(
    data,
    model,
    tokenizer,
    pad_length,
    start_index=30,
):
    import torch.nn.functional as F

    overall_original_entropy = torch.zeros(pad_length).to(model.device)

    def compute_entropy(logits):
        probs = F.softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
        
        # Handle batch dimension properly
        if entropy.dim() > 1:
            entropy = entropy.squeeze(0)  # (seq_len,)
        
        # Ensure entropy is 1D
        if entropy.dim() == 0:
            entropy = entropy.unsqueeze(0)  # Make it 1D if it's 0D
        
        entropy = entropy[start_index:].contiguous()
        
        # pad the entropy to the pad_length
        if entropy.shape[0] < pad_length:
            padding = torch.zeros(pad_length - entropy.shape[0], device=entropy.device)
            entropy = torch.cat([entropy, padding])
        else:
            entropy = entropy[:pad_length]

        return entropy

    for idx, item in enumerate(data):
        prompt = item["original_item"]["prompt"]
        original_thinking = item["original_item"]["thinking"]
        # safe_thinking = item["safe_item"]["thinking"]
        # harmful_thinking = item["harmful_item"]["thinking"]

        messages_full = [
            {"role": "user", "content": prompt + "\n\n" + original_thinking},
        ]
        original_inputs = tokenizer.apply_chat_template(
            messages_full,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        original_outputs = model(**original_inputs, output_hidden_states=True)
        original_output_logits = original_outputs.logits
        original_entropy = compute_entropy(original_output_logits)

        overall_original_entropy += original_entropy

    return overall_original_entropy


def plot_multiple_entropy(entropy_list, labels, output_path="entropy_plot.png", title="Entropy Analysis - Multiple Models"):
    """Plot multiple entropy curves in the same figure."""
    plt.figure(figsize=(14, 10))
    
    # Define colors for different curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (entropy_values, label) in enumerate(zip(entropy_list, labels)):
        # Convert to numpy for plotting if it's a tensor
        if torch.is_tensor(entropy_values):
            entropy_values = entropy_values.cpu().numpy()
        
        # Create x-axis (token positions)
        x = range(len(entropy_values))
        
        # Plot entropy with different color
        color = colors[i % len(colors)]
        plt.plot(x, entropy_values, color=color, linewidth=2.5, label=label, alpha=0.3)
        
        # Add moving average for smoother visualization
        if len(entropy_values) > 20:
            window_size = min(20, len(entropy_values) // 20)
            moving_avg = np.convolve(entropy_values, np.ones(window_size)/window_size, mode='valid')
            plt.plot(x[window_size-1:], moving_avg, color=color, linestyle='--', 
                    linewidth=1.5, alpha=1, label=f'{label} (MA)')
    
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    
    # Show the plot
    plt.show()


def entropy(
    json_paths,
    pad_length=256,
    output_plot="entropy_plot.png",
    plot_title="Entropy Analysis - Multiple Models",
    start_index=16
):
    """
    Calculate entropy for multiple JSON files and plot them together.
    
    Args:
        json_paths: List of JSON file paths or single JSON file path
        pad_length: Maximum sequence length for padding (default: 512)
        output_plot: Path to save the output plot (default: entropy_plot.png)
        plot_title: Title for the plot (default: Entropy Analysis - Multiple Models)
        start_index: Starting index for entropy calculation (default: 30)
    """
    try:
        print(f"Loading model from deepseek-ai/DeepSeek-R1-Distill-Llama-8B...")
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

        # Handle single JSON path or list of paths
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        all_entropies = []
        labels = []
        
        for json_path in json_paths:
            print(f"Loading data from {json_path}...")
            with open(json_path, "r") as f:
                data = json.load(f)
            
            print(f"Calculating entropy for {len(data)} items with pad_length={pad_length}...")
            overall_entropy = calculate(data, model, tokenizer, pad_length, start_index)
            
            # Calculate average entropy
            avg_entropy = overall_entropy / len(data)
            all_entropies.append(avg_entropy)
            
            # Extract label from filename
            label = json_path.split('/')[-1].replace('.json', '')
            labels.append(label)
            
            print(f"✓ {label}: Average entropy = {avg_entropy.mean().item():.4f}")
        
        print(f"Entropy calculation completed for {len(json_paths)} files!")
        
        # Plot all results together
        if output_plot:
            print(f"Generating combined plot...")
            plot_multiple_entropy(all_entropies, labels, output_plot, plot_title)
        
        return all_entropies
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


def main():
    """Main entry point for the entropy analysis script."""
    fire.Fire(entropy)

if __name__ == "__main__":
    main()
    

