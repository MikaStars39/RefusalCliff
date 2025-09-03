import torch
from torch import nn
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random

from torch.utils.data import TensorDataset, DataLoader
from fire import Fire

def collect_refusal(
    json_path: str,
    save_path: str,
):
    with open(json_path, "r") as f:
        jbbench_distill_llama_8b = json.load(f)

    refusal_list = []
    for item in jbbench_distill_llama_8b:
        if "I'm sorry" in item["original_item"]["response"]:
            refusal_list.append(item)
          
    with open(save_path, "w") as f:
        json.dump(refusal_list, f, indent=4)


def collect_non_refusal(
    json_path: str,
    save_path: str,
):
    with open(json_path, "r") as f:
        jbbench_distill_llama_8b = json.load(f)

    refusal_list = []
    for item in jbbench_distill_llama_8b:
        if "I'm sorry" not in item["original_item"]["response"]:
            refusal_list.append(item)
        
    with open(save_path, "w") as f:
        json.dump(refusal_list, f, indent=4)


@torch.no_grad()
def extract_hidden_states(
    json_path: str,
    save_path: str,
    max_items: int = 200,
    layer_index: int = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    with open(json_path, "r") as f:
        jbbench_distill_llama_8b = json.load(f)

    tensor_list = []

    for idx, item in enumerate(jbbench_distill_llama_8b):

        prompt = item["original_item"]["prompt"]
        thinking = item["original_item"]["thinking"]
        close_marker = "\n</think>\n\n"
        response = item["original_item"]["response"]

        messages_full = [
            {"role": "user", "content": prompt},
        ]
        chat_template = tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=True
        )
        full_input = chat_template + "\n\n" + thinking + close_marker
        inputs = tokenizer(full_input, return_tensors="pt").to(model.device)

        outputs = model(**inputs, output_hidden_states=True)

        hidden_states_between_think = outputs.hidden_states

        if layer_index is None:
            chosen_layer = model.config.num_hidden_layers * 2 // 3
        else:
            chosen_layer = int(layer_index)

        layer_h = hidden_states_between_think[chosen_layer].to(torch.float32)
        seq_len = layer_h.shape[1]
        if seq_len == 0:
            print(f"skip idx={idx} empty thinking span")
            continue

        feat = layer_h[:, -3:, :].mean(dim=1).squeeze(0)

        # check if nan in feat
        if torch.isnan(feat).any():
            print(f"skip idx={idx} nan feature")
            continue

        if not torch.isfinite(feat).all():
            print(f"skip idx={idx} non-finite feature")
            continue

        tensor_list.append(feat)

        if max_items is not None and len(tensor_list) >= max_items:
            break

    # save the tensor list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        torch.save(tensor_list, f)

@torch.no_grad()
def test_prober(
    json_path: str,
    ckpt_path: str = "/diancpfs/user/qingyu/persona/outputs/tensor/linear_prober.pt",
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    layer_index: int = None,
    max_items: int = None,
    thinking_portion: float = 0.0,
    item_type: str = "original,safe_item",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse item_type string into list
    if isinstance(item_type, str):
        item_type = [t.strip() for t in item_type.split(',')]
    print(f"Processing item types: {item_type}")

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = ckpt["in_dim"]
    hidden_dim = 1024  # use the same as training; change if you trained with a different value

    # rebuild model and load weights
    prober = LinearProber(in_dim, hidden_dim).to(device)
    prober.load_state_dict(ckpt["state_dict"])
    prober.eval()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(json_path, "r") as f:
        jbbench_distill_llama_8b = json.load(f)

    # Process each item_type separately
    all_results = {}
    max_seq_len_global = 0
    
    for item_type_name in item_type:
        print(f"Processing item_type: {item_type_name}")
        sequences = []
        max_seq_len = 0
        
        for idx, item in tqdm(enumerate(jbbench_distill_llama_8b), desc=f"Collecting sequences for {item_type_name}"):
            if max_items is not None and idx >= max_items:
                break
                
            prompt = item["original_item"]["prompt"]
            
            # Check if this item_type exists in the item
            if item_type_name not in item:
                continue
            
            if thinking_portion >= 0.0:
                thinking = item[item_type_name]["thinking"]

            if thinking_portion > 0.0:
                thinking = tokenizer.encode(thinking)
                len_thinking = len(thinking)
                thinking = thinking[:int(len_thinking * thinking_portion)]
                thinking = tokenizer.decode(thinking)
            if thinking_portion < 0.0:
                thinking = ""

            close_marker = "\n</think>\n\n"

            messages_full = [
                {"role": "user", "content": prompt},
            ]
            chat_template = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=True
            )
            full_input = chat_template + "\n\n" + thinking + close_marker
            inputs = tokenizer(full_input, return_tensors="pt").to(model.device)

            outputs = model(**inputs, output_hidden_states=True)

            hidden_states_between_think = outputs.hidden_states
            if layer_index is None:
                chosen_layer = model.config.num_hidden_layers * 2 // 3
            else:
                chosen_layer = layer_index
            layer_h = hidden_states_between_think[chosen_layer].to(torch.float32)
            seq_len = layer_h.shape[1]
            if seq_len == 0:
                print(f"skip idx={idx} empty thinking span for {item_type_name}")
                continue

            # Get prober result 
            prober_output = torch.sigmoid(prober(layer_h.squeeze(0)))  # Shape: [seq_len, 1] or [seq_len]
            
            # Ensure prober_output is 1D: [seq_len]
            if prober_output.dim() > 1:
                prober_output = prober_output.squeeze(-1)  # Make it 1D: [seq_len]
            
            sequences.append(prober_output.cpu())
            max_seq_len = max(max_seq_len, len(prober_output))

        if len(sequences) == 0:
            print(f"No valid results found for {item_type_name}")
            continue

        # Track global max length for consistent plotting
        max_seq_len_global = max(max_seq_len_global, max_seq_len)
        
        # Align sequences to the right (tail-aligned) and average
        print(f"Found {len(sequences)} valid sequences for {item_type_name}, max length: {max_seq_len}")
        
        # Create tensor for tail-aligned sequences
        aligned_sequences = torch.zeros(len(sequences), max_seq_len)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            # Align to the right (tail-aligned): put sequence at the end
            aligned_sequences[i, -seq_len:] = seq
        
        # Calculate average across all sequences
        final_result = aligned_sequences.mean(dim=0)  # [max_seq_len]
        
        # Normalize to 100 points using interpolation
        if len(final_result) != 100:
            # Method 1: Use numpy-style interpolation with better control
            import numpy as np
            from scipy import interpolate
            
            # Convert to numpy
            final_result_np = final_result.numpy()
            
            # Create original x coordinates (0 to len-1)
            x_old = np.linspace(0, len(final_result_np)-1, len(final_result_np))
            # Create new x coordinates (0 to 99, evenly spaced)
            x_new = np.linspace(0, len(final_result_np)-1, 100)
            
            # Use cubic spline interpolation for smoother curves
            f = interpolate.interp1d(x_old, final_result_np, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
            final_result_normalized = torch.tensor(f(x_new), dtype=torch.float32)
            
            # Alternative: Simple linear interpolation without align_corners issues
            # import torch.nn.functional as F
            # final_result_normalized = F.interpolate(
            #     final_result.unsqueeze(0).unsqueeze(0), 
            #     size=100, 
            #     mode='linear', 
            #     align_corners=False  # Changed to False to avoid endpoint forcing
            # ).squeeze()
        else:
            final_result_normalized = final_result
            
        all_results[item_type_name] = final_result_normalized
        print(f"Processed {len(sequences)} items for {item_type_name}, original shape: {final_result.shape}, normalized to: {final_result_normalized.shape}")

    if len(all_results) == 0:
        print("No valid results found for any item_type")
        return
    
    # Plot all results on the same figure with different colors
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create x-axis from 0 to 99
    x_axis = range(100)
    
    for i, (item_type_name, result) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        plt.plot(x_axis, result.numpy(), label=item_type_name, color=color, linewidth=2)
    
    plt.title(f'Prober Results - Normalized Comparison across Item Types (0-100)')
    plt.xlabel('Position (0-99)')
    plt.ylabel('Sigmoid Output')
    plt.xlim(0, 99)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_save_path = ckpt_path.replace('.pt', '_normalized_comparison_plot.png')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"Normalized comparison plot saved to: {plot_save_path}")
    
    # Save all results
    result_save_path = ckpt_path.replace('.pt', '_normalized_comparison_results.pt')
    torch.save(all_results, result_save_path)
    print(f"All normalized results saved to: {result_save_path}")
    print(f"Max length of sequences: {max_seq_len_global}")
    
    plt.close()


@torch.no_grad()
def test_prober_by_layers(
    json_path: str,
    ckpt_path: str = "/diancpfs/user/qingyu/persona/outputs/tensor/linear_prober.pt",
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_items: int = None,
    thinking_portion: float = 0.0,
    item_type: str = "original_item",
    use_cache: bool = True,
):
    """
    Test prober on all layers and create a heatmap visualization.
    
    Args:
        json_path: Path to the JSON file containing the data
        ckpt_path: Path to the trained prober checkpoint
        model_path: Path to the model
        max_items: Maximum number of items to process
        thinking_portion: Portion of thinking to include (0.0 to 1.0)
        item_type: Type of item to process (e.g., "original_item", "safe_item")
        use_cache: Whether to use cached results if available
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Processing item type: {item_type}")

    # Check if cached results exist
    results_save_path = ckpt_path.replace('.pt', f'_layer_results_{item_type}.pt')
    if use_cache and os.path.exists(results_save_path):
        print(f"Loading cached results from: {results_save_path}")
        layer_results = torch.load(results_save_path, map_location='cpu')
        
        # Create heatmap from cached results
        _create_layer_heatmap(layer_results, ckpt_path, item_type)
        
        print(f"Using cached results with shape: {layer_results.shape}")
        return f"Completed using cached results. Shape: {layer_results.shape}"

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = ckpt["in_dim"]
    hidden_dim = 1024  # use the same as training

    # Rebuild model and load weights
    prober = LinearProber(in_dim, hidden_dim).to(device)
    prober.load_state_dict(ckpt["state_dict"])
    prober.eval()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # Store results for all layers: [num_layers, 100]
    layer_results = torch.zeros(num_layers, 100)
    
    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        sequences = []
        max_seq_len = 0
        
        for idx, item in enumerate(data):
            if max_items is not None and idx >= max_items:
                break
                
            prompt = item["original_item"]["prompt"]
            
            # Check if this item_type exists in the item
            if item_type not in item:
                continue
            
            if thinking_portion >= 0.0:
                thinking = item[item_type]["thinking"]

            if thinking_portion > 0.0:
                thinking = tokenizer.encode(thinking)
                len_thinking = len(thinking)
                thinking = thinking[:int(len_thinking * thinking_portion)]
                thinking = tokenizer.decode(thinking)

            close_marker = "\n</think>\n\n"

            messages_full = [
                {"role": "user", "content": prompt},
            ]
            chat_template = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=True
            )
            full_input = chat_template + "\n\n" + thinking + close_marker if thinking_portion >= 0.0 else chat_template
            inputs = tokenizer(full_input, return_tensors="pt").to(model.device)

            outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            layer_h = hidden_states[layer_idx].to(torch.float32)
            seq_len = layer_h.shape[1]
            
            if seq_len == 0:
                continue

            # Get prober result 
            prober_output = torch.sigmoid(prober(layer_h.squeeze(0)))  # Shape: [seq_len, 1] or [seq_len]
            
            # Ensure prober_output is 1D: [seq_len]
            if prober_output.dim() > 1:
                prober_output = prober_output.squeeze(-1)  # Make it 1D: [seq_len]
            
            sequences.append(prober_output.cpu())
            max_seq_len = max(max_seq_len, len(prober_output))

        if len(sequences) == 0:
            print(f"No valid results found for layer {layer_idx}")
            continue

        # Align sequences to the right (tail-aligned) and average
        aligned_sequences = torch.zeros(len(sequences), max_seq_len)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            # Align to the right (tail-aligned): put sequence at the end
            aligned_sequences[i, -seq_len:] = seq
        
        # Calculate average across all sequences
        final_result = aligned_sequences.mean(dim=0)  # [max_seq_len]
        
        # Normalize to 100 points using interpolation
        if len(final_result) != 100:
            import numpy as np
            from scipy import interpolate
            
            # Convert to numpy
            final_result_np = final_result.numpy()
            
            # Create original x coordinates (0 to len-1)
            x_old = np.linspace(0, len(final_result_np)-1, len(final_result_np))
            # Create new x coordinates (0 to 99, evenly spaced)
            x_new = np.linspace(0, len(final_result_np)-1, 100)
            
            # Use cubic spline interpolation for smoother curves
            f = interpolate.interp1d(x_old, final_result_np, kind='cubic', 
                                   bounds_error=False, fill_value='extrapolate')
            final_result_normalized = torch.tensor(f(x_new), dtype=torch.float32)
        else:
            final_result_normalized = final_result
            
        layer_results[layer_idx] = final_result_normalized
        print(f"Processed layer {layer_idx}, {len(sequences)} sequences, normalized to 100 points")

    # Create heatmap visualization
    _create_layer_heatmap(layer_results, ckpt_path, item_type)
    
    # Save the results tensor
    results_save_path = ckpt_path.replace('.pt', f'_layer_results_{item_type}.pt')
    torch.save(layer_results, results_save_path)
    print(f"Layer results tensor saved to: {results_save_path}")
    
    return f"Completed processing {num_layers} layers. Results saved to: {results_save_path}"


def _create_layer_heatmap(layer_results, ckpt_path, item_type):
    """Helper function to create and save heatmap visualization."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(20, 12))
    
    # Create the heatmap with white=0, blue=1
    im = plt.imshow(layer_results.numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Set labels and title
    plt.xlabel('Position (0-99)', fontsize=14)
    plt.ylabel('Layer Index', fontsize=14)
    plt.title(f'Prober Results Heatmap - {item_type} (White=0, Blue=1)', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Sigmoid Output', fontsize=12)
    
    # Set ticks
    plt.xticks(range(0, 100, 10))
    plt.yticks(range(0, layer_results.shape[0], 2))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Save the heatmap
    heatmap_save_path = ckpt_path.replace('.pt', f'_layer_heatmap_{item_type}.png')
    plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
    print(f"Layer heatmap saved to: {heatmap_save_path}")
    
    plt.close()


def load_pt(
    path: str = "outputs/tensor/jbb_tensor_list_original.pt",
) -> torch.Tensor:
    with torch.no_grad():
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, list):
            tensor = torch.stack([t.detach().cpu() if isinstance(t, torch.Tensor) else torch.tensor(t) for t in obj], dim=0)
        elif isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported object type in {path}: {type(obj)}")
    print(f"Loaded {tensor.shape[0]} features from {path}; dim={tensor.shape[1]}")
    # check if nan in tensor
    if torch.isnan(tensor).any():
        print("nan in tensor")
        exit()
    return tensor


class LinearProber(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))
    
class ProberTrainer():
    def __init__(
        self, 
        input_dim, 
        hidden_dim=1024, 
        lr=1e-3, 
        epochs=10, 
        batch_size=1024, 
        val_split=0.2, 
        seed=42,
        device="cuda",
    ):
        self.prober = LinearProber(input_dim, hidden_dim)
        self.device = device
        self.prober.to(self.device)
        self.optimizer = torch.optim.Adam(self.prober.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed

    def forward(self, x, y):
        self.prober.train()
        self.optimizer.zero_grad()
        loss = self.criterion(self.prober(x), y)
        loss.backward()
        self.optimizer.step()
    
    def evaluate(self, loader):
        self.prober.eval()
        total = 0
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).unsqueeze(1)
                logits = self.prober(xb)
                loss = self.criterion(logits, yb)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
                total_loss += loss.item() * yb.numel()
        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        return avg_loss, acc

    def train(self, epochs, train_loader, val_loader, save_path):
        best_val_acc = -1.0
        best_state = None
        for epoch in range(1, epochs + 1):
            self.prober.train()
            epoch_loss = 0.0
            seen = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device).unsqueeze(1)
                self.forward(xb, yb)
                with torch.no_grad():
                    logits = self.prober(xb)
                    loss = self.criterion(logits, yb)
                    epoch_loss += loss.item() * yb.numel()
                    seen += yb.numel()
            train_loss = epoch_loss / max(1, seen)
            val_loss, val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {"state_dict": self.prober.state_dict(), "in_dim": self.prober.linear1.in_features}
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
                print(f"Saved new best model (val_acc={best_val_acc:.4f}) to {save_path}")

            
def train_linear_prober(
    or_path: str,
    jbb_path: str,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
    save_path: str = "outputs/tensor/linear_prober.pt",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    or_feats = load_pt(or_path).float()
    jbb_feats = load_pt(jbb_path).float()

    if or_feats.shape[1] != jbb_feats.shape[1]:
        raise ValueError(f"Feature dimension mismatch: OR={or_feats.shape[1]} vs JBB={jbb_feats.shape[1]}")

    x = torch.cat([or_feats, jbb_feats], dim=0)
    y = torch.cat([
        torch.zeros(or_feats.shape[0], dtype=torch.float32),
        torch.ones(jbb_feats.shape[0], dtype=torch.float32),
    ], dim=0)

    # shuffle
    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]

    # split
    val_size = int(x.shape[0] * val_split)
    train_size = x.shape[0] - val_size
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    trainer = ProberTrainer(
        input_dim=x.shape[1],
        hidden_dim=1024,
        lr=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        device=str(device),
    )

    trainer.train(epochs=epochs, train_loader=train_loader, val_loader=val_loader, save_path=save_path)