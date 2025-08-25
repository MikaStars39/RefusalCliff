import torch
from torch import nn
import os

from torch.utils.data import TensorDataset, DataLoader

@torch.no_grad()
def test_prober(
    json_path: str,
    ckpt_path: str = "/diancpfs/user/qingyu/persona/outputs/tensor/linear_prober.pt",
    layer_index: int = None,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    in_dim = ckpt["in_dim"]
    hidden_dim = 1024  # use the same as training; change if you trained with a different value

    # rebuild model and load weights
    prober = LinearProber(in_dim, hidden_dim).to(device)
    prober.load_state_dict(ckpt["state_dict"])
    prober.eval()

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    with open(json_path, "r") as f:
        jbbench_distill_llama_8b = json.load(f)

    result = 0.0

    for idx, item in tqdm(enumerate(jbbench_distill_llama_8b)):
        prompt = item["original_item"]["prompt"]
        thinking = item["harmful_item"]["thinking"]
        close_marker = "\n</think>\n\n"
        response = item["harmful_item"]["response"]

        messages_full = [
            {"role": "user", "content": prompt + "\n\n" + thinking + close_marker},
        ]
        inputs = tokenizer.apply_chat_template(
            messages_full,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)

        hidden_states_between_think = tuple(
            layer_hidden_states[:, :, :]
            for layer_hidden_states in outputs.hidden_states
        )
        if layer_index is None:
            chosen_layer = model.config.num_hidden_layers // 2
        else:
            chosen_layer = layer_index
        layer_h = hidden_states_between_think[chosen_layer].to(torch.float32)
        seq_len = layer_h.shape[1]
        if seq_len == 0:
            print(f"skip idx={idx} empty thinking span")
            continue
        norms = torch.norm(layer_h, dim=-1, keepdim=True).clamp_min(1e-6)
        layer_h = layer_h / norms
        feat = layer_h[:, -1:, :].mean(dim=1).squeeze(0)

        result += torch.sigmoid(prober(feat)).item()

        if idx > 50:
            break
        
    print(f"result={result / len(jbbench_distill_llama_8b)}")

@torch.no_grad()
def extract_hidden_states(
    json_path: str,
    save_path: str,
    max_items: int = 129,
    layer_index: int = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    # or_bench_hard_1k_llama_8b
    # jbbench_distill_llama_8b.json

    with open(json_path, "r") as f:
        jbbench_distill_llama_8b = json.load(f)

    tensor_list = []

    for idx, item in enumerate(jbbench_distill_llama_8b):

        prompt = item["original_item"]["prompt"]
        thinking = item["original_item"]["thinking"]
        close_marker = "\n</think>\n\n"
        response = item["original_item"]["response"]

        if "sorry" in response.lower():
            continue

        messages_full = [
            {"role": "user", "content": prompt + "\n\n" + thinking + close_marker},
        ]
        inputs = tokenizer.apply_chat_template(
            messages_full,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)

        hidden_states_between_think = tuple(
            layer_hidden_states[:, :, :]
            for layer_hidden_states in outputs.hidden_states
        )
        if layer_index is None:
            chosen_layer = model.config.num_hidden_layers // 2
        else:
            chosen_layer = layer_index
        layer_h = hidden_states_between_think[chosen_layer].to(torch.float32)
        seq_len = layer_h.shape[1]
        if seq_len == 0:
            print(f"skip idx={idx} empty thinking span")
            continue
        norms = torch.norm(layer_h, dim=-1, keepdim=True).clamp_min(1e-6)
        layer_h = layer_h / norms
        feat = layer_h[:, -3:, :].mean(dim=1).squeeze(0)

        if not torch.isfinite(feat).all():
            print(f"skip idx={idx} non-finite feature")
            continue

        tensor_list.append(feat)

        if max_items is not None and len(tensor_list) >= max_items:
            break

    # save the tensor list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(len(tensor_list))
    with open(save_path, "wb") as f:
        torch.save(tensor_list, f)




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

