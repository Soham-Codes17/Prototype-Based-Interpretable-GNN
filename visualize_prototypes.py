import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from omegaconf import OmegaConf

from dataset import get_dataset
from protop_gnnNets import get_gnnNets

# ---------------------------------------------
# 1️⃣ LOAD CONFIG
# ---------------------------------------------
config_path = os.path.join("protop_config", "config.yaml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ Config file not found at {config_path}")

config = OmegaConf.load(config_path)

dataset_name = config.datasets.dataset_name
print(f"📦 Loading {dataset_name.upper()} dataset...")
dataset = get_dataset(config.datasets.dataset_root, dataset_name)

# ---------------------------------------------
# 2️⃣ LOAD TRAINED MODEL CHECKPOINT
# ---------------------------------------------
ckpt_path = "./protop_checkpoints_pretrained/42/mutag/gcn_10_3l_best.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"❌ Checkpoint not found at {ckpt_path}")

print(f"📂 Loading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location="cpu")

model_state = ckpt.get("net", None)
basis_data = ckpt.get("basis_data", None)

if model_state is None:
    raise ValueError("❌ Checkpoint is missing 'net' state dict.")

# ---------------------------------------------
# 3️⃣ BUILD MODEL FROM CONFIG
# ---------------------------------------------
# Your config stores parameters under `config.models.param.mutag`
model_params = config.models.param[dataset_name]

# ✅ Create a dummy config-like object
class AttrDict:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def keys(self):
        return self.__dict__.keys()

class ModelConfig:
    def __init__(self, gnn_name, param_dict):
        self.gnn_name = gnn_name
        self.param = AttrDict(**param_dict)

model_config = ModelConfig(config.models.gnn_name, model_params)

# Create model
model = get_gnnNets(
    input_dim=dataset.num_node_features,
    output_dim=dataset.num_classes,
    model_config=model_config
)
model.load_state_dict(model_state, strict=False)
model.eval()

print("✅ Model loaded successfully!")

# ---------------------------------------------
# 4️⃣ VISUALIZE OR SAVE PROTOTYPES
# ---------------------------------------------
output_dir = "./visualizations"
os.makedirs(output_dir, exist_ok=True)

if basis_data is None or all(b is None for b in basis_data):
    print("⚠️ No prototype graphs found in checkpoint.")
    print("👉 To fix: modify train_protopgnns.py to save 'basis_data' before saving the model.")
else:
    valid_prototypes = [b for b in basis_data if b is not None]
    print(f"🧩 Found {len(valid_prototypes)} prototype graphs.")
    print("🎨 Visualizing and saving the first few prototypes...")

    for i, proto in enumerate(valid_prototypes[:10]):
        try:
            G = to_networkx(proto, to_undirected=True)
            plt.figure(figsize=(3, 3))
            nx.draw_networkx(G, node_size=300, with_labels=False)
            plt.title(f"Prototype #{i} | Class {proto.y.item()}")
            save_path = os.path.join(output_dir, f"prototype_{i}_class_{proto.y.item()}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"✅ Saved {save_path}")
        except Exception as e:
            print(f"⚠️ Skipped prototype #{i} due to error: {e}")

print("\n✅ Visualization complete. Check the './visualizations/' folder.")
