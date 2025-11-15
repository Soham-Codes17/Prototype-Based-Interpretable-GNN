import torch
import torch.nn.functional as F
import numpy as np
from protop_gnnNets import get_gnnNets
from dataset import get_dataset
from torch_geometric.loader import DataLoader


# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "mutag"

CHECKPOINT_PATH = "protop_checkpoints_pretrained/42/mutag/gcn_10_3l_best.pth"
SAVE_PRUNED_MODEL_PATH = "protop_checkpoints_pretrained/42/mutag/gcn_10_3l_pruned.pth"

NUM_CLASSES = 2
NUM_BASIS_PER_CLASS = 10
BASIS_DIM = 128
GNN_LATENT_DIM = [128, 128, 128]
FC_LATENT_DIM = [128]

GNN_DROPOUT = 0.0
FC_DROPOUT = 0.0

ACTIVATION_THRESHOLD = 0.02
SIMILARITY_THRESHOLD = 0.90


# =========================================================
# HYDRA-LIKE CONFIG (MUST MATCH TRAINING EXACTLY)
# =========================================================

class HybridDict(dict):
    """Behaves like dict + attribute access (Hydra style)"""
    def __init__(self, **items):
        super().__init__(items)
        for k, v in items.items():
            setattr(self, k, v)

    def __getattr__(self, key):
        return self[key]


class ModelConfig:
    """Matches config.models.param["mutag"] structure"""
    def __init__(self):
        self.gnn_name = "gcn"
        self.param = HybridDict(
            basis_dim=BASIS_DIM,
            num_basis_per_class=NUM_BASIS_PER_CLASS,
            gnn_latent_dim=GNN_LATENT_DIM,
            fc_latent_dim=FC_LATENT_DIM,
            gnn_dropout=GNN_DROPOUT,
            fc_dropout=FC_DROPOUT,
            gnn_emb_normalization=False,
            gcn_adj_normalization=True,
            add_self_loop=True,
            gnn_nonlinear="relu",
            fc_nonlinear="relu",
            readout="mean"
        )


# =========================================================
# LOAD MODEL + DATASET EXACTLY LIKE TRAINING
# =========================================================

def load_model_and_data():
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = get_dataset(dataset_root="./data", dataset_name=DATASET_NAME)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()

    config = ModelConfig()

    print("Building model...")
    model = get_gnnNets(
        input_dim=dataset.num_node_features,
        output_dim=NUM_CLASSES,
        model_config=config
    )

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(ckpt["net"])
    model.basis_data = ckpt["basis_data"]

    model.to(DEVICE)
    model.eval()

    return model, dataset


# =========================================================
# COMPUTE ACTIVATION & WIN COUNTS
# =========================================================

def compute_activation(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    num_proto = NUM_CLASSES * NUM_BASIS_PER_CLASS
    activation_sum = torch.zeros(num_proto).to(DEVICE)
    win_count = torch.zeros(num_proto).to(DEVICE)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)

            logits, emb, l2s = model(batch)

            # Activation = exp(-distance)
            activation = torch.exp(-l2s)
            activation_sum += activation.mean(dim=0)

            # Which prototype wins each node
            winners = torch.argmin(l2s, dim=1)
            for w in winners:
                win_count[w] += 1

    return activation_sum.cpu(), win_count.cpu()


# =========================================================
# COMPUTE PROTOTYPE SIMILARITY MATRIX
# =========================================================

def compute_similarity(model):
    # detach to prevent grad issues
    P = model.basis_concepts.detach()
    Pn = F.normalize(P, dim=0)

    sim = torch.matmul(Pn.T, Pn)

    return sim.detach().cpu().numpy()


# =========================================================
# PRUNE RULES
# =========================================================

def prune_prototypes(model, activation, wins, sim_matrix):
    num_proto = NUM_CLASSES * NUM_BASIS_PER_CLASS
    keep = np.ones(num_proto, dtype=bool)

    print("\n=== PRUNING STARTED ===\n")

    # 1. Low activation
    for i in range(num_proto):
        if activation[i] < ACTIVATION_THRESHOLD:
            keep[i] = False
            print(f"[LOW ACTIVATION] Prototype {i} removed")

    # 2. Never used
    for i in range(num_proto):
        if wins[i] == 0:
            keep[i] = False
            print(f"[NO WINS] Prototype {i} removed")

    # 3. Duplicate prototypes
    for i in range(num_proto):
        for j in range(i + 1, num_proto):
            if keep[j] and sim_matrix[i][j] > SIMILARITY_THRESHOLD:
                keep[j] = False
                print(f"[DUPLICATE] Prototype {j} removed (similar to {i})")

    return keep


# =========================================================
# SAVE NEW MODEL WITH FEWER PROTOTYPES
# =========================================================

def save_pruned_model(model, keep_mask):
    P = model.basis_concepts.detach().cpu().numpy()
    P_new = P[:, keep_mask]

    model.basis_concepts = torch.nn.Parameter(torch.tensor(P_new).to(DEVICE))

    torch.save({
        "net": model.state_dict(),
        "keep_mask": keep_mask
    }, SAVE_PRUNED_MODEL_PATH)

    print(f"\nSaved pruned model → {SAVE_PRUNED_MODEL_PATH}\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    model, dataset = load_model_and_data()

    print("\nComputing activation...")
    activation, wins = compute_activation(model, dataset)

    print("\nComputing similarity...")
    sim_matrix = compute_similarity(model)

    print("\nActivation:", activation)
    print("Wins:", wins)

    keep_mask = prune_prototypes(model, activation, wins, sim_matrix)

    print("\nFinal kept prototypes:", np.where(keep_mask)[0])

    save_pruned_model(model, keep_mask)

    print("\n=== PRUNING COMPLETE ===")
