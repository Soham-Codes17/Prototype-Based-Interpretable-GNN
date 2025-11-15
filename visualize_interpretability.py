# visualize_interpretability.py
# Final dynamic visualizer for Prototype-based GNNs (works with pruned & unpruned checkpoints)

import os
import json
import torch
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader as PyGDataLoader

from dataset import get_dataset
from protop_gnnNets import get_gnnNets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = "interpretability_output"
os.makedirs(OUTDIR, exist_ok=True)

# MUTAG atom mapping (your dataset uses 7-dim one-hot)
ATOM_TYPES = ["C", "N", "O", "F", "Cl", "Br", "I"]


def find_latest_checkpoint(root_dir="protop_checkpoints_pretrained"):
    """Return latest .pth under root_dir. Raises if none."""
    latest_file = None
    latest_mtime = -1
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Checkpoint root '{root_dir}' not found.")
    for rt, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".pth"):
                p = os.path.join(rt, f)
                t = os.path.getmtime(p)
                if t > latest_mtime:
                    latest_mtime = t
                    latest_file = p
    if latest_file is None:
        raise FileNotFoundError(f"No .pth found under {root_dir}")
    print("Auto-loaded checkpoint ->", latest_file)
    return latest_file


def pyg_to_rdkit(data):
    """Convert a PyG Data object to an RDKit Mol (best-effort)."""
    mol = Chem.RWMol()
    node_to_idx = {}
    x = data.x
    for i in range(x.size(0)):
        # assume one-hot or feature where argmax gives atom type index
        idx = int(torch.argmax(x[i]).item())
        if idx < 0 or idx >= len(ATOM_TYPES):
            symbol = "C"
        else:
            symbol = ATOM_TYPES[idx]
        atom = Chem.Atom(symbol)
        node_to_idx[i] = mol.AddAtom(atom)
    # add edges (undirected edge_index)
    if hasattr(data, "edge_index"):
        ei = data.edge_index
        for k in range(ei.size(1)):
            u = int(ei[0, k])
            v = int(ei[1, k])
            if u < v and u in node_to_idx and v in node_to_idx:
                try:
                    mol.AddBond(node_to_idx[u], node_to_idx[v], Chem.rdchem.BondType.SINGLE)
                except Exception:
                    pass
    m = mol.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def load_checkpoint_and_model(checkpoint_path=None, data_root="./data"):
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    # dataset name saved? fallback to 'mutag'
    dataset_name = ckpt.get("dataset_name", "mutag")
    print("Dataset detected ->", dataset_name)

    dataset = get_dataset(data_root, dataset_name)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()

    input_dim = dataset.num_node_features
    num_classes = int(dataset.data.y.max().item() + 1)
    print(f"Num classes -> {num_classes}, node features -> {input_dim}")

    # extract basis_concepts tensor from checkpoint state dict
    net = ckpt.get("net", {})
    basis_key = None
    if isinstance(net, dict):
        # keys might be direct or module-prefixed; find any key containing 'basis_concepts'
        for k in net.keys():
            if "basis_concepts" in k:
                basis_key = k
                break
    if basis_key is None:
        raise KeyError("Could not find 'basis_concepts' in checkpoint['net']")

    basis_tensor = net[basis_key]
    basis_dim = int(basis_tensor.shape[0])
    prototype_count = int(basis_tensor.shape[1])
    print(f"Detected basis_dim={basis_dim}, prototype_count={prototype_count}")

    # attempt to infer num_basis_per_class if available in checkpoint
    # fallback to dividing prototypes by classes (floor)
    num_basis_per_class = ckpt.get("num_basis_per_class", None)
    if num_basis_per_class is None:
        if num_classes > 0:
            num_basis_per_class = prototype_count // num_classes
        else:
            num_basis_per_class = prototype_count

    # build a tiny config-compatible object for get_gnnNets
    class HybridDict(dict):
        def __init__(self, **kw):
            super().__init__(**kw)
            for k, v in kw.items():
                setattr(self, k, v)

    class ModelConfig:
        def __init__(self):
            self.gnn_name = ckpt.get("gnn_name", "gcn")
            self.param = HybridDict(
                basis_dim=basis_dim,
                num_basis_per_class=num_basis_per_class,
                gnn_latent_dim=ckpt.get("gnn_latent_dim", [128, 128, 128]),
                fc_latent_dim=ckpt.get("fc_latent_dim", [128]),
                gnn_dropout=ckpt.get("gnn_dropout", 0.0),
                fc_dropout=ckpt.get("fc_dropout", 0.0),
                gnn_emb_normalization=ckpt.get("gnn_emb_normalization", False),
                gcn_adj_normalization=ckpt.get("gcn_adj_normalization", True),
                add_self_loop=True,
                gnn_nonlinear=ckpt.get("gnn_nonlinear", "relu"),
                fc_nonlinear=ckpt.get("fc_nonlinear", "relu"),
                readout=ckpt.get("readout", "mean"),
            )

    config = ModelConfig()
    model = get_gnnNets(input_dim, num_classes, config)

    # load matching keys (robust)
    try:
        model.load_state_dict(net)
    except Exception:
        # selective loading for matching shapes
        model_state = model.state_dict()
        matched = {k: v for k, v in net.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        model.load_state_dict(model_state)
        print("Loaded matching keys from checkpoint (some keys skipped due to shape mismatch).")

    model.to(DEVICE).eval()

    # handle keep_mask if present (mapping from original prototype indices)
    keep_mask = ckpt.get("keep_mask", None)
    original_indices = None
    if keep_mask is not None:
        # keep_mask may be numpy array, list, or tensor; coerce to list of ints
        try:
            keep_arr = np.array(keep_mask)
            # if keep_mask is boolean mask -> indices where True
            if keep_arr.dtype == bool:
                original_indices = np.where(keep_arr)[0].tolist()
            else:
                original_indices = [int(x) for x in keep_arr.tolist()]
        except Exception:
            # fallback
            original_indices = [int(x) for x in list(keep_mask)]
        # If checkpoint basis_concepts is pruned (columns == len(original_indices)),
        # treat position i -> original_indices[i]
        if len(original_indices) != prototype_count:
            # If lengths differ, try to use the smaller of them:
            if len(original_indices) > prototype_count:
                # maybe keep_mask was a full-length mask while basis_concepts is pruned
                original_indices = original_indices[:prototype_count]
            else:
                # fewer original indices than prototype_count — fallback to positions
                pass
        print("keep_mask detected. Prototype position -> original index mapping will be used.")
    else:
        print("No keep_mask in checkpoint; prototypes labeled by model positions.")

    return model, dataset, original_indices, prototype_count, checkpoint_path


def find_best_matches(model, dataset, proto_positions):
    """
    proto_positions: list of prototype *positions* to evaluate (0..M-1)
    returns dict mapping proto_position -> info (including 'data' and 'node_l2s')
    """
    results = {int(p): {"best_graph_idx": None, "best_node_idx": None, "best_l2": float("inf"),
                        "data": None, "node_l2s": None} for p in proto_positions}

    loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)

    for gi, batch in enumerate(loader):
        batch = batch.to(DEVICE)
        with torch.no_grad():
            _, _, l2s = model(batch)  # l2s: (num_nodes, total_prototypes)
        l2s_np = l2s.detach().cpu().numpy()  # shape (N_nodes, P)
        for p in proto_positions:
            if p >= l2s_np.shape[1]:
                continue
            node_idx = int(np.argmin(l2s_np[:, p]))
            val = float(l2s_np[node_idx, p])
            if val < results[p]["best_l2"]:
                results[p]["best_l2"] = val
                results[p]["best_graph_idx"] = gi
                results[p]["best_node_idx"] = node_idx
                results[p]["node_l2s"] = l2s_np[:, p].copy()
                results[p]["data"] = batch.cpu().clone()
    return results


def save_highlighted_molecule(outdir, proto_label, graph_data, node_l2s):
    os.makedirs(outdir, exist_ok=True)
    mol = pyg_to_rdkit(graph_data)
    plain = os.path.join(outdir, f"proto_{proto_label}_plain.png")
    Draw.MolToFile(mol, plain, size=(420, 420))

    atom_scores = -np.array(node_l2s, dtype=float)
    if atom_scores.size == 0:
        highlight = plain
    else:
        mn, mx = atom_scores.min(), atom_scores.max()
        if mx == mn:
            norm = np.zeros_like(atom_scores)
            norm[np.argmax(atom_scores)] = 1.0
        else:
            norm = (atom_scores - mn) / (mx - mn)
        n_atoms = mol.GetNumAtoms()
        k = max(1, int(math.ceil(0.15 * n_atoms)))
        top_idx = np.argsort(-norm)[:k].tolist()
        # build highlight colors map (simple constant color)
        colors = {int(i): (1.0, 0.6, 0.2) for i in top_idx}
        highlight = os.path.join(outdir, f"proto_{proto_label}_highlight.png")
        try:
            img = Draw.MolToImage(mol, size=(420, 420), highlightAtoms=top_idx, highlightAtomColors=colors)
            img.save(highlight)
        except Exception:
            # fallback to SVG
            try:
                from rdkit.Chem.Draw import rdMolDraw2D
                drawer = rdMolDraw2D.MolDraw2DSVG(420, 420)
                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=top_idx, highlightAtomColors=colors)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                with open(highlight.replace(".png", ".svg"), "w") as f:
                    f.write(svg)
                highlight = highlight.replace(".png", ".svg")
            except Exception:
                highlight = plain

    return plain, highlight


def visualize_and_save(checkpoint_path=None, data_root="./data", outdir=OUTDIR):
    model, dataset, original_indices, proto_count, ckpt_path = load_checkpoint_and_model(checkpoint_path, data_root)

    # prototype positions are 0..proto_count-1 (columns in model.basis_concepts)
    proto_positions = list(range(proto_count))

    matches = find_best_matches(model, dataset, proto_positions)

    saved_info = {}
    for pos in proto_positions:
        info = matches[pos]
        if info["data"] is None:
            # no nodes/graphs matched for this prototype
            continue

        # label: if original_indices present, map; else use position
        if original_indices is not None:
            try:
                label = int(original_indices[pos])  # original index
            except Exception:
                label = int(pos)
        else:
            label = int(pos)

        plain, highlight = save_highlighted_molecule(outdir, label, info["data"], info["node_l2s"])

        saved_info[int(label)] = {
            "proto_position": int(pos),
            "graph_index": int(info["best_graph_idx"]) if info["best_graph_idx"] is not None else None,
            "best_node": int(info["best_node_idx"]) if info["best_node_idx"] is not None else None,
            "best_l2": float(info["best_l2"]),
            "plain_image": plain,
            "highlight_image": highlight
        }
        print(f"Saved proto {label} (pos {pos}) -> {highlight}")

    # JSON keys must be strings — convert keys to strings and all numpy types to native
    json_safe = {str(k): {kk: (int(vv) if (isinstance(vv, (np.integer,)) or isinstance(vv, int)) else (float(vv) if isinstance(vv, (np.floating, float)) else vv)) for kk, vv in v.items()} for k, v in saved_info.items()}

    with open(os.path.join(outdir, "visualization_summary.json"), "w") as fh:
        json.dump(json_safe, fh, indent=2)

    print(f"\nSaved summary -> {os.path.join(outdir, 'visualization_summary.json')}")
    return json_safe


if __name__ == "__main__":
    visualize_and_save()
