import argparse
import os
import json

import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from umap import UMAP

from .datasets import load_planetoid
from .deepwalk import train_deepwalk_embeddings
from .metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer"])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=80)
    parser.add_argument("--walks_per_node", type=int, default=10)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_planetoid(args.dataset, data_dir="data")
    adj = data["adjacency"]
    labels = data["labels"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]

    print("Building NetworkX graph...")
    G = nx.from_scipy_sparse_array(adj)

    print("Training DeepWalk embeddings...")
    emb, idx_map = train_deepwalk_embeddings(
        G,
        dim=args.dim,
        window=args.window,
        negative=args.negative,
        walks_per_node=args.walks_per_node,
        walk_length=args.walk_length,
        seed=args.seed,
    )

    # Train a simple classifier
    clf = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=2000, multi_class="auto")),
        ]
    )
    clf.fit(emb[idx_train], labels[idx_train])

    proba_test = clf.predict_proba(emb[idx_test])
    metrics = compute_metrics(labels[idx_test], proba_test)

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", f"deepwalk_{args.dataset}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 2D visualization of embeddings
    os.makedirs("artifacts/plots", exist_ok=True)
    umap = UMAP(n_components=2, random_state=args.seed)
    z = umap.fit_transform(emb)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], c=labels, s=5)
    plt.title(f"DeepWalk Embeddings — {args.dataset}")
    plt.tight_layout()
    plt.savefig(os.path.join("artifacts/plots", f"deepwalk_{args.dataset}_umap.png"), dpi=200)

    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
