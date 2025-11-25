import os
from pathlib import Path
import pickle
import requests
import numpy as np
import scipy.sparse as sp
import networkx as nx

# URLs for the Planetoid Cora and Citeseer datasets
PLANETOID_URLS = {
    "cora": {
        "x": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x",
        "tx": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.tx",
        "allx": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx",
        "y": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y",
        "ty": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty",
        "ally": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally",
        "graph": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph",
        "test_index": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index",
    },
    "citeseer": {
        "x": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.x",
        "tx": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.tx",
        "allx": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.allx",
        "y": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.y",
        "ty": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.ty",
        "ally": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.ally",
        "graph": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.graph",
        "test_index": "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.test.index",
    },
}


def _pickle_load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _maybe_download(dataset_name: str, data_dir: str):
    urls = PLANETOID_URLS[dataset_name]
    out_dir = Path(data_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, url in urls.items():
        local_path = out_dir / f"{dataset_name}.{key}"
        if not local_path.exists():
            print(f"Downloading {key} for {dataset_name}...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        paths[key] = str(local_path)
    return paths


def load_planetoid(name: str, data_dir: str = "data"):
    """
    Load Cora or Citeseer in GCN format.
    Handles Citeseer isolated nodes correctly.
    Returns dict with:
      features (csr), adjacency (csr), labels (np array),
      idx_train, idx_val, idx_test, num_classes
    """
    name = name.lower()
    if name not in PLANETOID_URLS:
        raise ValueError(f"Unsupported dataset: {name}")

    paths = _maybe_download(name, data_dir)

    x = _pickle_load(paths["x"])        # train features
    tx = _pickle_load(paths["tx"])      # test features
    allx = _pickle_load(paths["allx"])  # train+val features
    y = _pickle_load(paths["y"])        # train labels
    ty = _pickle_load(paths["ty"])      # test labels
    ally = _pickle_load(paths["ally"])  # train+val labels
    graph = _pickle_load(paths["graph"])
    test_index = np.loadtxt(paths["test_index"], dtype=int)

    # --- Special fix for Citeseer isolated nodes ---
    if name == "citeseer":
        # In Citeseer, some test indices are not consecutive, so we must
        # "expand" tx and ty to cover the full test index range.
        test_idx_sorted = np.sort(test_index)
        test_idx_full = np.arange(test_idx_sorted[0], test_idx_sorted[-1] + 1)

        # Extend test features
        tx_ext = sp.lil_matrix((len(test_idx_full), x.shape[1]))
        tx_ext[test_idx_sorted - test_idx_sorted[0], :] = tx
        tx = tx_ext

        # Extend test labels
        ty_ext = np.zeros((len(test_idx_full), y.shape[1]))
        ty_ext[test_idx_sorted - test_idx_sorted[0], :] = ty
        ty = ty_ext

        # Use the full continuous test index range
        test_index = test_idx_full

    # Standard Planetoid splits
    idx_test = test_index
    idx_train = np.arange(y.shape[0])
    idx_val = np.arange(y.shape[0], y.shape[0] + 500)

    # Build full feature matrix
    features = sp.vstack((allx, tx)).tolil()
    features[idx_test, :] = features[idx_test, :]
    features = features.tocsr().astype("float32")

    # Build full labels matrix and integer label ids
    labels = np.vstack((ally, ty))
    labels[idx_test, :] = labels[idx_test, :]
    y_ids = labels.argmax(axis=1).astype("int64")

    # Build adjacency from graph dict
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G).astype("float32")

    num_classes = labels.shape[1]

    return {
        "features": features,
        "adjacency": adj,
        "labels": y_ids,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "num_classes": num_classes,
    }
