import argparse
import os
import json

import numpy as np
import tensorflow as tf

from .datasets import load_planetoid
from .gcn_keras import build_gcn, normalize_adj, to_sparse_tensor
from .metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "citeseer"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data = load_planetoid(args.dataset, data_dir="data")
    features = data["features"].toarray().astype("float32")  # [N, F]
    adj = normalize_adj(data["adjacency"])
    labels = data["labels"].astype("int32")
    idx_train = data["idx_train"]
    idx_val = data["idx_val"]
    idx_test = data["idx_test"]

    n_nodes, n_feats = features.shape
    n_classes = int(labels.max()) + 1

    # Build model
    model, optimizer = build_gcn(
        n_nodes=n_nodes,
        n_feats=n_feats,
        n_classes=n_classes,
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
    )

    adj_tf = to_sparse_tensor(adj)
    adj_tf = tf.sparse.reorder(adj_tf)

    # Masks
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    x_tf = tf.convert_to_tensor(features)
    y_tf = tf.convert_to_tensor(labels)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            logits = model([x_tf, adj_tf], training=True)  # [N, C]
            loss_all = loss_fn(y_tf, logits)              # [N]
            mask = tf.cast(train_mask, loss_all.dtype)
            mask = mask / tf.reduce_mean(mask)
            loss = tf.reduce_mean(loss_all * mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Training loop
    best_val_loss = np.inf
    best_state = None
    patience = 50
    wait = 0

    for epoch in range(args.epochs):
        loss = train_step()

        # Evaluate on val set
        logits = model([x_tf, adj_tf], training=False).numpy()
        val_probs = logits[val_mask]
        val_labels = labels[val_mask]
        if val_probs.shape[0] > 0:
            # Use cross-entropy on val nodes
            val_loss_all = tf.keras.losses.sparse_categorical_crossentropy(
                val_labels, val_probs
            ).numpy()
            val_loss = val_loss_all.mean()
        else:
            val_loss = float(loss.numpy())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_state = model.get_weights()
        else:
            wait += 1
            if wait >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break

        # Optional: uncomment to see training progress
        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch:03d} | train_loss={loss.numpy():.4f} | val_loss={val_loss:.4f}")

    # Restore best weights
    if best_state is not None:
        model.set_weights(best_state)

    # Final evaluation on test set
    logits = model([x_tf, adj_tf], training=False).numpy()
    test_probs = logits[test_mask]
    test_labels = labels[test_mask]
    metrics = compute_metrics(test_labels, test_probs)

    os.makedirs("artifacts", exist_ok=True)
    out_path = os.path.join("artifacts", f"gcn_{args.dataset}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
