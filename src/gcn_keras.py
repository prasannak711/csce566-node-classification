import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers, Model


def normalize_adj(adj: sp.spmatrix):
    """
    Symmetrically normalize adjacency matrix and add self-loops:
    Ā = D^{-1/2} (A + I) D^{-1/2}
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format="coo")
    rowsum = np.array(adj_.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv = sp.diags(d_inv_sqrt)
    return (D_inv @ adj_ @ D_inv).tocoo()


class GCNLayer(layers.Layer):
    """
    Simple GCN layer: H^{(l+1)} = σ(Â H^{(l)} W)
    """

    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        fin = int(input_shape[0][-1])  # input feature dimension
        self.w = self.add_weight(
            shape=(fin, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="w",
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="b",
            )

    def call(self, inputs):
        x, adj = inputs  # x: [N, F], adj: SparseTensor [N, N]
        xw = tf.matmul(x, self.w)  # [N, units]
        out = tf.sparse.sparse_dense_matmul(adj, xw)
        if self.use_bias:
            out = out + self.b
        return out


def build_gcn(n_nodes, n_feats, n_classes, hidden=64, dropout=0.5, lr=1e-2):
    """
    Build a 2-layer GCN model with dropout and softmax output.
    Full-batch training (all nodes at once).
    """
    x_in = layers.Input(shape=(n_feats,), name="x_in")
    a_in = layers.Input((n_nodes,), sparse=True, name="a_in")

    h = layers.Dropout(dropout)(x_in)
    h = GCNLayer(hidden)([h, a_in])
    h = layers.Activation("relu")(h)
    h = layers.Dropout(dropout)(h)
    h = GCNLayer(n_classes)([h, a_in])
    out = layers.Activation("softmax")(h)

    model = Model(inputs=[x_in, a_in], outputs=out)
    optimizer = tf.keras.optimizers.Adam(lr)

    return model, optimizer


def to_sparse_tensor(adj: sp.coo_matrix):
    indices = np.vstack((adj.row, adj.col)).T.astype(np.int64)
    return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
