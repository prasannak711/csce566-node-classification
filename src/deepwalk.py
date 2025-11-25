import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec


def random_walks(G: nx.Graph, walk_length=80, walks_per_node=10, seed=42):
    rnd = random.Random(seed)
    walks = []
    nodes = list(G.nodes())
    for _ in range(walks_per_node):
        rnd.shuffle(nodes)
        for n in nodes:
            walk = [n]
            while len(walk) < walk_length:
                cur = walk[-1]
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                walk.append(rnd.choice(nbrs))
            walks.append([str(x) for x in walk])  # Word2Vec expects strings
    return walks


def train_deepwalk_embeddings(
    G,
    dim=128,
    window=10,
    negative=5,
    walks_per_node=10,
    walk_length=80,
    workers=1,
    seed=42,
):
    walks = random_walks(
        G,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        seed=seed,
    )
    model = Word2Vec(
        sentences=walks,
        vector_size=dim,
        window=window,
        sg=1,              # skip-gram
        negative=negative,
        min_count=0,
        workers=workers,
        epochs=5,
        seed=seed,
    )
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    emb = np.zeros((len(nodes), dim), dtype=np.float32)
    for n in nodes:
        emb[idx_map[n]] = model.wv[str(n)]
    return emb, idx_map
