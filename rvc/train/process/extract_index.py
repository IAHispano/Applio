import os
import sys
from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Fixed so the same dataset always produces the same index.
SEED = 1234

# Upper bound on the vectors an index keeps. Inference reconstructs all of them
# (768 dims, float32 -> 293 MB at 100k), so this is the retrieval's memory cost
# as much as its fidelity. 200k is what the previous size test already allowed
# through untouched, so nothing built today gets larger than it could before.
MAX_INDEX_VECTORS = int(os.environ.get("APPLIO_INDEX_MAX_VECTORS", 200_000))

# Cluster count for the explicit "KMeans" algorithm, clamped to the frames
# available: MiniBatchKMeans raises outright when asked for more clusters than
# it has samples.
KMEANS_CLUSTERS = 10_000

# How many IVF cells a search visits. This is stored inside the .index file, so
# the value chosen here is the one every future search uses. It was 1, which
# means a search looked in a single cell out of thousands and returned whatever
# it found there as the "nearest" neighbours.
NPROBE_LADDER = (1, 2, 4, 8, 16, 32, 64)
NPROBE_TARGET_RECALL = 0.95
RECALL_QUERIES = 128
RECALL_K = 8


def load_features(feature_dir):
    """
    Load every extracted feature file into one array.

    Args:
        feature_dir (str): Directory holding the extracted .npy features.
    """
    npys = []
    dropped = 0
    for name in sorted(os.listdir(feature_dir)):
        if not name.endswith(".npy"):
            continue
        phone = np.load(os.path.join(feature_dir, name))
        if phone.ndim != 2 or phone.shape[0] == 0:
            print(f"Skipping {name}: unexpected shape {phone.shape}.")
            continue
        # A NaN vector matches nothing and poisons the k-means fit; extraction
        # already skips files it detects, but an index is cheap to protect.
        finite = np.isfinite(phone).all(axis=1)
        dropped += int((~finite).sum())
        if finite.any():
            npys.append(np.asarray(phone[finite], dtype=np.float32))

    if dropped:
        print(f"Dropped {dropped} non-finite frames.")
    if not npys:
        return None
    return np.ascontiguousarray(np.concatenate(npys, axis=0))


def reduce_features(big_npy, index_algorithm, rng):
    """
    Bring the vector count under the cap, or apply the requested clustering.

    Args:
        big_npy (np.ndarray): All extracted feature frames.
        index_algorithm (str): "Auto", "Faiss" or "KMeans".
        rng (np.random.Generator): Seeded generator, for a reproducible index.
    """
    total = big_npy.shape[0]

    if index_algorithm == "KMeans":
        # Clamped because MiniBatchKMeans raises when n_clusters exceeds
        # n_samples, which made this option unusable on a small dataset.
        clusters = min(KMEANS_CLUSTERS, total)
        print(f"Running KMeans: {total} vectors -> {clusters} centroids...")
        return np.ascontiguousarray(
            MiniBatchKMeans(
                n_clusters=clusters,
                verbose=True,
                batch_size=256 * cpu_count(),
                compute_labels=False,
                init="random",
                random_state=SEED,
            )
            .fit(big_npy)
            .cluster_centers_,
            dtype=np.float32,
        )

    if index_algorithm == "Faiss" or total <= MAX_INDEX_VECTORS:
        return big_npy

    # Over the cap on the automatic path. Keeping a random sample of real
    # frames rather than collapsing to a few thousand centroids: the retrieval
    # pulls a query toward something the speaker actually produced, and a
    # centroid is an average that no longer is. Clustering also used to make a
    # larger dataset produce a *smaller* index than a smaller one.
    print(f"Keeping a random {MAX_INDEX_VECTORS} of {total} vectors...")
    keep = np.sort(rng.choice(total, size=MAX_INDEX_VECTORS, replace=False))
    return np.ascontiguousarray(big_npy[keep])


def measure_recall(index, big_npy, rng, nprobe_values):
    """
    Top-k recall of the IVF search against an exact one, per nprobe value.

    An index that searches a single cell still loads, still returns results and
    still looks like it works; the neighbours it returns are simply not the
    nearest. Measuring is the only way that surfaces.

    Args:
        index (faiss.Index): The trained index.
        big_npy (np.ndarray): The vectors it was built from.
        rng (np.random.Generator): Seeded generator.
        nprobe_values (list of int): Values to measure, in order.
    """
    count = big_npy.shape[0]
    if count <= RECALL_K:
        return {}

    sample = rng.choice(count, size=min(RECALL_QUERIES, count), replace=False)
    probe = np.ascontiguousarray(big_npy[sample])

    # The exact answer does not depend on nprobe, so it is computed once.
    norms = (big_npy**2).sum(axis=1)
    exact = np.empty((probe.shape[0], RECALL_K), dtype=np.int64)
    for start in range(0, probe.shape[0], 32):
        block = probe[start : start + 32]
        scores = norms[None, :] - 2.0 * (block @ big_npy.T)
        exact[start : start + 32] = np.argpartition(scores, RECALL_K, axis=1)[
            :, :RECALL_K
        ]

    index_ivf = faiss.extract_index_ivf(index)
    recalls = {}
    for nprobe in nprobe_values:
        index_ivf.nprobe = nprobe
        _, approximate = index.search(probe, RECALL_K)
        recalls[nprobe] = float(
            np.mean(
                [len(set(a) & set(b)) / RECALL_K for a, b in zip(approximate, exact)]
            )
        )
    return recalls


def tune_nprobe(index, big_npy, rng):
    """
    Raise nprobe until the search finds the neighbours it claims to.

    The value is stored inside the .index file, so this runs once here rather
    than being guessed at every search.

    Args:
        index (faiss.Index): The trained index.
        big_npy (np.ndarray): The vectors it was built from.
        rng (np.random.Generator): Seeded generator.
    """
    index_ivf = faiss.extract_index_ivf(index)
    ladder = sorted({min(int(index_ivf.nlist), value) for value in NPROBE_LADDER})

    recalls = measure_recall(index, big_npy, rng, ladder)
    if not recalls:
        index_ivf.nprobe = ladder[-1]
        return index_ivf.nprobe, None

    for nprobe in ladder:
        if recalls[nprobe] >= NPROBE_TARGET_RECALL:
            index_ivf.nprobe = nprobe
            return nprobe, recalls[nprobe]

    best = max(recalls, key=recalls.get)
    index_ivf.nprobe = best
    return best, recalls[best]


# Parse command line arguments
exp_dir = str(sys.argv[1])
index_algorithm = str(sys.argv[2])

feature_dir = os.path.join(exp_dir, "extracted")
model_name = os.path.basename(exp_dir)

if not os.path.exists(feature_dir):
    print(
        f"Feature to generate index file not found at {feature_dir}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

index_filename_added = f"{model_name}.index"
index_filepath_added = os.path.join(exp_dir, index_filename_added)

# Regenerating used to be a silent no-op, so changing the algorithm and running
# again quietly handed back the previous index.
if os.path.exists(index_filepath_added):
    print(f"Replacing the existing index at '{index_filepath_added}'.")

print(f"Generating index for '{model_name}', this may take a while...")
big_npy = load_features(feature_dir)

if big_npy is None:
    print(
        f"Feature files in {feature_dir} could not be loaded correctly. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

rng = np.random.default_rng(SEED)
big_npy = reduce_features(big_npy, index_algorithm, rng)

n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))

# index_added
index_added = faiss.index_factory(768, f"IVF{n_ivf},Flat")
index_added.train(big_npy)

batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index_added.add(big_npy[i : i + batch_size_add])

nprobe, recall = tune_nprobe(index_added, big_npy, rng)

faiss.write_index(index_added, index_filepath_added)
print(
    f"Index: {big_npy.shape[0]} vectors, IVF{n_ivf}, nprobe {nprobe}"
    + (f", top-{RECALL_K} recall {recall:.1%}" if recall is not None else "")
)
if recall is not None and recall < NPROBE_TARGET_RECALL:
    print(
        "Recall stays low at the highest nprobe tried; searches will return "
        "some neighbours that are not the nearest."
    )
print(f"Saved index file '{index_filepath_added}'")
