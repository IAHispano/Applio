import os
import sys
import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import cpu_count

exp_dir = sys.argv[1]
version = sys.argv[2]

try:
    if version == "v1":
        feature_dir = os.path.join(exp_dir, "3_feature256")
    elif version == "v2":
        feature_dir = os.path.join(exp_dir, "3_feature768")

    npys = []
    listdir_res = sorted(os.listdir(feature_dir))

    for name in listdir_res:
        file_path = os.path.join(feature_dir, name)
        phone = np.load(file_path)
        npys.append(phone)

    big_npy = np.concatenate(npys, axis=0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        big_npy = (
            MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * cpu_count(),
                compute_labels=False,
                init="random",
            )
            .fit(big_npy)
            .cluster_centers_
        )

    np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)

    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")

    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    index_filename = (
        f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{version}.index"
    )
    index_filepath = os.path.join(exp_dir, index_filename)

    faiss.write_index(index, index_filepath)

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])

    faiss.write_index(index, index_filepath)

except Exception as error:
    print(f"Failed to train index: {error}")

print("Index training finished!")
