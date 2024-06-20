import os
import sys
import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from multiprocessing import cpu_count

exp_dir = sys.argv[1]
version = sys.argv[2]

try:
    feature_dir = os.path.join(exp_dir, f"{version}_extracted")
    model_name = os.path.basename(exp_dir)

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

    # index_trained
    index_trained = faiss.index_factory(
        256 if version == "v1" else 768, f"IVF{n_ivf},Flat"
    )
    index_ivf_trained = faiss.extract_index_ivf(index_trained)
    index_ivf_trained.nprobe = 1
    index_trained.train(big_npy)

    index_filename_trained = f"trained_{model_name}_{version}.index"
    index_filepath_trained = os.path.join(exp_dir, index_filename_trained)

    faiss.write_index(index_trained, index_filepath_trained)

    # index_added
    index_added = faiss.index_factory(
        256 if version == "v1" else 768, f"IVF{n_ivf},Flat"
    )
    index_ivf_added = faiss.extract_index_ivf(index_added)
    index_ivf_added.nprobe = 1
    index_added.train(big_npy)

    index_filename_added = f"added_{model_name}_{version}.index"
    index_filepath_added = os.path.join(exp_dir, index_filename_added)

    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index_added.add(big_npy[i : i + batch_size_add])

    faiss.write_index(index_added, index_filepath_added)
    print(f"Saved index file '{index_filepath_added}'")

except Exception as error:
    print(f"Failed to train index: {error}")
    if "one array to concatenate" in str(error):
        print(
            "If you are running this code in a virtual environment, make sure you have enough GPU available to generate the Index file."
        )
