import os
import sys
from multiprocessing import cpu_count

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Parse command line arguments
exp_dir = str(sys.argv[1])
index_algorithm = str(sys.argv[2])

try:
    feature_dir = os.path.join(exp_dir, f"extracted")
    model_name = os.path.basename(exp_dir)

    if not os.path.exists(feature_dir):
        print(
            f"Feature to generate index file not found at {feature_dir}. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)

    index_filename_added = f"{model_name}.index"
    index_filepath_added = os.path.join(exp_dir, index_filename_added)

    if os.path.exists(index_filepath_added):
        pass
    else:
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

        if big_npy.shape[0] > 2e5 and (
            index_algorithm == "Auto" or index_algorithm == "KMeans"
        ):
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

        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)

        # index_added
        index_added = faiss.index_factory(768, f"IVF{n_ivf},Flat")
        index_ivf_added = faiss.extract_index_ivf(index_added)
        index_ivf_added.nprobe = 1
        index_added.train(big_npy)

        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index_added.add(big_npy[i : i + batch_size_add])

        faiss.write_index(index_added, index_filepath_added)
        print(f"Saved index file '{index_filepath_added}'")

except Exception as error:
    print(f"An error occurred extracting the index: {error}")
    print(
        "If you are running this code in a virtual environment, make sure you have enough GPU available to generate the Index file."
    )
