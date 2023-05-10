# Fork Feature Mangio RVC Fork. Train the feature index (faiss) through the cli

import os, sys, warnings, shutil, numpy as np
import faiss

# Fork Feature: Get System Args
model_name = sys.argv[1]

fail_msg = "Training Failed: Please perform feature extraction first."

now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")

def train_index(exp_dir1):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = "%s/3_feature256" % (exp_dir)
    if os.path.exists(feature_dir) == False:
        return fail_msg
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return fail_msg
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield "%s,%s \n" % (big_npy.shape, n_ivf)
    index = faiss.index_factory(256, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    yield "training the index...\n"
    index_ivf = faiss.extract_index_ivf(index)  #
    # index_ivf.nprobe = int(np.power(n_ivf,0.3))
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan.index'%(exp_dir,n_ivf))
    yield "adding the index... \n"
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe),
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan.index'%(exp_dir,n_ivf))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan.index"%(n_ivf))
    yield "Done! added_IVF%s_Flat_nprobe_%s.index \n" % (n_ivf, index_ivf.nprobe)

train_output = train_index(model_name)
if(train_output == fail_msg):
    print("Mangio-RVC-Fork Feature Training: " + train_output)
else:
    for log in train_output:
        print("Mangio-RVC-Fork Feature Training: %s" % (log))