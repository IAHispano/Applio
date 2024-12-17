from rvc.lib.algorithm.xeus.xeus import XeusModel
import rvc.lib.zluda
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(tensor):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	X, Y = np.meshgrid(np.arange(tensor.shape[1]), np.arange(tensor.shape[0]))
	Z = tensor.squeeze(0)

	ax.plot_surface(X, Y, Z, cmap='viridis')
	ax.set_xlabel('Channel')
	ax.set_ylabel('Time')
	ax.set_zlabel('Value')
	plt.show()

device = "cuda"

wav, sampling_rate = librosa.load(r"X:\Applio\logs\Book_44k_v3_ref\sliced_audios_16k\0_0_0.wav", sr=16000)
wav_length = torch.LongTensor([len(wav)]).to(device)
wav= torch.FloatTensor(wav).unsqueeze(0).to(device)
model = XeusModel(model_path=r"X:\Applio\rvc\models\embedders\Xeus\xeus_checkpoint.pth")
model = model.to(device)
model.eval()

v3_feats = model.encode(wav, wav_length)

print(v3_feats)
print(v3_feats.shape)
