import os
import faiss
import torch
import torch.nn.functional as F


def circular_write(new_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    offset = new_data.shape[0]
    target[:-offset] = target[offset:].detach().clone()
    target[-offset:] = new_data
    return target


def frame(x: torch.Tensor, frame_length: int, hop_length: int, axis: int = -1):
    """
    Slice a tensor into overlapping frames using stride tricks.

    Args:
        x (torch.Tensor): Input tensor containing the signal data.
        frame_length (int): Number of samples in each frame.
        hop_length (int): Number of samples between consecutive frames.
        axis (int, optional): Axis along which framing is applied. Defaults to the last axis (-1).
    """

    if x.shape[axis] < frame_length or hop_length < 1:
        raise ValueError(
            "Target axis length must be >= frame_length and hop_length >= 1"
        )

    axis = axis % x.ndim
    if axis != x.ndim - 1: # Move target axis to the last dimension for easier processing
        x = x.movedim(axis, -1)

    # Compute output shape and stride configuration
    size = x.shape[:-1] + (1 + (x.shape[-1] - frame_length) // hop_length, frame_length)
    stride = x.stride()[:-1] + (hop_length * x.stride()[-1], x.stride()[-1])

    xw = torch.as_strided(x, size=size, stride=stride)  # Create framed tensor without copying memory
    if axis != x.ndim - 1: # Restore original axis order if needed
        xw = xw.movedim(-2, axis)

    return xw

def rms(
    y,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "constant",
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
):
    """
    Compute the Root Mean Square (RMS) energy of an audio signal.

    Args:
        y (torch.Tensor or numpy.ndarray): Input audio signal.
        frame_length (int, optional): Length of each analysis frame. Defaults to 2048.
        hop_length (int, optional): Number of samples between frames. Defaults to 512.
        center (bool, optional): If True, pad the signal so frames are centered. Defaults to True.
        pad_mode (str, optional): Padding mode used when centering. Defaults to "constant".
        dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
        device (str, optional): Target device for computation. Defaults to "cpu".
    """

    # Convert input to tensor and move to target device/dtype
    y = (
        y.to(device=device, dtype=dtype) 
        if torch.is_tensor(y) else 
        torch.from_numpy(y.copy()).to(device=device, dtype=dtype)
    )

    if center: # Pad signal so frames are centered
        y = torch.nn.functional.pad(
            y, 
            (frame_length // 2, frame_length // 2), 
            mode=pad_mode
        )
    
    x = frame(y, frame_length=frame_length, hop_length=hop_length)
    # Compute mean square energy per frame
    power = x.square().mean(dim=-1, keepdim=True)
    # Convert power to RMS
    result = power.sqrt().mT 

    return result


class AudioProcessorTorch:
    """
    A class for processing audio signals, specifically for adjusting RMS levels.
    """

    def change_rms(
        source_audio: torch,
        source_rate: int,
        target_audio: torch,
        target_rate: int,
        rate: float,
        dtype = torch.float32,
        device = "cpu",
    ):
        """
        Adjust the RMS level of target_audio to match the RMS of source_audio, with a given blending rate.

        Args:
            source_audio: The source audio signal as a Torch Tensor.
            source_rate: The sampling rate of the source audio.
            target_audio: The target audio signal to adjust.
            target_rate: The sampling rate of the target audio.
            rate: The blending rate between the source and target RMS levels.
            dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
            device (str, optional): Target device for computation. Defaults to "cpu".
        """
        # Calculate RMS of both audio data
        rms1 = rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
            device=device,
            dtype=dtype,
        )
        rms2 = rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
            device=device,
            dtype=dtype,
        )

        # Interpolate RMS to match target audio length
        rms1 = F.interpolate(
            rms1.unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            rms2.unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        # Adjust target audio RMS based on the source audio RMS
        adjusted_audio = (
            target_audio
            * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1))
        )
        return adjusted_audio


class IndexWrapper:
    """
    A lightweight wrapper for loading, converting, and searching FAISS indexes
    using PyTorch tensors.

    Supports:
    - Reading a FAISS index
    - Converting vectors to tensors
    - Performing brute-force L2 distance search with PyTorch
    """

    def __init__(self, index_path: str, nprobe: int = 1, device: str = "cuda", dtype: torch.dtype = torch.float32, clamp: float = 1e-8):
        """
        Initialize the index wrapper.

        Args:
            index_path (str): Path to the FAISS index file.
            nprobe (int, optional): Number of IVF clusters to probe during search. Higher values improve accuracy but increase search time.
            device (str, optional): Target device for computation. Defaults to "cpu".
            dtype (torch.dtype, optional): Tensor data type. Defaults to torch.float32.
            clamp (float, optional): Minimum distance value used to avoid negative.
        """

        self.index_path = index_path
        self.nprobe = nprobe
        self.device = device
        self.dtype = dtype
        self.clamp = clamp
        self.index = None
        self.big_npy = None
        self.b_norms = None
        self.big_tensor = None
    
    def read_index(self):
        """
        Load a FAISS index and reconstruct all stored vectors.
        """

        if self.index_path != "" and os.path.exists(self.index_path):
            try:
                index = faiss.read_index(self.index_path)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(f"An error occurred reading the FAISS index: {error}")
                index = big_npy = None
        else:
            index = big_npy = None

        if index is not None: index.nprobe = self.nprobe
        self.index = index
        self.big_npy = big_npy

        return index, big_npy
    
    def read_index_tensor(self):
        """
        Load the FAISS index and convert reconstructed vectors into contiguous PyTorch tensors. Also precomputes squared norms for efficient L2 distance search.
        """

        self.read_index()

        if self.index is None or self.big_npy is None:
            self.big_tensor = None
            self.b_norms = None
        else:
            self.big_tensor = torch.from_numpy(self.big_npy).to(device=self.device, dtype=self.dtype).contiguous()
            # Precompute ||b||² for distance calculation
            self.b_norms = (
                (self.big_tensor ** 2).sum(dim=-1, keepdim=True).T.contiguous()
            )

        return self.big_tensor, self.b_norms
    
    def search(self, query, k=8):
        """
        Perform brute-force L2 nearest neighbor search using PyTorch.

        Args:
            query (torch.Tensor): Query tensor with shape [N, D].
            k (int, optional): Number of nearest neighbors to return.
        """

        with torch.inference_mode():
            # Compute ||a||^2 for query vectors
            q_norm = (query ** 2).sum(dim=-1, keepdim=True)
            # Compute squared L2 distances
            distances = torch.addmm(self.b_norms, query, self.big_tensor.T, alpha=-2.0, beta=1.0) + q_norm
            # Prevent negative values caused by floating-point precision
            distances = distances.clamp(min=self.clamp)

            # Retrieve smallest distances
            scores, indices = torch.topk(-distances, k=k, dim=-1)

            return -scores, indices