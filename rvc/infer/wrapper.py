import os
import json
import torch
from typing import Optional
import numpy as np
import onnxruntime as ort


class ONNXRVC:
    """
    An inference wrapper for Retrieval-based Voice Conversion (RVC) models exported to ONNX format.

    This class supports runtime optimization via ONNX Runtime I/O Binding to achieve zero-copy memory
    transfers between PyTorch tensors and the underlying ONNX Runtime execution providers (e.g., CUDA).
    """

    def __init__(
        self,
        model_path,
        log_severity_level=3,
        device="cpu",
        is_zluda=False,
        capture=True,
        precision="float32",
    ):
        """
        Initializes the ONNXRVC instance, reads embed metadata, and setups execution routes.
        """

        self.device = device
        self.capture = capture
        self.precision = precision

        providers = self.get_provider(device)

        # Configure the Session options to limit log pollution
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = log_severity_level
        if providers[0][0].startswith("Tensorrt"):
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
        # Initialize the actual core ONNX computational runtime session
        self.net_g = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )

        # Retrieve and parse serialized architectural settings attached to the ONNX graph metadata block
        metadata_dict = json.loads(
            self.net_g.get_modelmeta().custom_metadata_map["model_info"]
        )
        self.use_f0 = metadata_dict.get("f0", 1)
        self.tgt_sr = metadata_dict.get("sr", 32000)
        self.version = metadata_dict.get("version", "v1")
        self.vocoder = metadata_dict.get("vocoder", "None")

        # Allocate reusable placeholder arrays for traditional NumPy CPU runtime fallback paths
        self.sid = np.empty(1, dtype=np.int64)
        self.rate = np.ones(1, dtype=np.float32)

        # Identify execution pipeline (Optimize utilizing direct pointers if backed by CUDA or CPU)
        self._device = (
            "cuda"
            if providers[0][0].startswith(("Tensorrt", "CUDA")) and not is_zluda
            else "cpu"
        )
        self.infer = (
            self.infer_io
            if providers[0][0].startswith(("Tensorrt", "CUDA", "CPU")) and not is_zluda
            else self.infer_non_io
        )

    def get_provider(self, device):
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()

        self.capture_enabled = (
            self.capture and os.environ.get("APPLIO_DISABLE_CUDA_GRAPHS") != "1"
        )

        if str(device) == "mps" and "CoreMLExecutionProvider" in available:
            providers.insert(
                0,
                (
                    "CoreMLExecutionProvider",
                    {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
                ),
            )
        elif (
            str(device).startswith("cuda")
            and "CUDAExecutionProvider" in available
            and not getattr(torch.version, "hip", None)
        ):
            selected = torch.device(device)
            self.cuda_device = torch.device(
                "cuda",
                (
                    selected.index
                    if selected.index is not None
                    else torch.cuda.current_device()
                ),
            )
            # Own a non-default stream: pointer 0 can be interpreted as an
            # unspecified stream by providers. Keep all staging, replay and
            # output copies on the same explicit stream.
            self.cuda_stream = torch.cuda.Stream(device=self.cuda_device)
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        # ORT 1.22 also excludes FP16 tensor-core plans when this
                        # option is false. FP16 graphs have no FP32 convolution
                        # inputs to round to TF32; enable their native plans.
                        "use_tf32": "1" if self.precision == "float16" else "0",
                        "user_compute_stream": str(self.cuda_stream.cuda_stream),
                        "enable_cuda_graph": "1" if self.capture_enabled else "0",
                        "cudnn_conv1d_pad_to_nc1d": "1",
                        "device_id": (str(self.cuda_device.index)),
                    },
                )
            ]

        return providers

    def get_argument(self, feats, p_len, sid, pitch=None, pitchf=None, rate=None):
        """
        Constructs standard raw NumPy dictionary inputs including pitch values.

        Args:
            feats (torch.Tensor): Audio features/phone tensor.
            p_len (torch.Tensor): Length array of the features.
            sid (torch.Tensor): Speaker identity token tensor.
            pitch (torch.Tensor, optional): Integer pitch (F0) contour values.
            pitchf (torch.Tensor, optional): Floating-point normalized pitch values.
            rate (torch.Tensor, optional): Speed/stretching factor.
        """

        self.sid[0] = sid.item()
        self.rate[0] = (
            1.0 if rate is None else rate.item() if hasattr(rate, "item") else rate
        )

        args = {
            "phone": feats.cpu().numpy().astype(np.float32),
            "phone_lengths": p_len.cpu().numpy(),
            "sid": self.sid,
            "rate": self.rate,
        }

        if self.use_f0:
            args.update(
                {
                    "pitch": pitch.cpu().numpy().astype(np.int64),
                    "nsff0": pitchf.cpu().numpy().astype(np.float32),
                }
            )

        return args

    def infer_non_io(
        self,
        feats: torch.Tensor = None,
        p_len: torch.Tensor = None,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
        skip_head: Optional[int] = None,
        decoder_fp16: bool = False,
    ):
        """
        Runs classic inference by mapping device matrices back into CPU NumPy arrays.
        High data copying overhead, acts as fallback.

        Returns:
            torch.Tensor: The resulting synthesized raw audio waveform tensor.
        """

        # Execute model using classic key-value Python dictionaries
        output = self.net_g.run(
            ["audio"],
            (
                self.get_argument(
                    feats,
                    p_len,
                    sid,
                    pitch,
                    pitchf,
                    (1.0 - skip_head / feats.shape[1]) if rate is None else rate,
                )
            ),
        )

        # Wrap array response back into PyTorch and push to the assigned destination device
        return torch.from_numpy(output[0]).to(self.device)

    def infer_io(
        self,
        feats: torch.Tensor = None,
        p_len: torch.Tensor = None,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
        skip_head: Optional[int] = None,
        decoder_fp16: bool = False,
    ):
        """
        Runs highly efficient execution passing physical GPU/CPU memory addresses via I/O bindings.
        This cuts out all conversion lags from crossing frameworks.

        Returns:
            torch.Tensor: The resulting synthesized raw audio waveform tensor.
        """

        # Construct and format parameters directly inside native tensor layers
        rate = (
            torch.FloatTensor([1.0]).to(self.device)
            if skip_head is None
            else (
                torch.FloatTensor([1.0 - skip_head / feats.shape[1]]).to(self.device)
                if rate is None
                else rate.float()
            )
        )
        feats, p_len = feats.float().contiguous(), p_len.contiguous()
        device_idx = feats.device.index or 0

        # Construct individual runtime binding instance
        io_binding = self.net_g.io_binding()

        io_binding.bind_input(
            name="phone",
            device_type=self._device,
            device_id=device_idx,
            element_type=np.float32,
            shape=tuple(feats.shape),
            buffer_ptr=feats.data_ptr(),
        )
        io_binding.bind_input(
            name="phone_lengths",
            device_type=self._device,
            device_id=device_idx,
            element_type=np.int64,
            shape=tuple(p_len.shape),
            buffer_ptr=p_len.data_ptr(),
        )

        io_binding.bind_input(
            name="sid",
            device_type=self._device,
            device_id=device_idx,
            element_type=np.int64,
            shape=tuple(sid.shape),
            buffer_ptr=sid.data_ptr(),
        )
        io_binding.bind_input(
            name="rate",
            device_type=self._device,
            device_id=device_idx,
            element_type=np.float32,
            shape=tuple(rate.shape),
            buffer_ptr=rate.data_ptr(),
        )

        if self.use_f0:  # Include F0 frequency layers conditionally if flagged
            pitch, pitchf = pitch.contiguous(), pitchf.float().contiguous()

            io_binding.bind_input(
                name="pitch",
                device_type=self._device,
                device_id=device_idx,
                element_type=np.int64,
                shape=tuple(pitch.shape),
                buffer_ptr=pitch.data_ptr(),
            )

            io_binding.bind_input(
                name="nsff0",
                device_type=self._device,
                device_id=device_idx,
                element_type=np.float32,
                shape=tuple(pitchf.shape),
                buffer_ptr=pitchf.data_ptr(),
            )

        # Bind output receiver allocation pointing directly to target execution space block
        io_binding.bind_output(
            name="audio", device_type=self._device, device_id=device_idx
        )
        # Direct zero-copy inference execution step
        self.net_g.run_with_iobinding(io_binding)

        return [
            torch.from_numpy(io_binding.get_outputs()[0].numpy()).to(device=self.device)
        ]
