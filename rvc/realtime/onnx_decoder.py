"""Optional HiFi-GAN NSF decoder export/runtime; encoder, flow and source stay in Torch."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DecoderGraph(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent, source, speaker):
        dec = self.decoder
        value = dec.conv_pre(latent) + dec.cond(speaker)
        for i, (upsample, noise) in enumerate(zip(dec.ups, dec.noise_convs)):
            value = upsample(F.leaky_relu(value, dec.lrelu_slope)) + noise(source)
            value = (
                sum(
                    block(value)
                    for block in dec.resblocks[
                        i * dec.num_kernels : (i + 1) * dec.num_kernels
                    ]
                )
                / dec.num_kernels
            )
        return torch.tanh(dec.conv_post(F.leaky_relu(value)))


def export_decoder(decoder, checkpoint, output, precision="float32"):
    from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator

    if not isinstance(decoder, HiFiGANNSFGenerator):
        raise ValueError("ONNX decoder export currently supports f0 HiFi-GAN NSF only")
    if precision not in ("float32", "float16"):
        raise ValueError("Decoder precision must be float32 or float16")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = DecoderGraph(decoder.cpu().float().eval())
    frames = 16
    args = (
        torch.randn(1, decoder.conv_pre.in_channels, frames),
        torch.randn(1, 1, frames * decoder.upp),
        torch.randn(1, decoder.cond.in_channels, 1),
    )
    torch.onnx.export(
        graph,
        args,
        str(output),
        dynamo=False,
        opset_version=17,
        input_names=["latent", "source", "speaker"],
        output_names=["audio"],
        dynamic_axes={
            "latent": {2: "frames"},
            "source": {2: "samples"},
            "audio": {2: "samples"},
        },
    )
    import onnx

    if precision not in ("float32", "float16"):
        raise ValueError("Decoder precision must be float32 or float16")
    if precision == "float16":
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model = convert_float_to_float16(onnx.load(str(output)), keep_io_types=False)
        onnx.save(model, str(output))

    onnx.checker.check_model(str(output))
    manifest = {
        "schema": 1,
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "onnx_sha256": checkpoint_hash(output),
        "upsample_factor": decoder.upp,
        "latent_channels": decoder.conv_pre.in_channels,
        "speaker_channels": decoder.cond.in_channels,
        "precision": precision,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2)
    )
    return graph, manifest


class OnnxDecoder(torch.nn.Module):
    def __init__(self, original, checkpoint, path, device):
        super().__init__()
        import onnxruntime as ort

        path = Path(path)
        manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        if (
            manifest.get("schema") != 1
            or manifest.get("checkpoint_sha256") != checkpoint_hash(checkpoint)
            or manifest.get("onnx_sha256") != checkpoint_hash(path)
        ):
            raise ValueError(
                "ONNX decoder cache does not match its checkpoint or manifest"
            )
        from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator

        if not isinstance(original, HiFiGANNSFGenerator):
            raise ValueError("ONNX decoder requires an f0 HiFi-GAN NSF model")
        if (
            manifest.get("upsample_factor") != original.upp
            or manifest.get("latent_channels") != original.conv_pre.in_channels
            or manifest.get("speaker_channels") != original.cond.in_channels
        ):
            raise ValueError("ONNX decoder dimensions do not match the voice model")
        providers = ["CPUExecutionProvider"]
        self.precision = manifest.get("precision", "float32")
        if self.precision not in ("float32", "float16"):
            raise ValueError("Unsupported ONNX decoder precision")
        self.io_dtype = torch.float16 if self.precision == "float16" else torch.float32
        self.numpy_dtype = np.float16 if self.precision == "float16" else np.float32
        self.cuda_device = None
        self.cuda_stream = None
        self._cuda_buffers = {}
        available = ort.get_available_providers()
        if device == "mps" and "CoreMLExecutionProvider" in available:
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
            self.cuda_stream = torch.cuda.current_stream(self.cuda_device)
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {
                        # ORT 1.22 also excludes FP16 tensor-core plans when this
                        # option is false. FP16 graphs have no FP32 convolution
                        # inputs to round to TF32; enable their native plans.
                        "use_tf32": "1" if self.precision == "float16" else "0",
                        "user_compute_stream": str(self.cuda_stream.cuda_stream),
                        "enable_cuda_graph": "1",
                        "cudnn_conv1d_pad_to_nc1d": "1",
                        "device_id": (str(self.cuda_device.index)),
                    },
                ),
            )
        options = ort.SessionOptions()
        if self.precision == "float16" and self.cuda_device is None:
            raise ValueError("The float16 ONNX decoder requires the CUDA provider")
        options.intra_op_num_threads = 2
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        if (
            self.cuda_device is not None
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError("The requested ONNX CUDA provider failed to load")
        expected_type = (
            "tensor(float16)" if self.precision == "float16" else "tensor(float)"
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if (
            [item.name for item in inputs] != ["latent", "source", "speaker"]
            or [item.name for item in outputs] != ["audio"]
            or any(item.type != expected_type for item in inputs + outputs)
        ):
            raise ValueError("ONNX decoder graph has incompatible inputs or precision")
        self.source = original.m_source
        self.upp = original.upp
        print(
            f"Experimental ONNX decoder: {self.session.get_providers()} "
            f"({'CUDA I/O binding' if self.cuda_device is not None else 'host input/output transfers'})."
        )

    def forward(self, latent, f0, g=None):
        source = self.source(f0, self.upp)[0].transpose(1, 2)
        return self.run_decoder(latent, source, g)

    def run_decoder(self, latent, source, g):
        """Run only the ONNX graph, with excitation prepared by Torch."""
        if self.cuda_device is not None:
            if latent.device != self.cuda_device:
                raise ValueError(
                    "ONNX decoder inputs must stay on the selected CUDA device"
                )
            caller = torch.cuda.current_stream(self.cuda_device)
            self.cuda_stream.wait_stream(caller)
            with torch.cuda.stream(self.cuda_stream):
                tensors = [
                    value.detach().to(self.io_dtype).contiguous()
                    for value in (latent, source, g)
                ]
                key = tuple(tuple(tensor.shape) for tensor in tensors)
                cached = self._cuda_buffers.get(key)
                if cached is None:
                    # Keep at most four captured shapes. Uncommon additional
                    # sizes run without capture rather than retaining graphs
                    # and allocator memory for every live chunk-size change.
                    graph_id = (
                        len(self._cuda_buffers) if len(self._cuda_buffers) < 4 else -1
                    )
                    static = [torch.empty_like(tensor) for tensor in tensors]
                    output = torch.empty(
                        (latent.shape[0], 1, source.shape[-1]),
                        device=self.cuda_device,
                        dtype=self.io_dtype,
                    )
                    binding = self.session.io_binding()
                    for name, tensor in zip(("latent", "source", "speaker"), static):
                        binding.bind_input(
                            name,
                            "cuda",
                            self.cuda_device.index,
                            self.numpy_dtype,
                            tuple(tensor.shape),
                            tensor.data_ptr(),
                        )
                    binding.bind_output(
                        "audio",
                        "cuda",
                        self.cuda_device.index,
                        self.numpy_dtype,
                        tuple(output.shape),
                        output.data_ptr(),
                    )
                    import onnxruntime as ort

                    run_options = ort.RunOptions()
                    run_options.add_run_config_entry("gpu_graph_id", str(graph_id))
                    cached = static, output, binding, run_options
                    if graph_id >= 0:
                        self._cuda_buffers[key] = cached
                static, output, binding, run_options = cached
                for destination, tensor in zip(static, tensors):
                    destination.copy_(tensor)
                self.session.run_with_iobinding(binding, run_options)
                for tensor in tensors:
                    tensor.record_stream(self.cuda_stream)
            caller.wait_stream(self.cuda_stream)
            output.record_stream(caller)
            if not torch.isfinite(output).all().item():
                raise RuntimeError("ONNX decoder produced non-finite audio")
            # Independent storage prevents a later replay from changing audio
            # retained by a caller (or downstream in-place clipping).
            return output.to(latent.dtype).clone()
        inputs = {
            name: tensor.detach().float().cpu().contiguous().numpy()
            for name, tensor in (("latent", latent), ("source", source), ("speaker", g))
        }
        output = self.session.run(["audio"], inputs)[0]
        if not np.isfinite(output).all():
            raise RuntimeError("ONNX decoder produced non-finite audio")
        return torch.from_numpy(output).to(device=latent.device, dtype=latent.dtype)
