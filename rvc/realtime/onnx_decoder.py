"""Optional HiFi-GAN NSF decoder export/runtime; encoder, flow and source stay in Torch."""

import copy
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

EXPORT_VERSION = 1
_export_lock = threading.RLock()


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


def export_decoder(
    decoder, checkpoint, output, precision="float32", expected_checkpoint_hash=None
):
    """Publish a checked export without moving or changing the loaded decoder."""
    # The legacy exporter has global state; serialize exports within a process.
    with _export_lock:
        return _export_decoder(
            decoder, checkpoint, output, precision, expected_checkpoint_hash
        )


def _export_decoder(decoder, checkpoint, output, precision, expected_checkpoint_hash):
    from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator

    if not isinstance(decoder, HiFiGANNSFGenerator):
        raise ValueError("ONNX decoder export currently supports f0 HiFi-GAN NSF only")
    if precision not in ("float32", "float16"):
        raise ValueError("Decoder precision must be float32 or float16")
    output = Path(output)
    if (
        output.suffix.lower() != ".onnx"
        or output.resolve() == Path(checkpoint).resolve()
    ):
        raise ValueError("ONNX export requires a separate .onnx destination")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = checkpoint_hash(checkpoint)
    if expected_checkpoint_hash is not None and before != expected_checkpoint_hash:
        raise ValueError("Voice checkpoint changed before ONNX export; reload")
    decoder = copy.deepcopy(decoder).cpu().float().eval()
    graph = DecoderGraph(decoder)
    frames = 16
    args = (
        torch.zeros(1, decoder.conv_pre.in_channels, frames),
        torch.zeros(1, 1, frames * decoder.upp),
        torch.zeros(1, decoder.cond.in_channels, 1),
    )
    # Work in the destination filesystem so publication uses atomic renames.
    # Failed conversion/checking leaves the previous graph and manifest intact.
    with tempfile.TemporaryDirectory(
        prefix=".onnx-export-", dir=output.parent
    ) as folder:
        staged = Path(folder) / "decoder.onnx"
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.onnx.export(
                graph,
                args,
                str(staged),
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

        if precision == "float16":
            from onnxruntime.transformers.float16 import convert_float_to_float16

            model = convert_float_to_float16(
                onnx.load(str(staged)), keep_io_types=False
            )
            onnx.save(model, str(staged))
        onnx.checker.check_model(str(staged))
        if checkpoint_hash(checkpoint) != before:
            raise ValueError("Voice checkpoint changed during ONNX export; retry")
        manifest = {
            "schema": 1,
            "export_version": EXPORT_VERSION,
            "checkpoint_sha256": before,
            "onnx_sha256": checkpoint_hash(staged),
            "upsample_factor": decoder.upp,
            "latent_channels": decoder.conv_pre.in_channels,
            "speaker_channels": decoder.cond.in_channels,
            "precision": precision,
        }
        staged_manifest = staged.with_suffix(".onnx.json")
        staged_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        backup = Path(folder) / "previous.onnx"
        if output.exists():
            shutil.copyfile(output, backup)
        os.replace(staged, output)
        try:
            os.replace(staged_manifest, output.with_suffix(output.suffix + ".json"))
        except OSError:
            # Roll back if the second rename fails (e.g. a locked manifest).
            if backup.exists():
                os.replace(backup, output)
            else:
                output.unlink()
            raise
    return graph, manifest


@contextmanager
def _cache_lock(path):
    """Coordinate cache repair/export across workers without another dependency."""
    with open(path, "a+b") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"\0")
            file.flush()
        deadline = time.monotonic() + 120
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Another worker is still exporting this ONNX decoder"
                    )
                time.sleep(0.1)
        try:
            yield
        finally:
            if os.name == "nt":
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def resolve_decoder_path(
    original,
    checkpoint,
    requested,
    device,
    expected_checkpoint_hash=None,
    precision=None,
):
    """Resolve a supplied graph or export/reuse a model-specific automatic cache."""
    digest = checkpoint_hash(checkpoint)
    if expected_checkpoint_hash is not None and digest != expected_checkpoint_hash:
        raise ValueError(
            "Voice checkpoint changed after loading; reload before ONNX export"
        )
    if str(requested).lower() != "auto":
        return Path(requested)
    from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator

    if not isinstance(original, HiFiGANNSFGenerator):
        raise ValueError("ONNX decoder requires an f0 HiFi-GAN NSF model")
    import onnxruntime as ort

    cuda = (
        torch.device(device).type == "cuda"
        and not getattr(torch.version, "hip", None)
        and "CUDAExecutionProvider" in ort.get_available_providers()
    )
    precision = precision or os.environ.get(
        "APPLIO_ONNX_PRECISION", "float16" if cuda else "float32"
    )
    if precision not in ("float32", "float16"):
        raise ValueError("APPLIO_ONNX_PRECISION must be float32 or float16")
    if precision == "float16" and not cuda:
        raise ValueError("The float16 ONNX decoder requires the CUDA provider")
    directory = Path(
        os.environ.get("APPLIO_ONNX_CACHE_DIR")
        or Path(checkpoint).resolve().parent / ".onnx-cache"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.v{EXPORT_VERSION}.{precision}.onnx"
    with _cache_lock(path.with_suffix(".lock")):
        try:
            manifest = json.loads(
                path.with_suffix(".onnx.json").read_text(encoding="utf-8")
            )
            valid = (
                isinstance(manifest, dict)
                and manifest.get("schema") == 1
                and manifest.get("export_version") == EXPORT_VERSION
                and manifest.get("checkpoint_sha256") == digest
                and manifest.get("onnx_sha256") == checkpoint_hash(path)
                and manifest.get("precision") == precision
                and manifest.get("upsample_factor") == original.upp
                and manifest.get("latent_channels") == original.conv_pre.in_channels
                and manifest.get("speaker_channels") == original.cond.in_channels
            )
        except (OSError, ValueError):
            valid = False
        if not valid:
            print(f"Exporting experimental {precision} ONNX decoder cache: {path}")
            try:
                export_decoder(
                    original,
                    checkpoint,
                    path,
                    precision,
                    expected_checkpoint_hash=digest,
                )
            except ImportError as error:
                raise RuntimeError(
                    "ONNX auto export requires the optional 'onnx' package"
                ) from error
    return path


class OnnxDecoder(torch.nn.Module):
    def __init__(
        self,
        original,
        checkpoint,
        path,
        device,
        expected_checkpoint_hash=None,
        capture=True,
    ):
        super().__init__()
        import onnxruntime as ort

        path = Path(path)
        manifest = json.loads(
            path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != 1
            or manifest.get("checkpoint_sha256") != checkpoint_hash(checkpoint)
            or (
                expected_checkpoint_hash is not None
                and manifest.get("checkpoint_sha256") != expected_checkpoint_hash
            )
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
        self.capture_enabled = (
            capture and os.environ.get("APPLIO_DISABLE_CUDA_GRAPHS") != "1"
        )
        self._cuda_buffers = {}
        self._run_lock = threading.RLock()
        available = ort.get_available_providers()
        if (
            torch.device(device).type == "cuda"
            and not getattr(torch.version, "hip", None)
            and "CUDAExecutionProvider" not in available
        ):
            raise RuntimeError(
                "ONNX CUDA decoding requires a compatible onnxruntime-gpu installation"
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
        options = ort.SessionOptions()
        if self.precision == "float16" and self.cuda_device is None:
            raise ValueError("The float16 ONNX decoder requires the CUDA provider")
        options.intra_op_num_threads = 2
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if self.cuda_device is not None:
            # CUDA graph capture requires every graph node to run on CUDA.
            # Refuse partial CPU placement and provider auto-fallback.
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        if (
            self.cuda_device is not None
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError("The requested ONNX CUDA provider failed to load")
        self.session.disable_fallback()
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
        self.path = path.resolve()
        self.checkpoint_sha256 = manifest["checkpoint_sha256"]
        self.latent_channels = original.conv_pre.in_channels
        self.speaker_channels = original.cond.in_channels
        expected_shapes = (
            [1, self.latent_channels, None],
            [1, 1, None],
            [1, self.speaker_channels, 1],
            [1, 1, None],
        )
        for item, expected in zip(inputs + outputs, expected_shapes):
            if len(item.shape) != 3 or any(
                (
                    actual != dimension
                    if dimension is not None
                    else isinstance(actual, int)
                )
                for actual, dimension in zip(item.shape, expected)
            ):
                raise ValueError(
                    "ONNX decoder graph has incompatible tensor dimensions"
                )
        print(
            f"Experimental ONNX decoder: {self.session.get_providers()} "
            f"({'CUDA I/O binding' if self.cuda_device is not None else 'host input/output transfers'})."
        )

    def forward(self, latent, f0, g=None):
        source = self.source(f0, self.upp)[0].transpose(1, 2)
        return self.run_decoder(latent, source, g)

    def run_decoder(self, latent, source, g):
        """Run only the ONNX graph, with excitation prepared by Torch."""
        with self._run_lock:
            self._validate_inputs(latent, source, g)
            return self._run_decoder(latent, source, g)

    def _validate_inputs(self, latent, source, g):
        for value in (latent, source, g):
            if not isinstance(value, torch.Tensor) or value.dtype not in (
                torch.float16,
                torch.float32,
            ):
                raise ValueError(
                    "ONNX decoder requires floating latent, source and speaker tensors"
                )
            if value.device != latent.device or (
                self.cuda_device is not None and value.device != self.cuda_device
            ):
                raise ValueError("ONNX decoder inputs must stay on the selected device")
        if (
            latent.ndim != 3
            or tuple(latent.shape[:2]) != (1, self.latent_channels)
            or latent.shape[-1] < 1
            or tuple(source.shape) != (1, 1, latent.shape[-1] * self.upp)
            or tuple(g.shape) != (1, self.speaker_channels, 1)
        ):
            raise ValueError("ONNX decoder input shapes do not match the voice model")

    def _run_decoder(self, latent, source, g):
        if self.cuda_device is not None:
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
                    cache_slot = (
                        len(self._cuda_buffers)
                        if self.capture_enabled and len(self._cuda_buffers) < 4
                        else -1
                    )
                    graph_id = cache_slot if self.capture_enabled else -1
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
                    if cache_slot >= 0:
                        self._cuda_buffers[key] = cached
                static, output, binding, run_options = cached
                for destination, tensor in zip(static, tensors):
                    destination.copy_(tensor)
                self.session.run_with_iobinding(binding, run_options)
                for tensor in tensors:
                    tensor.record_stream(self.cuda_stream)
                # Finish the owned copy before another caller can replay into
                # the shared output. A dtype conversion already owns storage.
                result = (
                    output.to(latent.dtype)
                    if output.dtype != latent.dtype
                    else output.clone()
                )
                if not torch.isfinite(result).all().item():
                    raise RuntimeError("ONNX decoder produced non-finite audio")
            caller.wait_stream(self.cuda_stream)
            result.record_stream(caller)
            return result
        inputs = {
            name: tensor.detach().float().cpu().contiguous().numpy()
            for name, tensor in (("latent", latent), ("source", source), ("speaker", g))
        }
        output = self.session.run(["audio"], inputs)[0]
        if not np.isfinite(output).all():
            raise RuntimeError("ONNX decoder produced non-finite audio")
        return torch.from_numpy(output).to(device=latent.device, dtype=latent.dtype)
