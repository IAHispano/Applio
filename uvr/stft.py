import torch

from uvr.device import should_fallback_to_cpu_for_complex_ops


class STFT:
    """
    This class performs the Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
    These functions are essential for converting the audio between the time domain and the frequency domain,
    which is a crucial aspect of audio processing in neural networks.
    """

    def __init__(self, logger, n_fft, hop_length, dim_f, device):
        self.logger = logger
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.device = device
        # Create a Hann window tensor for use in the STFT.
        self.hann_window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, input_tensor):
        original_device = input_tensor.device
        should_fallback = should_fallback_to_cpu_for_complex_ops(original_device)

        # Use CPU only when the current backend lacks the required complex operations.
        if should_fallback:
            input_tensor = input_tensor.cpu()

        # Transfer the pre-defined window tensor to the same device as the input tensor.
        stft_window = self.hann_window.to(input_tensor.device)

        # Extract batch dimensions (all dimensions except the last two which are channel and time).
        batch_dimensions = input_tensor.shape[:-2]

        # Extract channel and time dimensions (last two dimensions of the tensor).
        channel_dim, time_dim = input_tensor.shape[-2:]

        # Reshape the tensor to merge batch and channel dimensions for STFT processing.
        reshaped_tensor = input_tensor.reshape([-1, time_dim])

        # Perform the Short-Time Fourier Transform (STFT) on the reshaped tensor.
        stft_output = torch.stft(
            reshaped_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            center=True,
            return_complex=False,
        )

        # Rearrange the dimensions of the STFT output to bring the frequency dimension forward.
        permuted_stft_output = stft_output.permute([0, 3, 1, 2])

        # Reshape the output to restore the original batch and channel dimensions, while keeping the newly formed frequency and time dimensions.
        final_output = permuted_stft_output.reshape(
            [*batch_dimensions, channel_dim, 2, -1, permuted_stft_output.shape[-1]]
        ).reshape(
            [*batch_dimensions, channel_dim * 2, -1, permuted_stft_output.shape[-1]]
        )

        # Restore the input device after using the CPU fallback.
        if should_fallback:
            final_output = final_output.to(original_device)

        # Return the transformed tensor, sliced to retain only the required frequency dimension (`dim_f`).
        return final_output[..., : self.dim_f, :]

    def pad_frequency_dimension(
        self,
        input_tensor,
        batch_dimensions,
        channel_dim,
        freq_dim,
        time_dim,
        num_freq_bins,
    ):
        """
        Adds zero padding to the frequency dimension of the input tensor.
        """
        # Create a padding tensor for the frequency dimension
        freq_padding = torch.zeros(
            [*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim]
        ).to(input_tensor.device)

        # Concatenate the padding to the input tensor along the frequency dimension.
        padded_tensor = torch.cat([input_tensor, freq_padding], -2)

        return padded_tensor

    def calculate_inverse_dimensions(self, input_tensor):
        # Extract batch dimensions and frequency-time dimensions.
        batch_dimensions = input_tensor.shape[:-3]
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]

        # Calculate the number of frequency bins for the inverse STFT.
        num_freq_bins = self.n_fft // 2 + 1

        return batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins

    def prepare_for_istft(
        self, padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim
    ):
        """
        Prepares the tensor for Inverse Short-Time Fourier Transform (ISTFT) by reshaping
        and creating a complex tensor from the real and imaginary parts.
        """
        # Reshape the tensor to separate real and imaginary parts and prepare for ISTFT.
        reshaped_tensor = padded_tensor.reshape(
            [*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim]
        )

        # Flatten batch dimensions and rearrange for ISTFT.
        flattened_tensor = reshaped_tensor.reshape([-1, 2, num_freq_bins, time_dim])

        # Rearrange the dimensions of the tensor to bring the frequency dimension forward.
        permuted_tensor = flattened_tensor.permute([0, 2, 3, 1])

        # Combine real and imaginary parts into a complex tensor.
        complex_tensor = permuted_tensor[..., 0] + permuted_tensor[..., 1] * 1.0j

        return complex_tensor

    def inverse(self, input_tensor):
        original_device = input_tensor.device
        should_fallback = should_fallback_to_cpu_for_complex_ops(original_device)

        # Use CPU only when the current backend lacks the required complex operations.
        if should_fallback:
            input_tensor = input_tensor.cpu()

        # Transfer the pre-defined Hann window tensor to the same device as the input tensor.
        stft_window = self.hann_window.to(input_tensor.device)

        batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins = (
            self.calculate_inverse_dimensions(input_tensor)
        )

        padded_tensor = self.pad_frequency_dimension(
            input_tensor,
            batch_dimensions,
            channel_dim,
            freq_dim,
            time_dim,
            num_freq_bins,
        )

        complex_tensor = self.prepare_for_istft(
            padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim
        )

        # Perform the Inverse Short-Time Fourier Transform (ISTFT).
        istft_result = torch.istft(
            complex_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            center=True,
        )

        # Reshape ISTFT result to restore original batch and channel dimensions.
        final_output = istft_result.reshape([*batch_dimensions, 2, -1])

        # Restore the input device after using the CPU fallback.
        if should_fallback:
            final_output = final_output.to(original_device)

        return final_output
