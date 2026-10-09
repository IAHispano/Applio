def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
