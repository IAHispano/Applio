import torch


def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list[list[torch.Tensor]]): Reference maps grouped by discriminator,
            then layer. Preserve each tensor's batch dimension.
        fmap_g (list[list[torch.Tensor]]): Generated maps with matching groups,
            layer counts and tensor shapes.

    RVC sums layer means and multiplies by two. Flattening the outer groups
    would make the inner loop iterate over batch items instead of layers and
    silently change the loss scale. Validate nesting before doing arithmetic;
    score tensors may be flattened, but feature-map groups must stay nested.
    """
    if not isinstance(fmap_r, (list, tuple)) or not isinstance(fmap_g, (list, tuple)):
        raise TypeError("Feature maps must be grouped by discriminator and layer")
    losses = []
    for dr, dg in zip(fmap_r, fmap_g, strict=True):
        if not isinstance(dr, (list, tuple)) or not isinstance(dg, (list, tuple)):
            raise TypeError("Keep feature maps nested: [discriminator][layer], not a flat tensor list")
        for rl, gl in zip(dr, dg, strict=True):
            if not torch.is_tensor(rl) or not torch.is_tensor(gl):
                raise TypeError("Each feature map must be a tensor")
            if rl.shape != gl.shape:
                raise ValueError("Real and generated feature-map shapes must match")
            losses.append(torch.mean(torch.abs(rl - gl)))
    return 2 * sum(losses)


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)

        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())
        loss += r_loss + g_loss

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Compute the generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        loss += l

    return loss, gen_losses


def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    """
    Compute the scaled discriminator loss for real and generated outputs.

    Args:
        disc_real (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_fake (list of torch.Tensor): List of discriminator outputs for generated samples.
        scale (float, optional): Scaling factor applied to losses beyond the midpoint. Default is 1.0.
    """
    midpoint = len(disc_real) // 2
    losses = []
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = (1 - d_real).pow(2).mean()
        fake_loss = d_fake.pow(2).mean()
        total_loss = real_loss + fake_loss
        if i >= midpoint:
            total_loss *= scale
        losses.append(total_loss)
    loss = sum(losses)
    return loss, None, None


def generator_loss_scaled(disc_outputs, scale=1.0):
    """
    Compute the scaled generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
        scale (float, optional): Scaling factor applied to losses beyond the midpoint. Default is 1.0.
    """
    midpoint = len(disc_outputs) // 2
    losses = []
    for i, d_fake in enumerate(disc_outputs):
        loss_value = (1 - d_fake).pow(2).mean()
        if i >= midpoint:
            loss_value *= scale
        losses.append(loss_value)
    loss = sum(losses)
    return loss, None, None


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Latent variable z_p [b, h, t_t].
        logs_q (torch.Tensor): Log variance of q [b, h, t_t].
        m_p (torch.Tensor): Mean of p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    return loss
