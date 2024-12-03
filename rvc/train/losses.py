import torch

def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    loss = sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )
    return loss * 2


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

        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
        loss += r_loss + g_loss

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    Compute the generator loss based on discriminator outputs.

    Args:
        disc_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    gen_losses = []
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        gen_losses.append(l.item())
        loss += l

    return loss, gen_losses


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
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)

    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)

    return loss
