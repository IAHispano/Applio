import torch
import torch.nn.functional as F


def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    for d_real, d_fake in zip(fmap_real, fmap_fake):
        for o_real, o_fake in zip(d_real, d_fake):
            loss += torch.mean(torch.abs(o_real - o_fake))
    return loss


def discriminator_loss_san(disc_real, disc_fake, scale=1.0):
    loss = 0
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        d_real_fun, d_real_dir = d_real
        d_fake_fun, d_fake_dir = d_fake
        real_fun_loss = torch.mean(F.softplus(1 - d_real_fun) ** 2)
        real_dir_loss = torch.mean(F.softplus(1 - d_real_dir) ** 2)
        fake_fun_loss = torch.mean(F.softplus(d_fake_fun) ** 2)
        fake_dir_loss = torch.mean(-(F.softplus(1 - d_fake_dir) ** 2))
        if i < len(disc_real):
            loss += real_fun_loss + real_dir_loss + fake_fun_loss + fake_dir_loss
        else:
            loss += scale * (
                real_fun_loss + real_dir_loss + fake_fun_loss + fake_dir_loss
            )
    return loss


def generator_loss_san(disc_outputs):
    loss = 0
    for d_fake in disc_outputs:
        loss += torch.mean(F.softplus(1 - d_fake.float()) ** 2)
    return loss
