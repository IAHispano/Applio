import sys, os

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
sys.path.append(os.path.join(now_dir, "train"))
import utils
import datetime

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import shuffle, randint
import traceback, json, argparse, itertools, math, torch, pdb

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from lib.infer_pack import commons
from time import sleep
from time import time as ttime
from data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)

import csv

if hps.version == "v1":
    from lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid as RVC_Model_f0,
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminator,
    )
else:
    from lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from process_ckpt import savee

global global_step
global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(
                i,
                n_gpus,
                hps,
            ),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()

def reset_stop_flag():
    with open("csvdb/stop.csv", "w+", newline="") as STOPCSVwrite:
        csv_writer = csv.writer(STOPCSVwrite, delimiter=",")
        csv_writer.writerow(["False"])

def create_model(hps, model_f0, model_nof0):
    filter_length_adjusted = hps.data.filter_length // 2 + 1
    segment_size_adjusted = hps.train.segment_size // hps.data.hop_length
    is_half = hps.train.fp16_run
    sr = hps.sample_rate

    model = model_f0 if hps.if_f0 == 1 else model_nof0

    return model(
        filter_length_adjusted,
        segment_size_adjusted,
        **hps.model,
        is_half=is_half,
        sr=sr
    )

def move_model_to_cuda_if_available(model, rank):
    if torch.cuda.is_available():
        return model.cuda(rank)
    else:
        return model

def create_optimizer(model, hps):
    return torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

def create_ddp_model(model, rank):
    if torch.cuda.is_available():
        return DDP(model, device_ids=[rank])
    else:
        return DDP(model)

def create_dataset(hps, if_f0=True):
    return TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data) if if_f0 else TextAudioLoader(hps.data.training_files, hps.data)

def create_sampler(dataset, batch_size, n_gpus, rank):
    return DistributedBucketSampler(
            dataset,
            batch_size * n_gpus,
            # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
            [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )

def set_collate_fn(if_f0=True):
    return TextAudioCollateMultiNSFsid() if if_f0 else TextAudioCollate()

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    
    train_dataset = TextAudioLoaderMultiNSFsid(
        hps.data.training_files, hps.data
    ) if hps.if_f0 == 1 else TextAudioLoader(hps.data.training_files, hps.data)

    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    
    collate_fn = TextAudioCollateMultiNSFsid() if hps.if_f0 == 1 else TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    net_g = create_model(hps, RVC_Model_f0, RVC_Model_nof0)

    net_g = move_model_to_cuda_if_available(net_g, rank)
    net_d = move_model_to_cuda_if_available(MultiPeriodDiscriminator(hps.model.use_spectral_norm), rank)

    optim_g = create_optimizer(net_g, hps)
    optim_d = create_optimizer(net_d, hps)
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    net_g = create_ddp_model(net_g, rank)
    net_d = create_ddp_model(net_d, rank)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(f"loaded pretrained {hps.pretrainG}")
            print(
                net_g.module.load_state_dict(
                    torch.load(hps.pretrainG, map_location="cpu")["model"]
                )
            )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            print(
                net_d.module.load_state_dict(
                    torch.load(hps.pretrainD, map_location="cpu")["model"]
                )
            )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = (writers if writers is not None else (None, None))

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    nets = [net_g, net_d]
    for net in nets:
        net.train()

    def save_checkpoint(name):
        ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
        result = savee(ckpt, hps.sample_rate, hps.if_f0, name, epoch, hps.version, hps)
        logger.info("Saving final ckpt: {}".format(result))
        sleep(1)

    if hps.if_cache_data_in_gpu:
        # Use Cache
        data_iterator = cache
        if len(cache) == 0:
            gpu_available = torch.cuda.is_available()

            for batch_idx, info in enumerate(train_loader):
                # Unpack
                info = list(info)
                if hps.if_f0:
                    tensors = info
                else:
                    # We consider that pitch and pitchf are not included in this case
                    tensors = info[:2] + info[4:]

                # Load on CUDA
                if gpu_available:
                    tensors = [tensor.cuda(rank, non_blocking=True) for tensor in tensors]

                # Cache on list
                cache.extend([(batch_idx, tuple(tensor for tensor in tensors if tensor is not None))])
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    def to_gpu_if_available(tensor):
        return tensor.cuda(rank, non_blocking=True) if torch.cuda.is_available() else tensor

    # Run steps
    gpu_available = torch.cuda.is_available()
    epoch_recorder = EpochRecorder()
    fp16_run = hps.train.fp16_run
    c_mel = hps.train.c_mel

    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (not hps.if_cache_data_in_gpu) and gpu_available:
            phone = to_gpu_if_available(phone)
            phone_lengths = to_gpu_if_available(phone_lengths)
            sid = to_gpu_if_available(sid)
            spec = to_gpu_if_available(spec)
            spec_lengths = to_gpu_if_available(spec_lengths)
            wave = to_gpu_if_available(wave)

            if hps.if_f0 == 1:
                pitch = to_gpu_if_available(pitch)
                pitchf = to_gpu_if_available(pitchf)

        # Calculate
        with autocast(enabled=fp16_run):
            if hps.if_f0 == 1:
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
                    net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
                    net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            if fp16_run: y_hat_mel = y_hat_mel.half()

            wave = commons.slice_segments(wave, ids_slice * hps.data.hop_length,
                                        hps.train.segment_size)  # slice

            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())

            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            net_d_params = net_d.parameters()
            net_g_params = net_g.parameters()
            lr_scalar = optim_g.param_groups[0]["lr"]
            
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d_params, None)
            scaler.step(optim_d)

            with autocast(enabled=fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

                loss_mel = F.l1_loss(y_mel, y_hat_mel) * c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g_params, None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0 and global_step % hps.train.log_interval == 0:
                lr = lr_scalar  # use stored lr scalar here
                logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))

                # Amor For Tensorboard display
                loss_mel, loss_kl = min(loss_mel, 75), min(loss_kl, 9)

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                    **{"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)},
                    **{"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)},
                    **{"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)},
                }

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                }
                    
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
            global_step += 1

    if epoch % hps.save_every_epoch == 0:
        if rank == 0:
            save_format = str(2333333) if hps.if_latest else str(global_step)
            model_dir = hps.model_dir
            learning_rate = hps.train.learning_rate
            name_epoch = f"{hps.name}_e{epoch}"
            models = {'G': net_g, 'D': net_d}
            optims = {'G': optim_g, 'D': optim_d}
            
            for model_name, model in models.items():
                path = os.path.join(model_dir, f"{model_name}_{save_format}.pth")
                utils.save_checkpoint(model, optims[model_name], learning_rate, epoch, path)

            if hps.save_every_weights == "1":
                ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                logger.info(
                    "saving ckpt %s_%s"
                    % (
                        name_epoch,
                        savee(
                            ckpt,
                            hps.sample_rate,
                            hps.if_f0,
                            f"{name_epoch}_s{global_step}",
                            epoch,
                            hps.version,
                            hps,
                        ),
                    )
                )

    stopbtn = False
    try:
        with open("csvdb/stop.csv", 'r') as csv_file:
            stopbtn_str = next(csv.reader(csv_file), [None])[0]
            if stopbtn_str is not None: stopbtn = stopbtn_str.lower() == 'true'
    except (ValueError, TypeError, FileNotFoundError, IndexError) as e:
        print(f"Handling exception: {e}")
        stopbtn = False

    if stopbtn:
        logger.info("Stop Button was pressed. The program is closed.")
        ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
        logger.info(f"Saving final ckpt:{savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps)}")
        sleep(1)
        reset_stop_flag()
        os._exit(2333333)

    if rank == 0:
        logger.info(f"====> Epoch: {epoch} {epoch_recorder.record()}")

        if epoch >= hps.total_epoch:
            logger.info("Training is done. The program is closed.")
            save_checkpoint(hps.name)
            os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
