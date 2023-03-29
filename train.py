import os
local_rank = int(os.environ["LOCAL_RANK"]) #也可以自动获取
import torch
import torch.optim as optim
import customAudioDataset as data
import numpy as np
import random
from tqdm import tqdm
from model import EncodecModel 
from msstftd import MultiScaleSTFTDiscriminator
from audio_to_mel import Audio2Mel
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore")
import wandb
wandb.init(project='encodec', name='train_encodec')

config = wandb.config

config.batch_size = 2
config.tensor_cut = 3200000
config.max_epoch = 10000 
config.save_folder = f'saves/new7/'
config.save_location = f'{config.save_folder}batch{config.batch_size}_cut{config.tensor_cut}_' 
config.lr = 3e-4
config.disc_lr = 3e-4
config.seed = 3401
config.target_bandwidths = [1.5, 3., 6, 12., 24.]
config.sample_rate = 24_000
config.channels = 1
config.log_interval = 100
config.step_size = 1000
config.disc_step_size =1000
config.gamma = 0.1
config.disc_gamma = 0.1
config.fixed_length = 0
config.data_parallel = True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def total_loss(fmap_real, logits_fake, fmap_fake, wav1, wav2, sample_rate=24000):
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    loss = torch.tensor([0.0], device='cuda', requires_grad=True)
    factor = 100 / (len(fmap_real) * len(fmap_real[0]))

    for tt1 in range(len(fmap_real)):
        loss = loss + (torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake))
        for tt2 in range(len(fmap_real[tt1])):
            loss = loss + (l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) * factor)
    loss = loss * (2/3)

    for i in range(5, 11):
        fft = Audio2Mel(win_length=2 ** i, hop_length=2 ** i // 4, n_mel_channels=64, sampling_rate=sample_rate)
        loss = loss + l1Loss(fft(wav1), fft(wav2)) + l2Loss(fft(wav1), fft(wav2))
    loss = (loss / 6) + l1Loss(wav1, wav2)
    return loss

def disc_loss(logits_real, logits_fake):
    cx = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(cx(1-logits_real[tt1])) + torch.mean(cx(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors

def train_one_step(epoch,optimizer,optimizer_disc, model, disc, trainloader):
    last_loss = 0
    train_d = False
    for input_wav in tqdm(trainloader):
        train_d = not train_d
        input_wav = input_wav.cuda()

        optimizer.zero_grad()
        model.zero_grad()
        optimizer_disc.zero_grad()
        disc.zero_grad()

        output, loss_enc, _ = model(input_wav)

        logits_real, fmap_real = disc(input_wav)
        if train_d:
            logits_fake, _ = disc(model(input_wav)[0].detach())
            loss = disc_loss(logits_real, logits_fake)
            if loss > last_loss/2:
                loss.backward()
                optimizer_disc.step()
            last_loss = 0

        logits_fake, fmap_fake = disc(output)
        loss = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)
        last_loss += loss.item()
        loss_enc.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
    if dist.get_rank() == 0:
        wandb.log({
            'epoch': epoch,
            'loss': loss.item(),
        })
        print(f'| epoch: {epoch} | loss: {loss.item()} |')

def train(train_csv_path,config):
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    set_seed(config.seed)

    if config.fixed_length > 0:
        trainset = data.CustomAudioDataset(train_csv_path,tensor_cut=config.tensor_cut, fixed_length=config.fixed_length)
    else:
        trainset = data.CustomAudioDataset(train_csv_path,tensor_cut=config.tensor_cut)
    
    if config.data_parallel:
        # 创建Dataloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,)



    model = EncodecModel._get_model(
                config.target_bandwidths, config.sample_rate, config.channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=1., name='my_encodec_24khz')
    disc_model = MultiScaleSTFTDiscriminator(filters=32)
    if config.data_parallel:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
        disc_model.cuda()
        disc_model = torch.nn.parallel.DistributedDataParallel(disc_model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)

    model.train()
    disc_model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config.disc_lr}], betas=(0.5, 0.9))
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    disc_scheduler = StepLR(optimizer_disc, step_size=config.disc_step_size, gamma=config.disc_gamma)

    for epoch in range(1, config.max_epoch):
        train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader)
        scheduler.step()
        disc_scheduler.step()
        if epoch % config.log_interval == 0 and dist.get_rank() == 0:
            torch.save(model.module.cpu().state_dict(), f'{config.save_location}epoch{epoch}.pth')
            torch.save(disc_model.module.cpu().state_dict(), f'{config.save_location}epoch{epoch}_disc.pth')





torch.cuda.set_device(local_rank) # 调整计算的位置
torch.distributed.init_process_group(backend='nccl') # 选择nccl后端，初始化进程组   
train_csv_path = '/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/librispeech_train960h.csv'
train(train_csv_path=train_csv_path,config=config)

