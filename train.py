import os
import torch
import torch.optim as optim
import customAudioDataset as data
from customAudioDataset import collate_fn,pad_sequence
from utils import set_seed
import numpy as np
import random
from tqdm import tqdm
from model import EncodecModel 
from msstftd import MultiScaleSTFTDiscriminator
from audio_to_mel import Audio2Mel
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from scheduler import WarmUpLR
import hydra
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_one_step(epoch,optimizer,optimizer_disc, model, disc_model, trainloader,config,scheduler,disc_scheduler,warmup_scheduler,):
    for input_wav in tqdm(trainloader):
        if epoch <= config.warmup_epoch:
            warmup_scheduler.step()

        input_wav = input_wav.cuda() #[B, 1, T]: eg. [2, 1, 203760]
        optimizer.zero_grad()
        optimizer_disc.zero_grad()

        output, loss_enc, _ = model(input_wav) #output: [B, 1, T]: eg. [2, 1, 203760] | loss_enc: [1] 
        logits_real, fmap_real = disc_model(input_wav)

        if config.train_discriminator and epoch > config.warmup_epoch:
            logits_fake, _ = disc_model(model(input_wav)[0].detach())
            loss_disc = disc_loss(logits_real, logits_fake)
            # avoid discriminator overpower the encoder
            loss_disc.backward() 
            optimizer_disc.step()
  

        logits_fake, fmap_fake = disc_model(output)
        loss = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output) + loss_enc
        loss.backward()
        optimizer.step()
    if epoch > config.warmup_epoch:
        scheduler.step()
        disc_scheduler.step()

    logger.info(f'| epoch: {epoch} | loss: {loss.item()} | loss_enc: {loss_enc.item()} | lr: {optimizer.param_groups[0]["lr"]} | disc_lr: {optimizer_disc.param_groups[0]["lr"]}')
    if config.train_discriminator and epoch > config.warmup_epoch:
        logger.info(f'| loss_disc: {loss_disc.item()}')
@hydra.main(config_path='config', config_name='config')
def train(config):
    file_handler = logging.FileHandler(f"train_encodec__bs{config.batch_size}_lr{config.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    print(config)
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    if config.seed is not None:
        set_seed(config.seed)

    if config.fixed_length > 0:
        trainset = data.CustomAudioDataset(config.train_csv_path,tensor_cut=config.tensor_cut, fixed_length=config.fixed_length)
    else:
        trainset = data.CustomAudioDataset(config.train_csv_path,tensor_cut=config.tensor_cut)
    
    
    model = EncodecModel._get_model(
                config.target_bandwidths, config.sample_rate, config.channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=1., name='my_encodec')
    disc_model = MultiScaleSTFTDiscriminator(filters=32)
 
 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,pin_memory=True)
    model.cuda()
    disc_model.cuda()
    model.train()
    disc_model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config.disc_lr}], betas=(0.5, 0.9))
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    disc_scheduler = CosineAnnealingLR(optimizer_disc, T_max=100, eta_min=0)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch,config.warmup_epoch)

    for epoch in range(1, config.max_epoch):
        train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader)
        if epoch % config.log_interval == 0:
            torch.save(model.state_dict(), f'{config.save_location}epoch{epoch}.pth')
            torch.save(disc_model.state_dict(), f'{config.save_location}epoch{epoch}_disc.pth')


if __name__ == '__main__':
    train()
