import os
import torch
import torch.optim as optim
import customAudioDataset as data
from customAudioDataset import collate_fn,pad_sequence
from utils import set_seed
from tqdm import tqdm
import torch.nn as nn
import torch._dynamo.config
from model import EncodecModel 
from msstftd import MultiScaleSTFTDiscriminator
from losses import total_loss, disc_loss
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from scheduler import WarmUpLR
import torch.distributed as dist
import torch.multiprocessing as mp
import omegaconf
import hydra
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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

    if dist.get_rank()==0:
        logger.info(f'| epoch: {epoch} | loss: {loss.item()} | loss_enc: {loss_enc.item()} | lr: {optimizer.param_groups[0]["lr"]} | disc_lr: {optimizer_disc.param_groups[0]["lr"]}')
        if config.train_discriminator and epoch > config.warmup_epoch:
            logger.info(f'| loss_disc: {loss_disc.item()}')
def train(local_rank,world_size,config):
    file_handler = logging.FileHandler(f"train_encodec__bs{config.batch_size}_lr{config.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if not config.data_parallel:
        print(config)

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

    if config.data_parallel:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(
            backend='nccl',
            rank=local_rank,
            world_size=world_size)
        
        torch.cuda.set_device(local_rank) 
        torch.cuda.empty_cache()
        if config.compile_debug:
            torch._dynamo.config.verbose=True
            torch._dynamo.config.suppress_errors = True
            torch.autograd.set_detect_anomaly(True)
        if dist.get_rank()==0:
            print(config)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn,pin_memory=True,num_workers=config.num_workers)
        model.cuda()
        model.train_quantization = True
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,find_unused_parameters=config.find_unused_parameters)
        disc_model.cuda()
        disc_model = torch.nn.parallel.DistributedDataParallel(disc_model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,find_unused_parameters=config.find_unused_parameters)
        
        if config.compile:
            model = torch.compile(model,mode="max-autotune")
            disc_model = torch.compile(disc_model,mode="max-autotune")
            
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,pin_memory=True)
        model.cuda()
        disc_model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config.disc_lr}], betas=(0.5, 0.9))
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    disc_scheduler = CosineAnnealingLR(optimizer_disc, T_max=100, eta_min=0)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch,config.warmup_epoch)

    model.train()
    disc_model.train()
    for epoch in range(1, config.max_epoch):
        train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader,config,scheduler,disc_scheduler,warmup_scheduler)
            
        if epoch % config.log_interval == 0 and dist.get_rank()==0:
                torch.save(model.module.state_dict(), f'{config.save_location}epoch{epoch}_lr{config.lr}.pt')
                torch.save(disc_model.module.state_dict(), f'{config.save_location}epoch{epoch}_disc_lr{config.lr}.pt')
    if config.data_parallel:
        dist.destroy_process_group()

@hydra.main(config_path='config', config_name='config')
def main(config):
    if config.torch_distributed_debug:
        os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    if config.data_parallel:
        world_size=config.world_size
        # world_size=
        torch.multiprocessing.set_start_method('spawn')
        mp.spawn(
            train,
            args=(world_size,config,),
            nprocs=world_size,
            join=True
        )
    else:
        train(1,1,config)


if __name__ == '__main__':
    main()