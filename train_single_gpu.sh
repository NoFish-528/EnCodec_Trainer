python train.py \
    world_size=4 \
    train_csv_path=/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/librispeech_train100h.csv \
    max_epoch=200 \
    batch_size=4 \
    tensor_cut=180000 \
    train_discriminator=True \
    lr=5e-5 \
    disc_lr=1e-5 \
    log_interval=2 \
    warmup_epoch=20 \
    data_parallel=False