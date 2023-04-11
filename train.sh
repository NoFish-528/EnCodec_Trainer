CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi_gpu.py \
                        torch_distributed_debug=False \
                        world_size=4 \
                        train_csv_path=/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/librispeech_train100h.csv \
                        max_epoch=200 \
                        step_size=80 \
                        tensor_cut=200000 \
                        disc_step_size=80 \
                        compile_debug=True \
                        find_unused_parameters=True > gpu4_lr3e-4_200epochs_train100h_200000cut_encodec.txt

