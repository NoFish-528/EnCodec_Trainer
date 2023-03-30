# Notes
Thank you for Mikxox/EnCodec_Trainer
I add some notes to help you understand the code and train the model.
## Step 1 Generate the training data csv

Use the generate_train_file.py to generate the training data.

I Only focus on the data's path and if you use other data official csv, you need to change the customAudioDataset.py.

## Step 2 Train the model
if you want to train the model, you can use the train.py.

This code support the multi-gpu training and single gpu training.

* multi-gpu training

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py data_parallel=True [some args you want to change]
```

* single-gpu training
  
```shell
python train.py [some args you want to change]
```

# EnCodec_Trainer

Implementation to add training function with loss to https://github.com/facebookresearch/encodec \
Used audio_to_mel.py from https://github.com/descriptinc/melgan-neurips \
Some minor adjustments to other code to add new model. \
Changes to the forward function in model.py to make training quantizer easier. \
Only training.py & customAudioDataset.py is new code. \
You can use testing.py to use the trained model, however we use struct as a binary writer. \
This loses some compression power since we have to write 16bits instead of 10bits. \
Download the used database e-gmd from https://magenta.tensorflow.org/datasets/e-gmd

### Important Notice
This code is meant as proof of concept trainer code used to try and train an EnCodec model from scratch.
I did not delve into how the codebooks get updated for quantization.
The quantization code training thus still works via the original code and this means codebooks still get updated when using testing.py.
You will need to put the model in evaluation mode for real-world usage and codebooks might not be well-trained for this.
It is thus recommended to use the pre-trained model made available by the facebook research team as a basis for retraining.

## Citation
If you use the original code or results in your paper, please cite the original work as:
```
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}

@article{Melgan,
      title={MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis}, 
      author={Kundan Kumar and Rithesh Kumar and Thibaukt de Boissiere and Lucas Gestin and Wei Zhen Teoh and Jose Sotelo and Alexandre de Brebisson and Yoshua Bengio and Aaron Courville},
      journal={arXiv preprint arXiv:1910.06711},
      year={2019}
}

@misc{egmd,
    title={Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset},
    author={Lee Callender and Curtis Hawthorne and Jesse Engel},
    year={2020},
    eprint={2004.00188},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
}
```

Also citing the added training code if you use it is always appreciated.
