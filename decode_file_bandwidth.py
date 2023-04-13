from pathlib import Path
from utils import save_audio, convert_audio
from compress import compress, decompress, MODELS
import torchaudio
import sys

def check_clipping(wav, rescale):
    if rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)



song = Path('/mnt/lustre/sjtu/home/zkn02/data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac')
output = Path('/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/audio/84-121123-0001_1500_epoch32.ecdc')
outputw = Path('/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/audio/84-121123-0001_1500_epoch32.wav')
model_name = 'my_encodec' # 'encodec_24khz'
bandwidth = 1.5
rescale = True
device = 'cpu' # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, privateuseone


model = MODELS[model_name](checkpoint_name='/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/outputs/2023-04-11/09-10-26/save/batch4_cut180000_lr5e-05_epoch32_lr5e-05.pt').to(device)
model.set_target_bandwidth(bandwidth)

wav, sr = torchaudio.load(song)

wav = convert_audio(wav, sr, 24000, 1)
wav = wav[None,:]
output_wav, _, frames = model.forward(wav)


compressed = compress(model, wav, use_lm=False)
out, out_sample_rate = decompress(compressed)
check_clipping(out, rescale)
save_audio(out, output, out_sample_rate, rescale=rescale)