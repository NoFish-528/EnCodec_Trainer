from pathlib import Path
import torch
from compress import MODELS
from utils import save_audio
import struct

# path to encoded file
input = Path('/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/audio/1_funk_80_beat_4-4_1.wav')

# path to output decoded wav file
output = Path('audio/output_song_new.wav')


device = 'cpu' # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, privateuseone
# notice, using cpu to encode and then cuda to decode throws errors, possibly due to gpu rounding via TF32 operations?

# If you want to use cpu / cuda interchangeably, these NEED to be set to false
# Setting them to true should allow for a speedup but cuda will get different results than cpu
# Also the amount of speedup seems to be negligible
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

model_name = 'my_encodec' # 'encodec_24khz'
model = MODELS[model_name](checkpoint_name='/mnt/lustre/sjtu/home/zkn02/EnCodec_Trainer/outputs/2023-04-11/09-10-26/save/batch4_cut180000_lr5e-05_epoch32_lr5e-05.pt').to(device)
model.set_target_bandwidth(1.5)
encoded_list = []
with open(input, 'rb') as newFile:
    buf = newFile.read()
    l1 = struct.unpack_from('i', buf)[0]
    offset = 1 * 4
    for i in range(l1):
        l2 = struct.unpack_from('i', buf, offset)[0]
        offset += 1 * 4
        tt = torch.tensor(struct.unpack_from('%sh' % l2, buf, offset))
        offset += l2 * 2
        tt = torch.reshape(tt, (32, 1, l2//32)).to(device)
        tt = model.quantizer.decode(tt)
        encoded_list.append((tt, torch.tensor(struct.unpack_from('f', buf, offset)).to(device)))
        offset += 1 * 4

output_wav = model.decode(encoded_list)
output_wav = output_wav.to('cpu')
output_wav = torch.squeeze(output_wav)
output_wav = output_wav[None,:]
save_audio(output_wav, output, model.sample_rate, rescale=True)
