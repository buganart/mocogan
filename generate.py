# coding: utf-8

import os
import argparse
import glob
import time
import math
import skvideo.io
import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.autograd import Variable

from models import Discriminator_I, Discriminator_V, Generator_I, GRU


parser = argparse.ArgumentParser(description="Start generate MoCoGAN.....")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="set the output directory of generated files",
)
parser.add_argument(
    "--resume_id",
    type=str,
    default=None,
    help="set wandb run id of logged run to resume from there",
)
parser.add_argument(
    "--num_generate_video",
    type=int,
    default=3,
    help="the height and width of input (resized) video",
)

args = parser.parse_args()
# these variables may be when resume run
resume_id = args.resume_id
video_dir = args.video_dir
num_generate_video = int(args.num_generate_video)

if num_generate_video <= 0:
    import sys
    sys.exit()

""" set wandb run """
# args will be replaced by the one stored in wandb
api = wandb.Api()
previous_run = api.run(f"demiurge/moco-gan/{resume_id}")
args = argparse.Namespace(**previous_run.config)

run = wandb.init(
    project="moco-gan",
    id=resume_id,
    entity="demiurge",
    resume=True,
    dir="./",
    mode="offline",
)

print("run id: " + str(wandb.run.id))
print("run name: " + str(wandb.run.name))

""" parameters """
# cuda = args.cuda
# ngpu = args.ngpu
batch_size = args.batch_size

n_iter = args.niter

lr = args.lr
img_size = args.img_size
nc = args.nc
ndf = args.ndf
ngf = args.ngf
d_E = args.d_E
hidden_size = args.hidden_size
d_C = args.d_C
d_M = args.d_M
nz = d_C + d_M
criterion = nn.BCELoss()
video_epoch = args.video_epoch
checkpoint_epoch = args.checkpoint_epoch

ngpu = torch.cuda.device_count()

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if ngpu > 0:
    torch.cuda.set_device(0)
    cuda = 1
else:
    cuda = -1


""" prepare video sampling """
T = 16

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T + 1))
    end = start + T
    return video[:, start:end, :, :]


# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    start = np.random.randint(0, noise.size(1) - (T + 1))
    end = start + T
    return noise[:, start:end, :, :, :]


# def random_choice():
#     X = []
#     for _ in range(batch_size):
#         if n_videos > 1:
#             video = videos[np.random.randint(0, n_videos - 1)]
#         else:
#             video = videos[0]
#         video = torch.Tensor(trim(video))
#         X.append(video)
#     X = torch.stack(X)
#     return X


""" set models """
# dis_i = Discriminator_I(nc, ndf, ngpu=ngpu)
# dis_v = Discriminator_V(nc, ndf, T=T, ngpu=ngpu)
gen_i = Generator_I(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, gpu=cuda)
gru.initWeight()


def save_video(fake_video, index, run_id):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    file_path = os.path.join(video_dir, f"fakeVideo_index{index}+{run_id}.mp4")
    skvideo.io.vwrite(file_path, outputdata)
    print(f"saved video {index}")


""" adjust to cuda """

if cuda == True:
    gen_i.cuda()
    gru.cuda()



""" use pre-trained models """


print(f"resuming model from wandb run_id: {resume_id}......")
# DI_model = wandb.restore("Discriminator_I.model")
# DV_model = wandb.restore("Discriminator_V.model")
GI_model = wandb.restore("Generator_I.model")
GRU_model = wandb.restore("GRU.model")

# dis_i.load_state_dict(torch.load(DI_model.name))
# dis_v.load_state_dict(torch.load(DV_model.name))
gen_i.load_state_dict(torch.load(GI_model.name))
gru.load_state_dict(torch.load(GRU_model.name))




""" gen input noise for fake video """


def gen_z(n_frames):
    z_C = Variable(torch.randn(batch_size, d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    eps = Variable(torch.randn(batch_size, d_E))
    if cuda == True:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps, n_frames).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    return z.view(batch_size, n_frames, nz, 1, 1)


""" train models """
print("start prediction......")
start_time = time.time()

i = 0
while i < num_generate_video:
    """ generate fake/predict images """
    # 5 seconds, 25FPS
    n_frames = 5*25
    Z = gen_z(n_frames)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
    # trim => (batch_size, T, nz, 1, 1)
    Z = trim_noise(Z)
    # generate videos
    Z = Z.contiguous().view(batch_size * T, nz, 1, 1)
    fake_videos = gen_i(Z)
    fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
    # transpose => (batch_size, nc, T, img_size, img_size)
    fake_videos = fake_videos.transpose(2, 1)


    for v in range(fake_videos.shape[0]):
        save_video(
            fake_videos[v].data.cpu().numpy().transpose(1, 2, 3, 0),
            i,
            str(wandb.run.id),
        )
        
        i += 1
        if i >= num_generate_video:
            break

