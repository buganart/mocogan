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


parser = argparse.ArgumentParser(description="Start trainning MoCoGAN.....")
parser.add_argument("--cuda", type=int, default=1, help="set -1 when you use cpu")
parser.add_argument(
    "--ngpu", type=int, default=-1, help="set the number of gpu you use"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="set batch_size, default: 16"
)
parser.add_argument(
    "--niter", type=int, default=120000, help="set num of iterations, default: 120000"
)
parser.add_argument(
    "--out_dir",
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
    "--mode_run",
    type=bool,
    default=True,
    help="set wandb run mode ('run':True or 'offline':False)",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0002,
    help="learning rate of the model (all components)",
)
parser.add_argument(
    "--img_size",
    type=int,
    default=96,
    help="the height and width of input (resized) video",
)
parser.add_argument(
    "--nc", type=int, default=3, help="the number of channel of input (resized) video"
)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--d_E", type=int, default=10)
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--d_C", type=int, default=50)
parser.add_argument("--d_M", type=int, default=10)
parser.add_argument(
    "--video_epoch",
    type=int,
    default=1000,
    help="the number of epoch to save 1 generated video",
)
parser.add_argument(
    "--checkpoint_epoch",
    type=int,
    default=10000,
    help="the number of epoch to save model checkpoint",
)


args = parser.parse_args()
# these variables may be when resume run
resume_id = args.resume_id
mode_run = args.mode_run
out_dir = args.out_dir
cuda = args.cuda
ngpu = args.ngpu
batch_size = args.batch_size


if mode_run:
    mode = "run"
else:
    mode = "offline"

""" set wandb run """
start_epoch = 1
if resume_id:
    run_id = resume_id

    # args will be replaced by the one stored in wandb
    api = wandb.Api()
    previous_run = api.run(f"demiurge/moco-gan/{resume_id}")
    args = argparse.Namespace(**previous_run.config)
    start_epoch = previous_run.lastHistoryStep
    # as checkpoint is saved based on args.checkpoint_epoch
    start_epoch = start_epoch - (start_epoch % args.checkpoint_epoch) + 1
else:
    run_id = wandb.util.generate_id()

run = wandb.init(
    project="moco-gan",
    id=run_id,
    entity="demiurge",
    resume=True,
    dir="./",
    mode=mode,
)
wandb.config.update(args, allow_val_change=True)
print("run id: " + str(wandb.run.id))
print("run name: " + str(wandb.run.name))

""" parameters """
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

if cuda > 0 and ngpu < 0:
    ngpu = torch.cuda.device_count()

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)


""" prepare dataset """

current_path = "./"

resized_path = os.path.join(current_path, "resized_data")
files = glob.glob(resized_path + "/*")
videos = [skvideo.io.vread(file) for file in files]
# transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
videos = [video.transpose(3, 0, 1, 2) / 255.0 for video in videos]


""" prepare video sampling """

n_videos = len(videos)
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


def random_choice():
    X = []
    for _ in range(batch_size):
        if n_videos > 1:
            video = videos[np.random.randint(0, n_videos - 1)]
        else:
            video = videos[0]
        video = torch.Tensor(trim(video))
        X.append(video)
    X = torch.stack(X)
    return X


# video length distribution
video_lengths = [video.shape[1] for video in videos]

""" set models """
dis_i = Discriminator_I(nc, ndf, ngpu=ngpu)
dis_v = Discriminator_V(nc, ndf, T=T, ngpu=ngpu)
gen_i = Generator_I(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, gpu=cuda)
gru.initWeight()


""" prepare for train """

label = torch.FloatTensor()


def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60 ** 2) * 24))
    h = math.floor(s / (60 ** 2)) - d * 24
    m = math.floor(s / 60) - h * 60 - d * 24 * 60
    s = s - m * 60 - h * (60 ** 2) - d * 24 * (60 ** 2)
    return "%dd %dh %dm %ds" % (d, h, m, s)


trained_path = os.path.join(current_path, "trained_models")


def checkpoint(model, optimizer, epoch):
    filename = os.path.join(
        trained_path, "%s_epoch-%d" % (model.__class__.__name__, epoch)
    )
    torch.save(model.state_dict(), filename + ".model")
    torch.save(optimizer.state_dict(), filename + ".state")

    # also save as latest checkpoint
    filename_latest = os.path.join(current_path, "%s" % (model.__class__.__name__))
    torch.save(model.state_dict(), filename_latest + ".model")
    torch.save(optimizer.state_dict(), filename_latest + ".state")

    # upload to wandb
    wandb.save(filename + ".model")
    wandb.save(filename + ".state")
    wandb.save(filename_latest + ".model")
    wandb.save(filename_latest + ".state")


def save_video(fake_video, epoch, run_id):
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, "generated_videos")
    file_path = os.path.join(dir_path, f"fakeVideo_epoch-{epoch}+{run_id}.mp4")
    skvideo.io.vwrite(file_path, outputdata)
    wandb.log({"generated_videos": wandb.Video(file_path)}, step=epoch)
    if out_dir is not None:
        file_path = os.path.join(out_dir, f"fakeVideo_epoch-{epoch}+{run_id}.mp4")
        skvideo.io.vwrite(file_path, outputdata)


""" adjust to cuda """

if cuda == True:
    dis_i.cuda()
    dis_v.cuda()
    gen_i.cuda()
    gru.cuda()
    criterion.cuda()
    label = label.cuda()


# setup optimizer
betas = (0.5, 0.999)
optim_Di = optim.Adam(dis_i.parameters(), lr=lr, betas=betas)
optim_Dv = optim.Adam(dis_v.parameters(), lr=lr, betas=betas)
optim_Gi = optim.Adam(gen_i.parameters(), lr=lr, betas=betas)
optim_GRU = optim.Adam(gru.parameters(), lr=lr, betas=betas)


""" use pre-trained models """

if resume_id:
    print(f"resuming model from wandb run_id: {resume_id}......")
    DI_model = wandb.restore("Discriminator_I.model")
    DV_model = wandb.restore("Discriminator_V.model")
    GI_model = wandb.restore("Generator_I.model")
    GRU_model = wandb.restore("GRU.model")
    DI_state = wandb.restore("Discriminator_I.state")
    DV_state = wandb.restore("Discriminator_V.state")
    GI_state = wandb.restore("Generator_I.state")
    GRU_state = wandb.restore("GRU.state")

    dis_i.load_state_dict(torch.load(DI_model.name))
    dis_v.load_state_dict(torch.load(DV_model.name))
    gen_i.load_state_dict(torch.load(GI_model.name))
    gru.load_state_dict(torch.load(GRU_model.name))
    optim_Di.load_state_dict(torch.load(DI_state.name))
    optim_Dv.load_state_dict(torch.load(DV_state.name))
    optim_Gi.load_state_dict(torch.load(GI_state.name))
    optim_GRU.load_state_dict(torch.load(GRU_state.name))


""" calc grad of models """


def bp_i(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_i(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    return err.detach(), outputs.detach().mean()


def bp_v(inputs, y, retain=False):
    label.resize_(inputs.size(0)).fill_(y)
    labelv = Variable(label)
    outputs = dis_v(inputs)
    err = criterion(outputs, labelv)
    err.backward(retain_graph=retain)
    return err.detach(), outputs.detach().mean()


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
print("start training......")
start_time = time.time()

for epoch in range(start_epoch, n_iter + 1):
    """prepare real images"""
    # real_videos.size() => (batch_size, nc, T, img_size, img_size)
    real_videos = random_choice()
    if cuda == True:
        real_videos = real_videos.cuda()
    real_videos = Variable(real_videos)
    real_img = real_videos[:, :, np.random.randint(0, T), :, :]

    """ prepare fake images """
    # note that n_frames is sampled from video length distribution
    n_frames = video_lengths[np.random.randint(0, n_videos)]
    Z = gen_z(n_frames)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
    # trim => (batch_size, T, nz, 1, 1)
    Z = trim_noise(Z)
    # generate videos
    Z = Z.contiguous().view(batch_size * T, nz, 1, 1)
    fake_videos = gen_i(Z)
    fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
    # transpose => (batch_size, nc, T, img_size, img_size)
    fake_videos = fake_videos.transpose(2, 1)
    # img sampling
    fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

    """ train discriminators """
    # video
    dis_v.zero_grad()
    err_Dv_real, Dv_real_mean = bp_v(real_videos, 0.9)
    err_Dv_fake, Dv_fake_mean = bp_v(fake_videos.detach(), 0)
    err_Dv = err_Dv_real + err_Dv_fake
    optim_Dv.step()
    # image
    dis_i.zero_grad()
    err_Di_real, Di_real_mean = bp_i(real_img, 0.9)
    err_Di_fake, Di_fake_mean = bp_i(fake_img.detach(), 0)
    err_Di = err_Di_real + err_Di_fake
    optim_Di.step()

    """ train generators """
    gen_i.zero_grad()
    gru.zero_grad()
    # video. notice retain=True for back prop twice
    err_Gv, _ = bp_v(fake_videos, 0.9, retain=True)
    # images
    err_Gi, _ = bp_i(fake_img, 0.9)
    optim_Gi.step()
    optim_GRU.step()

    if epoch % 100 == 0:
        # wandb log
        log_dict = {}
        log_dict["Loss_Di"] = err_Di
        log_dict["Loss_Dv"] = err_Dv
        log_dict["Loss_Gi"] = err_Gi
        log_dict["Loss_Gv"] = err_Gv
        log_dict["Di_real_mean"] = Di_real_mean
        log_dict["Di_fake_mean"] = Di_fake_mean
        log_dict["Dv_real_mean"] = Dv_real_mean
        log_dict["Dv_fake_mean"] = Dv_fake_mean
        wandb.log(log_dict, step=epoch)

        print(
            "[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f"
            % (
                epoch,
                n_iter,
                timeSince(start_time),
                err_Di,
                err_Dv,
                err_Gi,
                err_Gv,
                Di_real_mean,
                Di_fake_mean,
                Dv_real_mean,
                Dv_fake_mean,
            )
        )

    if epoch % video_epoch == 0:
        save_video(
            fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0),
            epoch,
            str(wandb.run.id),
        )

    if epoch % checkpoint_epoch == 0:
        checkpoint(dis_i, optim_Di, epoch)
        checkpoint(dis_v, optim_Dv, epoch)
        checkpoint(gen_i, optim_Gi, epoch)
        checkpoint(gru, optim_GRU, epoch)
