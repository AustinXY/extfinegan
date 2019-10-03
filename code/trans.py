from __future__ import print_function
from six.moves import range
import sys
import numpy as np
import os
import random
import time
from PIL import Image
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
from torch.nn.functional import softmax, log_softmax
from torch.nn.functional import cosine_similarity
from tensorboardX import summary
from tensorboardX import FileWriter

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file

from model import G_NET, D_NET
import cv2

def save_image(images, save_dir, iname, flag=1):
    """
    flag = 0 : grey scale image
    flag = 1 : RGB
    """

    img_name = '%s.png' % (iname)
    full_path = os.path.join(save_dir, img_name)

    if flag == 1:
        vutils.save_image(images.data, full_path, normalize=True)

        # img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
        # ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        # im = Image.fromarray(ndarr)
        # im.save(full_path)

    else:
        # print(iname)
        img = images.mul(255).clamp(0, 255).byte()
        ndarr = img.data.cpu().numpy()
        ndarr = np.reshape(ndarr, (ndarr.shape[-2], ndarr.shape[-1], 1))
        ndarr = np.repeat(ndarr, 3, axis=2)
        im = Image.fromarray(ndarr)
        im.save(full_path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


cfg_from_file('cfg/trans.yml')

save_dir = os.path.join(cfg.SAVE_DIR, 'images')
mkdir_p(save_dir)
s_gpus = cfg.GPU_ID.split(',')
gpus = [int(ix) for ix in s_gpus]
num_gpus = len(gpus)
torch.cuda.set_device(gpus[0])
cudnn.benchmark = True
batch_size = cfg.TRAIN.BATCH_SIZE * num_gpus

netG = G_NET()
netG.apply(weights_init)
netG = torch.nn.DataParallel(netG, device_ids=[3])
model_dict = netG.state_dict()

state_dict = \
    torch.load(cfg.TRAIN.NET_G,
                map_location=lambda storage, loc: storage)

state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

model_dict.update(state_dict)
netG.load_state_dict(model_dict)
print('Load ', cfg.TRAIN.NET_G,)
if cfg.CUDA:
    netG.cuda()

netG.eval()

nz = cfg.GAN.Z_DIM


# r: change c
# c: change z
NUM_ROW = 10
NUM_COL = 10
p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

for j in range(batch_size):
    bg_code[j][b_ind] = 1
    p_code[j][p_ind] = 1

c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES - NUM_ROW),1)[0]
noise_full = torch.FloatTensor(NUM_COL, batch_size, nz)
noise_full.data.normal_(0, 1)
for k in range(NUM_ROW):

    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    for j in range(batch_size):
        c_code[j][c_ind + k] = 1

    row_li = list()
    for i in range(NUM_COL):

        noise = noise_full[i]

        fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code)
        row_li.append(fake_imgs[2][0])

    im_row = row_li[0]
    for n in range(1, len(row_li)):
        im_row = torch.cat((im_row, row_li[n]), 2)

    save_image(im_row, save_dir, 'r:c;c:z_' + str(k))


# r: change p
# c: change c
NUM_ROW = 10
NUM_COL = 10

b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES - NUM_COL),1)[0]
bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

for j in range(batch_size):
    bg_code[j][b_ind] = 1

p_ind = random.sample(range(cfg.SUPER_CATEGORIES - NUM_ROW),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES - NUM_COL),1)[0]
for k in range(NUM_ROW):

    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    for j in range(batch_size):
        p_code[j][p_ind + k] = 1

    row_li = list()
    for i in range(NUM_COL):

        c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
        for j in range(batch_size):
            c_code[j][c_ind + i] = 1

        fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code)
        row_li.append(fake_imgs[2][0])

    im_row = row_li[0]
    for n in range(1, len(row_li)):
        im_row = torch.cat((im_row, row_li[n]), 2)

    save_image(im_row, save_dir, 'r:p;c:c_' + str(k))

# r: change p c z
# c: change bg
NUM_ROW = 10
NUM_COL = 10

for k in range(NUM_ROW):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    row_li = list()
    for i in range(NUM_COL):

        b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
        bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
        for j in range(batch_size):
            bg_code[j][b_ind] = 1

        fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code)
        row_li.append(fake_imgs[2][0])

    im_row = row_li[0]
    for n in range(1, len(row_li)):
        im_row = torch.cat((im_row, row_li[n]), 2)

    save_image(im_row, save_dir, 'r:pcz;c:bg_' + str(k))


# r: change p
# c: change z
NUM_ROW = 10
NUM_COL = 10

b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

for j in range(batch_size):
    bg_code[j][b_ind] = 1
    c_code[j][c_ind] = 1

p_ind = random.sample(range(cfg.SUPER_CATEGORIES - NUM_ROW),1)[0]
noise_full = torch.FloatTensor(NUM_COL, batch_size, nz)
noise_full.data.normal_(0, 1)
for k in range(NUM_ROW):

    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    for j in range(batch_size):
        p_code[j][p_ind + k] = 1

    row_li = list()
    for i in range(NUM_COL):

        noise = noise_full[i]

        fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code)
        row_li.append(fake_imgs[2][0])

    im_row = row_li[0]
    for n in range(1, len(row_li)):
        im_row = torch.cat((im_row, row_li[n]), 2)

    save_image(im_row, save_dir, 'r:p;c:z_' + str(k))



# interpolation
num_intp = 11

p_ind_old = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise_old = torch.FloatTensor(batch_size, nz)
noise_old.data.normal_(0, 1)

p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

p_code = torch.zeros([num_intp, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
noisef = torch.zeros([num_intp, nz])

for my_it in range(num_intp):
    noisef[my_it] = ((1-(float(my_it)/(float(num_intp-1)))) * noise_old) + (
        ((float(my_it)/(float(num_intp-1)))) * noise)

    bg_code[my_it, b_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    bg_code[my_it, b_ind] = (float(my_it)/(float(num_intp-1)))

    p_code[my_it, p_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    p_code[my_it, p_ind] = (float(my_it)/(float(num_intp-1)))

    c_code[my_it, c_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    c_code[my_it, c_ind] = (float(my_it)/(float(num_intp-1)))

fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noisef, c_code, p_code, bg_code)

temp = fake_imgs[2][0]
for i in range(1, num_intp):
    temp = torch.cat((temp, fake_imgs[2][i]), 2)

# print(fake_imgs[2].size())
save_image(temp, save_dir, 'interpolation_0')

p_ind_old = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise_old = torch.FloatTensor(batch_size, nz)
noise_old.data.normal_(0, 1)

p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

p_code = torch.zeros([num_intp, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
noisef = torch.zeros([num_intp, nz])

for my_it in range(num_intp):
    noisef[my_it] = ((1-(float(my_it)/(float(num_intp-1)))) * noise_old) + (
        ((float(my_it)/(float(num_intp-1)))) * noise)

    bg_code[my_it, b_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    bg_code[my_it, b_ind] = (float(my_it)/(float(num_intp-1)))

    p_code[my_it, p_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    p_code[my_it, p_ind] = (float(my_it)/(float(num_intp-1)))

    c_code[my_it, c_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    c_code[my_it, c_ind] = (float(my_it)/(float(num_intp-1)))

fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noisef, c_code, p_code, bg_code)

temp = fake_imgs[2][0]
for i in range(1, num_intp):
    temp = torch.cat((temp, fake_imgs[2][i]), 2)

# print(fake_imgs[2].size())
save_image(temp, save_dir, 'interpolation_1')

p_ind_old = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise_old = torch.FloatTensor(batch_size, nz)
noise_old.data.normal_(0, 1)

p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

p_code = torch.zeros([num_intp, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
noisef = torch.zeros([num_intp, nz])

for my_it in range(num_intp):
    noisef[my_it] = ((1-(float(my_it)/(float(num_intp-1)))) * noise_old) + (
        ((float(my_it)/(float(num_intp-1)))) * noise)

    bg_code[my_it, b_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    bg_code[my_it, b_ind] = (float(my_it)/(float(num_intp-1)))

    p_code[my_it, p_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    p_code[my_it, p_ind] = (float(my_it)/(float(num_intp-1)))

    c_code[my_it, c_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    c_code[my_it, c_ind] = (float(my_it)/(float(num_intp-1)))

fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noisef, c_code, p_code, bg_code)

temp = fake_imgs[2][0]
for i in range(1, num_intp):
    temp = torch.cat((temp, fake_imgs[2][i]), 2)

# print(fake_imgs[2].size())
save_image(temp, save_dir, 'interpolation_2')

p_ind_old = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise_old = torch.FloatTensor(batch_size, nz)
noise_old.data.normal_(0, 1)

p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

p_code = torch.zeros([num_intp, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
noisef = torch.zeros([num_intp, nz])

for my_it in range(num_intp):
    noisef[my_it] = ((1-(float(my_it)/(float(num_intp-1)))) * noise_old) + (
        ((float(my_it)/(float(num_intp-1)))) * noise)

    bg_code[my_it, b_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    bg_code[my_it, b_ind] = (float(my_it)/(float(num_intp-1)))

    p_code[my_it, p_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    p_code[my_it, p_ind] = (float(my_it)/(float(num_intp-1)))

    c_code[my_it, c_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    c_code[my_it, c_ind] = (float(my_it)/(float(num_intp-1)))

fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noisef, c_code, p_code, bg_code)

temp = fake_imgs[2][0]
for i in range(1, num_intp):
    temp = torch.cat((temp, fake_imgs[2][i]), 2)

# print(fake_imgs[2].size())
save_image(temp, save_dir, 'interpolation_3')

p_ind_old = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise_old = torch.FloatTensor(batch_size, nz)
noise_old.data.normal_(0, 1)

p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

p_code = torch.zeros([num_intp, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
noisef = torch.zeros([num_intp, nz])

for my_it in range(num_intp):
    noisef[my_it] = ((1-(float(my_it)/(float(num_intp-1)))) * noise_old) + (
        ((float(my_it)/(float(num_intp-1)))) * noise)

    bg_code[my_it, b_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    bg_code[my_it, b_ind] = (float(my_it)/(float(num_intp-1)))

    p_code[my_it, p_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    p_code[my_it, p_ind] = (float(my_it)/(float(num_intp-1)))

    c_code[my_it, c_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    c_code[my_it, c_ind] = (float(my_it)/(float(num_intp-1)))

fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noisef, c_code, p_code, bg_code)

temp = fake_imgs[2][0]
for i in range(1, num_intp):
    temp = torch.cat((temp, fake_imgs[2][i]), 2)

save_image(temp, save_dir, 'interpolation_4')

p_ind_old = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind_old = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise_old = torch.FloatTensor(batch_size, nz)
noise_old.data.normal_(0, 1)

p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

noise = torch.FloatTensor(batch_size, nz)
noise.data.normal_(0, 1)

p_code = torch.zeros([num_intp, cfg.SUPER_CATEGORIES])
bg_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
c_code = torch.zeros([num_intp, cfg.FINE_GRAINED_CATEGORIES])
noisef = torch.zeros([num_intp, nz])

for my_it in range(num_intp):
    noisef[my_it] = ((1-(float(my_it)/(float(num_intp-1)))) * noise_old) + (
        ((float(my_it)/(float(num_intp-1)))) * noise)

    bg_code[my_it, b_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    bg_code[my_it, b_ind] = (float(my_it)/(float(num_intp-1)))

    p_code[my_it, p_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    p_code[my_it, p_ind] = (float(my_it)/(float(num_intp-1)))

    c_code[my_it, c_ind_old] = 1 - (float(my_it)/(float(num_intp-1)))
    c_code[my_it, c_ind] = (float(my_it)/(float(num_intp-1)))

fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noisef, c_code, p_code, bg_code)

temp = fake_imgs[2][0]
for i in range(1, num_intp):
    temp = torch.cat((temp, fake_imgs[2][i]), 2)

# print(fake_imgs[2].size())
save_image(temp, save_dir, 'interpolation_5')


# sample
chrli = []
pmrli = []
for i in range(10):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        bg_code[j][b_ind] = 1
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

    chrli.append(fake_imgs[2][0])
    pmrli.append(mk_imgs[0][0])

r = chrli[0]
for n in range(1, len(chrli)):
    r = torch.cat((r, chrli[n]), 2)

save_image(r, save_dir, 'samples_0')

r = pmrli[0]
for n in range(1, len(pmrli)):
    r = torch.cat((r, pmrli[n]), 2)

save_image(r, save_dir, 'samplesm_0', 0)

# sample
chrli = []
pmrli = []
for i in range(10):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        bg_code[j][b_ind] = 1
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

    chrli.append(fake_imgs[2][0])
    pmrli.append(mk_imgs[0][0])

r = chrli[0]
for n in range(1, len(chrli)):
    r = torch.cat((r, chrli[n]), 2)

save_image(r, save_dir, 'samples_1')

r = pmrli[0]
for n in range(1, len(pmrli)):
    r = torch.cat((r, pmrli[n]), 2)

save_image(r, save_dir, 'samplesm_1', 0)

# sample
chrli = []
pmrli = []
for i in range(10):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        bg_code[j][b_ind] = 1
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

    chrli.append(fake_imgs[2][0])
    pmrli.append(mk_imgs[0][0])

r = chrli[0]
for n in range(1, len(chrli)):
    r = torch.cat((r, chrli[n]), 2)

save_image(r, save_dir, 'samples_2')

r = pmrli[0]
for n in range(1, len(pmrli)):
    r = torch.cat((r, pmrli[n]), 2)

save_image(r, save_dir, 'samplesm_2', 0)

# sample
chrli = []
pmrli = []
for i in range(10):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        bg_code[j][b_ind] = 1
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

    chrli.append(fake_imgs[2][0])
    pmrli.append(mk_imgs[0][0])

r = chrli[0]
for n in range(1, len(chrli)):
    r = torch.cat((r, chrli[n]), 2)

save_image(r, save_dir, 'samples_3')

r = pmrli[0]
for n in range(1, len(pmrli)):
    r = torch.cat((r, pmrli[n]), 2)

save_image(r, save_dir, 'samplesm_3', 0)

# sample
chrli = []
pmrli = []
for i in range(10):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        bg_code[j][b_ind] = 1
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

    chrli.append(fake_imgs[2][0])
    pmrli.append(mk_imgs[0][0])

r = chrli[0]
for n in range(1, len(chrli)):
    r = torch.cat((r, chrli[n]), 2)

save_image(r, save_dir, 'samples_4')

r = pmrli[0]
for n in range(1, len(pmrli)):
    r = torch.cat((r, pmrli[n]), 2)

save_image(r, save_dir, 'samplesm_4', 0)

# sample
chrli = []
pmrli = []
for i in range(10):

    p_ind = random.sample(range(cfg.SUPER_CATEGORIES),1)[0]
    b_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]
    c_ind = random.sample(range(cfg.FINE_GRAINED_CATEGORIES),1)[0]

    noise = torch.FloatTensor(batch_size, nz)
    noise.data.normal_(0, 1)

    bg_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])
    p_code = torch.zeros([batch_size, cfg.SUPER_CATEGORIES])
    c_code = torch.zeros([batch_size, cfg.FINE_GRAINED_CATEGORIES])

    for j in range(batch_size):
        bg_code[j][b_ind] = 1
        p_code[j][p_ind] = 1
        c_code[j][c_ind] = 1

    fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator

    chrli.append(fake_imgs[2][0])
    pmrli.append(mk_imgs[0][0])

r = chrli[0]
for n in range(1, len(chrli)):
    r = torch.cat((r, chrli[n]), 2)

save_image(r, save_dir, 'samples_5')

r = pmrli[0]
for n in range(1, len(pmrli)):
    r = torch.cat((r, pmrli[n]), 2)

save_image(r, save_dir, 'samplesm_5', 0)
