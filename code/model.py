import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def convlxl(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
                     padding=1, bias=False)


def child_to_parent(child_c_code, classes_child, classes_parent):

    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code,  dim = 1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][arg_parent[i]] = 1
    return parent_c_code


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, c_flag):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.c_flag= c_flag

        if self.c_flag==1 :
            self.in_dim = cfg.GAN.Z_DIM + cfg.SUPER_CATEGORIES
        elif self.c_flag==2:
            self.in_dim = cfg.GAN.Z_DIM + cfg.FINE_GRAINED_CATEGORIES

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 16)


    def forward(self, z_code, code):
        in_code = torch.cat((code, z_code), 1)
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, use_hrc, outf=0, num_residual=cfg.GAN.R_NUM):
        """
        outf : out features
        """
        super(NEXT_STAGE_G, self).__init__()
        if use_hrc == 1:                 # For parent stage
            self.ef_dim = cfg.SUPER_CATEGORIES

        elif use_hrc == 0:               # For child stage
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES

        else:                            # For part stage
            self.ef_dim = cfg.NUM_PARTS

        if outf == 0:
            outf = ngf // 2

        self.gf_dim = ngf
        self.out_dim = outf
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.samesample = sameBlock(ngf, self.out_dim)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
        # print(code.size())
        # print(h_code.size())
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img



class GET_MASK_G(nn.Module):
    def __init__(self, ngf):
        super(GET_MASK_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()
        self.upsampling = Upsample(scale_factor=2, mode='bilinear')
        self.scale_fimg = nn.UpsamplingBilinear2d(size = [126, 126])

    def define_module(self):

        #Background stage
        self.h_net1_bg = INIT_STAGE_G(self.gf_dim * 16, c_flag=2)
        self.img_net1_bg = GET_IMAGE_G(self.gf_dim) # Background generation network

        # Parent stage networks
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, c_flag=1)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc=1)
        self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)  # Parent foreground generation network
        self.img_net2_mask= GET_MASK_G(self.gf_dim // 2) # Parent mask generation network

        # Child stage networks
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 4, use_hrc=0)
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 8) # Child foreground generation network
        self.img_net3_mask = GET_MASK_G(self.gf_dim // 8) # Child mask generation network

        # Part stage networks
        self.h_net4 = NEXT_STAGE_G(self.gf_dim // 2, use_hrc=2)
        self.img_net4 = GET_IMAGE_G(self.gf_dim // 4) # Part foreground generation network
        self.img_net4_mask = GET_MASK_G(self.gf_dim // 4) # Part mask generation network

    def forward(self, z_code, c_code, p_code=None, bg_code=None):

        fake_imgs = []  # Will contain [B_126, P, C]
        fg_imgs = []  # Will contain [P_f]
        mk_imgs = []  # Will contain [P_m, C_m]
        fg_mk = []  # Will contain [P_masked, C_masked]

        pt_mk = []  # Will contain [Pt1_m, Pt2_m, ...]
        pt_fg = []  # Will contain [Pt1_f, Pt2_f, ...]
        pt_masked = []  # Will contain [Pt1_masked, Pt2_masked, ...]
        c_mk = []  # will contain [C1_m, C2_m, ...]
        c_fg = []  # will contain [C1_f, C2_f, ...]
        c_masked = []  # Will contain [C1_masked, C2_masked, ...]

        if cfg.TIED_CODES:
            p_code = child_to_parent(c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES) # Obtaining the parent code from child code
            bg_code = c_code

        # Background stage
        h_code1_bg = self.h_net1_bg(z_code, bg_code)
        B = self.img_net1_bg(h_code1_bg) # Background image
        B_126 = self.scale_fimg(B) # Resizing fake background image from 128x128 to the resolution which background discriminator expects: 126 x 126.
        fake_imgs.append(B_126)

        # Parent stage
        h_code1 = self.h_net1(z_code, p_code)
        h_code2 = self.h_net2(h_code1, p_code)

        P_f = self.img_net2(h_code2) # Parent foreground
        P_m = self.img_net2_mask(h_code2) # Parent mask
        ones_mask_p = torch.ones_like(P_m)
        opp_mask_p = ones_mask_p - P_m
        P_masked = torch.mul(P_f, P_m) # Parent foreground masked
        fg_mk.append(P_masked)
        B_masked = torch.mul(B, opp_mask_p)
        P = P_masked + B_masked # Parent image
        fake_imgs.append(P)
        fg_imgs.append(P_f)
        mk_imgs.append(P_m)

        # Part stage
        s_gpus = cfg.GPU_ID.split(',')
        gpus = [int(ix) for ix in s_gpus]
        num_gpus = len(gpus)
        batch_size = cfg.TRAIN.BATCH_SIZE * num_gpus

        C_masked = torch.zeros([batch_size, 3, 128, 128]).cuda()
        C_m = torch.zeros([batch_size, 1, 128, 128]).cuda()

        for i in range(cfg.NUM_PARTS):
            pti_code = torch.zeros([batch_size, cfg.NUM_PARTS]).cuda()
            pti_code[:, i] = 1

            h_code4 = self.h_net4(h_code2, pti_code)
            Pti_f = self.img_net4(h_code4) # Part foreground
            Pti_m = self.img_net4_mask(h_code4) # Part mask
            Pti_masked = torch.mul(Pti_f, Pti_m) # Part foreground masked

            pt_fg.append(Pti_f)
            pt_mk.append(Pti_m)
            pt_masked.append(Pti_masked)

            # Child stage
            h_code3 = self.h_net3(h_code4, c_code)
            Ci_f = self.img_net3(h_code3) # Child part i foreground
            Ci_m = self.img_net3_mask(h_code3) # Child part i mask
            Ci_masked = torch.mul(Ci_f, Ci_m) # Child part i foreground masked

            c_fg.append(Ci_f)
            c_mk.append(Ci_m)
            c_masked.append(Ci_masked)

            C_m += Ci_m
            C_masked += Ci_masked

        C_m = torch.clamp(C_m, 0, 1)
        mk_imgs.append(C_m)
        # C_masked = torch.clamp(C_masked, -1, 1)
        fg_mk.append(C_masked)

        ones_mask_c = torch.ones_like(C_m)
        opp_mask_c = ones_mask_c - C_m
        P_bg_masked = torch.mul(P, opp_mask_c)
        C = C_masked + P_bg_masked

        fake_imgs.append(C)

        return fake_imgs, fg_imgs, mk_imgs, fg_mk, pt_mk, pt_fg, pt_masked, c_mk, c_fg, c_masked


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block



def encode_parent_and_child_img(ndf): # Defines the encoder network used for parent and child image
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

# Defines the encoder network used for part mask
def encode_part_mask(ndf):
    encode_img = nn.Sequential(
        nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


def encode_background_img(ndf): # Defines the encoder network used for background image
    encode_img = nn.Sequential(
        nn.Conv2d(3, ndf, 4, 2, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return encode_img


class D_NET(nn.Module):
    def __init__(self, stg_no):
        super(D_NET, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.stg_no = stg_no

        if self.stg_no  == 0:
            self.ef_dim = 1
        elif self.stg_no == 1:
            self.ef_dim = cfg.SUPER_CATEGORIES
        elif self.stg_no == 2:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        elif self.stg_no == 3:
            self.ef_dim = cfg.NUM_PARTS
        else:
            print ("Invalid stage number. Set stage number as follows:")
            print ("0 - for background stage")
            print ("1 - for parent stage")
            print ("2 - for child stage")
            print ("3 - for part stage")
            print ("...Exiting now")
            sys.exit(0)
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim

        if self.stg_no == 0:
            self.patchgan_img_code_s16 = encode_background_img(ndf)
            self.uncond_logits1 = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1),
                nn.Sigmoid())
            self.uncond_logits2 = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1),
                nn.Sigmoid())

        elif self.stg_no == 3:
            self.img_code_s16 = encode_part_mask(ndf)
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

            self.logits = nn.Sequential(
                nn.Conv2d(ndf * 8, efg, kernel_size=4, stride=4))

            self.jointConv = Block3x3_leakRelu(ndf * 8, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

        else:
            self.img_code_s16 = encode_parent_and_child_img(ndf)
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

            self.logits = nn.Sequential(
                nn.Conv2d(ndf * 8, efg, kernel_size=4, stride=4))

            self.jointConv = Block3x3_leakRelu(ndf * 8, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var):
        if self.stg_no == 0:
            x_code = self.patchgan_img_code_s16(x_var)
            # Background vs Foreground classification score (0 - background and 1 - foreground)
            classi_score = self.uncond_logits1(x_code)
            # Real/Fake score for the background image
            rf_score = self.uncond_logits2(x_code)
            return [classi_score, rf_score]

        elif self.stg_no > 0:
            x_code = self.img_code_s16(x_var)
            x_code = self.img_code_s32(x_code)
            x_code = self.img_code_s32_1(x_code)
            h_c_code = self.jointConv(x_code)
            # Predicts the parent code and child code in parent and child stage respectively
            code_pred = self.logits(h_c_code)
            # This score is not used in parent stage while training
            rf_score = self.uncond_logits(x_code)
            return [code_pred.view(-1, self.ef_dim), rf_score.view(-1)]


