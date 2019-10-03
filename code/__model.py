import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample
import time
from collections import deque

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
    arg_parent = torch.argmax(child_c_code,  dim=1) / ratio
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
        self.c_flag = c_flag

        if self.c_flag == 1:
            self.in_dim = cfg.GAN.Z_DIM + cfg.SUPER_CATEGORIES
        elif self.c_flag == 2:
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
    def __init__(self, ngf, use_hrc=1, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if use_hrc == 1:  # For parent stage
            self.ef_dim = cfg.SUPER_CATEGORIES

        else:            # For child stage
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES

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
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
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
        self.scale_fimg = nn.UpsamplingBilinear2d(size=[126, 126])

    def define_module(self):

        # Background stage
        self.h_net1_bg = INIT_STAGE_G(self.gf_dim * 16, 2)
        # Background generation network
        self.img_net1_bg = GET_IMAGE_G(self.gf_dim)

        # Parent stage networks
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, 1)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc=1)
        # Parent foreground generation network
        self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        # Parent mask generation network
        self.img_net2_mask = GET_MASK_G(self.gf_dim // 2)

        # Child stage networks
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, use_hrc=0)
        # Child foreground generation network
        self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        # Child mask generation network
        self.img_net3_mask = GET_MASK_G(self.gf_dim // 4)

    def forward(self, z_code, c_code, p_code=None, bg_code=None):

        fake_imgs = []  # Will contain [background image, parent image, child image]
        fg_imgs = []  # Will contain [parent foreground, child foreground]
        mk_imgs = []  # Will contain [parent mask, child mask]
        fg_mk = []  # Will contain [masked parent foreground, masked child foreground]

        if cfg.TIED_CODES:
            # Obtaining the parent code from child code
            p_code = child_to_parent(c_code, cfg.FINE_GRAINED_CATEGORIES, cfg.SUPER_CATEGORIES)
            bg_code = c_code
            print("warning: b and p code have been tampered within generator")

        # Background stage
        h_code1_bg = self.h_net1_bg(z_code, bg_code)
        fake_img1 = self.img_net1_bg(h_code1_bg)  # Background image
        # Resizing fake background image from 128x128 to the resolution which background discriminator expects: 126 x 126.
        fake_img1_126 = self.scale_fimg(fake_img1)
        fake_imgs.append(fake_img1_126)

        # Parent stage
        h_code1 = self.h_net1(z_code, p_code)
        h_code2 = self.h_net2(h_code1, p_code)
        fake_img2_foreground = self.img_net2(h_code2)  # Parent foreground
        fake_img2_mask = self.img_net2_mask(h_code2)  # Parent mask
        ones_mask_p = torch.ones_like(fake_img2_mask)
        opp_mask_p = ones_mask_p - fake_img2_mask
        fg_masked2 = torch.mul(fake_img2_foreground, fake_img2_mask)
        fg_mk.append(fg_masked2)
        bg_masked2 = torch.mul(fake_img1, opp_mask_p)
        fake_img2_final = fg_masked2 + bg_masked2  # Parent image
        fake_imgs.append(fake_img2_final)
        fg_imgs.append(fake_img2_foreground)
        mk_imgs.append(fake_img2_mask)

        # Child stage
        h_code3 = self.h_net3(h_code2, c_code)
        fake_img3_foreground = self.img_net3(h_code3)  # Child foreground
        fake_img3_mask = self.img_net3_mask(h_code3)  # Child mask
        ones_mask_c = torch.ones_like(fake_img3_mask)
        opp_mask_c = ones_mask_c - fake_img3_mask
        fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask)
        fg_mk.append(fg_masked3)
        bg_masked3 = torch.mul(fake_img2_final, opp_mask_c)
        fake_img3_final = fg_masked3 + bg_masked3  # Child image
        fake_imgs.append(fake_img3_final)
        fg_imgs.append(fake_img3_foreground)
        mk_imgs.append(fake_img3_mask)

        return fake_imgs, fg_imgs, mk_imgs, fg_mk


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


# Defines the encoder network used for parent and child image
def encode_parent_and_child_img(ndf, in_c=3):
    encode_img = nn.Sequential(
        nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False),
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


# Defines the encoder network used for background image
def encode_background_img(ndf):
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

        if self.stg_no == 0:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        elif self.stg_no == 1:
            self.ef_dim = cfg.SUPER_CATEGORIES
        elif self.stg_no == 2:
            self.ef_dim = cfg.FINE_GRAINED_CATEGORIES
        else:
            print("Invalid stage number. Set stage number as follows:")
            print("0 - for background stage")
            print("1 - for parent stage")
            print("2 - for child stage")
            print("...Exiting now")
            sys.exit(0)
        self.define_module()
        self.softmax = torch.nn.Softmax(dim=1)

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim

        if self.stg_no == 0:
            self.patchgan_img_code_s16 = encode_background_img(ndf)
            self.uncond_logits1 = nn.Sequential( nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1), nn.Sigmoid() )
            self.uncond_logits2 = nn.Sequential( nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1), nn.Sigmoid() )
        else:
            self.img_code_s16 = encode_parent_and_child_img(ndf)
            self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
            self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
            self.jointConv = Block3x3_leakRelu(ndf * 8, ndf * 8)
            self.logits = nn.Sequential( nn.Conv2d(ndf * 8, efg, kernel_size=4, stride=4) )
            self.uncond_logits = nn.Sequential( nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid() )

    def forward(self, x_var):

        if self.stg_no == 0:
            x_code = self.patchgan_img_code_s16(x_var)
            # Background vs Foreground classification score and RF score for the background image
            classi_score = self.uncond_logits1(x_code)
            rf_score = self.uncond_logits2(x_code)
            return  classi_score, rf_score

        elif self.stg_no > 0:
            x_code = self.img_code_s16(x_var)
            x_code = self.img_code_s32(x_code)
            x_code = self.img_code_s32_1(x_code)
            h_c_code = self.jointConv(x_code)
            # Predicts the parent code and child code in parent and child stage respectively
            code_pred = self.logits(h_c_code).view(-1, self.ef_dim)
            # This score is not used in parent stage while training
            rf_score = self.uncond_logits(x_code)
            return [code_pred, rf_score.view(-1)]






################################# Encoder ######################################

class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.ndf = cfg.GAN.DF_DIM

        print('warning: b,p,c encoder softmax is an argument')
        print('warning: current z encoder use 4*tanh at last layer')
        time.sleep(1)

        self.softmax = torch.nn.Softmax(dim=1)


        self.model_z = nn.Sequential(encode_parent_and_child_img(self.ndf),
                                   downBlock(self.ndf*8, self.ndf*16),
                                   Block3x3_leakRelu(self.ndf*16, self.ndf*8),
                                   Block3x3_leakRelu(self.ndf*8, self.ndf*8),
                                   nn.Conv2d(self.ndf*8, cfg.GAN.Z_DIM,  kernel_size=4, stride=4),
                                   nn.Tanh())


        self.model_b = nn.Sequential(encode_parent_and_child_img(self.ndf),
                                   downBlock(self.ndf*8, self.ndf*16),
                                   Block3x3_leakRelu(self.ndf*16, self.ndf*8),
                                   Block3x3_leakRelu(self.ndf*8, self.ndf*8),
                                   nn.Conv2d(self.ndf*8, cfg.FINE_GRAINED_CATEGORIES,  kernel_size=4, stride=4))

        self.model_p = nn.Sequential(encode_parent_and_child_img(self.ndf),
                                   downBlock(self.ndf*8, self.ndf*16),
                                   Block3x3_leakRelu(self.ndf*16, self.ndf*8),
                                   Block3x3_leakRelu(self.ndf*8, self.ndf*8),
                                   nn.Conv2d(self.ndf*8, cfg.SUPER_CATEGORIES,  kernel_size=4, stride=4))

        self.model_c = nn.Sequential(encode_parent_and_child_img(self.ndf),
                                   downBlock(self.ndf*8, self.ndf*16),
                                   Block3x3_leakRelu(self.ndf*16, self.ndf*8),
                                   Block3x3_leakRelu(self.ndf*8, self.ndf*8),
                                   nn.Conv2d(self.ndf*8, cfg.FINE_GRAINED_CATEGORIES,  kernel_size=4, stride=4))

    # def rgb2gray(self,x):
    #     x = (x*0.5)+0.5
    #     x = 0.2989*x[:,0].unsqueeze(1) + 0.5870*x[:,1].unsqueeze(1) + 0.1140*x[:,2].unsqueeze(1)
    #     return (x-0.5)/0.5


    def forward(self, x_var, type_):

        code_z =   self.model_z(x_var).view(-1, cfg.GAN.Z_DIM)*4
        code_b =   self.model_b(x_var).view(-1, cfg.FINE_GRAINED_CATEGORIES)
        code_p =   self.model_p(x_var).view(-1, cfg.SUPER_CATEGORIES)
        code_c =   self.model_c(x_var).view(-1, cfg.FINE_GRAINED_CATEGORIES)

        if type_ == 'logits':
            return code_z, code_b, code_p, code_c
        if type_ == 'softmax':
            return code_z, self.softmax(code_b),self.softmax(code_p),self.softmax(code_c),


################################## BI_DIS #######################################

class Gaussian(nn.Module):
    def __init__(self, std):
        super(Gaussian, self).__init__()
        self.std = std
    def forward(self, x):
        n =  torch.randn_like(x)*self.std
        return x+n

class ConvT_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, bn=True, activation=None, noise=False, std=None ):
        super(ConvT_Block, self).__init__()
        model = [ nn.ConvTranspose2d( in_c, out_c, k ,s, p) ]

        if bn:
            model.append( nn.BatchNorm2d(out_c) )

        if activation == 'relu':
            model.append( nn.ReLU() )
        elif activation == 'elu':
            model.append( nn.ELU() )
        elif activation == 'leaky':
            model.append( nn.LeakyReLU(0.2) )
        elif activation == 'tanh':
            model.append( nn.Tanh() )
        elif activation == 'sigmoid':
            model.append( nn.Sigmoid() )
        elif activation == 'softmax':
            model.append( nn.Softmax(dim=1) )

        if noise:
            model.append(  Gaussian(std) )

        self.model = nn.Sequential( *model )
    def forward(self,x):
        return self.model(x)

class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=0, bn=True, activation=None, noise=False, std=None ):
        super(Conv_Block, self).__init__()
        model = [ nn.Conv2d( in_c, out_c, k ,s, p) ]

        if bn:
            model.append( nn.BatchNorm2d(out_c) )

        if activation == 'relu':
            model.append( nn.ReLU() )
        if activation == 'elu':
            model.append( nn.ELU() )
        if activation == 'leaky':
            model.append( nn.LeakyReLU(0.2) )
        if activation == 'tanh':
            model.append( nn.Tanh() )
        if activation == 'sigmoid':
            model.append( nn.Sigmoid() )
        if activation == 'softmax':
            model.append( nn.Softmax(dim=1) )

        if noise:
            model.append(  Gaussian(std) )

        self.model = nn.Sequential( *model )
    def forward(self,x):
        return self.model(x)

class Linear_Block(nn.Module):
    def __init__(self, in_c, out_c, bn=True, activation=None, noise=False, std=None  ):
        super(Linear_Block, self).__init__()
        model = [ nn.Linear(in_c, out_c) ]

        if bn:
            model.append( nn.BatchNorm1d(out_c) )

        if activation == 'relu':
            model.append( nn.ReLU() )
        if activation == 'elu':
            model.append( nn.ELU() )
        if activation == 'leaky':
            model.append( nn.LeakyReLU(0.2) )
        if activation == 'tanh':
            model.append( nn.Tanh() )
        if activation == 'sigmoid':
            model.append( nn.Sigmoid() )
        if activation == 'softmax':
            model.append( nn.Softmax(dim=1) )

        if noise:
            model.append(  Gaussian(std) )

        self.model = nn.Sequential( *model )
    def forward(self,x):
        return self.model(x)

class Viewer(nn.Module):
    def __init__(self, shape):
        super(Viewer, self).__init__()
        self.shape = shape
    def forward(self,x):
        return x.view(*self.shape)





class Bi_Dis_base(nn.Module):
    def __init__(self, code_len, ngf=16):
        super(Bi_Dis_base, self).__init__()

        self.image_layer = nn.Sequential(  Conv_Block( 3,     ngf, 4,2,1, bn=False, activation='leaky', noise=False, std=0.3),
                                             Conv_Block( ngf,   ngf*2, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*2, ngf*4, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*4, ngf*8, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*8, ngf*16, 4,2,1, bn=False, activation='leaky', noise=False, std=0.5),
                                             Conv_Block( ngf*16, 512, 4,1,0, bn=False, activation='leaky', noise=False, std=0.5),
                                             Viewer( [-1,512] ) )

        self.code_layer = nn.Sequential( Linear_Block( code_len, 512, bn=False, activation='leaky', noise=True, std=0.5  ),
                                           Linear_Block( 512, 512, bn=False, activation='leaky', noise=True, std=0.3  ),
                                           Linear_Block( 512, 512, bn=False, activation='leaky', noise=True, std=0.3  ) )

        self.joint = nn.Sequential(  Linear_Block(1024,1024, bn=False, activation='leaky', noise=False, std=0.5),
                                     Linear_Block(1024, 1,  bn=False,  activation='None' ),
                                     Viewer([-1]) )

    def forward(self, img, code ):
        t1 = self.image_layer(img)
        t2 = self.code_layer( code )
        return  self.joint(  torch.cat( [t1,t2],dim=1) )





class Bi_Dis(nn.Module):
    def __init__(self):
        super(Bi_Dis, self).__init__()

        print('BD does not have sigmoid or bn, due to wgan')
        time.sleep(1)

        #  z stage BD # # # # # # # # # # # # # # # # # # # #
        self.BD_z = Bi_Dis_base( cfg.GAN.Z_DIM )
        self.BD_b = Bi_Dis_base( cfg.FINE_GRAINED_CATEGORIES )
        self.BD_p = Bi_Dis_base( cfg.SUPER_CATEGORIES )
        self.BD_c = Bi_Dis_base( cfg.FINE_GRAINED_CATEGORIES )


    def forward(self, img, z_code, b_code, p_code, c_code):

        which_pair_z = self.BD_z( img, z_code )
        which_pair_b = self.BD_b( img, b_code )
        which_pair_p = self.BD_p( img, p_code )
        which_pair_c = self.BD_c( img, c_code )

        return which_pair_z, which_pair_b, which_pair_p, which_pair_c




class Sampler():
    def __init__(self):
        super(Sampler, self).__init__()
        "This is a deterministic Sampler"
    def __call__(self, logits):
        idx = logits.argmax(dim=1)
        hard = torch.zeros_like(logits).scatter_(1, idx.view(-1, 1), 1)
        return hard


class Code_Buffer():
    def __init__(self, max_len, sample=False):
        # if sample then p c code will be sampled
        self.z_container = deque(maxlen=max_len)
        self.b_container = deque(maxlen=max_len)
        self.p_container = deque(maxlen=max_len)
        self.c_container = deque(maxlen=max_len)

        self.sample = sample
        if sample:
            self.sampler = Sampler()


    def append(self, z, b, p, c):

        if self.sample:
            p = self.sampler(p)
            c = self.sampler(c)

        for i in range(z.shape[0]):
            self.z_container.append(z[i])
            self.b_container.append(b[i])
            self.p_container.append(p[i])
            self.c_container.append(c[i])

    def fetch(self, bs):
        idxs = torch.LongTensor(bs).random_(0, len(self.z_container))
        zz, bb, pp, cc = [],[],[],[]
        for idx in idxs:
            zz.append( self.z_container[idx] )
            bb.append( self.b_container[idx] )
            pp.append( self.p_container[idx] )
            cc.append( self.c_container[idx] )
        return torch.stack(zz), torch.stack(bb), torch.stack(pp), torch.stack(cc)


    def __len__(self):
        return len(self.z_container)






