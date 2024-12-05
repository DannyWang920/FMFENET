import math
from typing import List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn import CTCLoss
#from unet_parts import *
from deepspeech_pytorch.configs.train_config import SpectConfig, BiDirectionalConfig, OptimConfig, AdamConfig, \
    SGDConfig, UniDirectionalConfig
from deepspeech_pytorch.decoder import GreedyDecoder, BeamSearchDecoder
from deepspeech_pytorch.validation import CharErrorRate, WordErrorRate


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, h=None):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x, h)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x, h


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            self.n_features,
            self.n_features,
            kernel_size=self.context,
            stride=1,
            groups=self.n_features,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(pl.LightningModule):
    def __init__(self,
                 labels: List,
                 model_cfg: Union[UniDirectionalConfig, BiDirectionalConfig],
                 precision: int,
                 optim_cfg: Union[AdamConfig, SGDConfig],
                 spect_cfg: SpectConfig
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.precision = precision
        self.optim_cfg = optim_cfg
        self.spect_cfg = spect_cfg
        self.bidirectional = True if OmegaConf.get_type(model_cfg) is BiDirectionalConfig else False

        self.labels = labels
        num_classes = len(self.labels)

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True), 
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self.spect_cfg.sample_rate * self.spect_cfg.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=rnn_input_size,
                hidden_size=self.model_cfg.hidden_size,
                rnn_type=self.model_cfg.rnn_type.value,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.model_cfg.hidden_size,
                    hidden_size=self.model_cfg.hidden_size,
                    rnn_type=self.model_cfg.rnn_type.value,
                    bidirectional=self.bidirectional
                ) for x in range(self.model_cfg.hidden_layers - 1)
            )
        )

        self.down1 = DoubleConv(32, 64)
        self.down2 = DoubleConv(64, 128)
        #self.down2 = DoubleConv(32, 32)

        self.up1 = Up(64,32,True)
        # self.up2 = Up(96,32,True)
        
        self.myjpu1 = JPU([64, 128])
        self.myjpu2 = JPU([64, 128])

    #    # 引入多头自注意力机制
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=1312, num_heads=8)
        # self.multihead_attention1 = nn.MultiheadAttention(embed_dim=1312, num_heads=8)
        # # 双注意力模块层数为2层
        self.dual_attention1 = DualAttentionModule(32)
        self.dual_attention2 = DualAttentionModule(128)
                
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self.model_cfg.hidden_size, context=self.model_cfg.lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.model_cfg.hidden_size),
            nn.Linear(self.model_cfg.hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='sum', zero_infinity=True)
        self.focal_loss = FocalLoss()
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        #self.evaluation_decoder = BeamSearchDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )
        self.cer = CharErrorRate(
            decoder=self.evaluation_decoder,
            target_decoder=self.evaluation_decoder
        )

    def forward(self, x, lengths, hs=None):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)
        sizes = x.size()
        
        x1 = self.down1(x)
        #x1 = self.dual_attention1(x1_)
        x2 = self.down2(x1)
        #x2 = self.dual_attention2(x2_)        
        _, _, x = self.myjpu1(x1, x2)
        
        #x = self.up1(x)
        #_, _, x_o = self.myjpu1(x1_, x2_)
        #x = self.dual_attention1(x)
        #x = x + x_o
        diffY = sizes[2] - x.size()[2]
        diffX = sizes[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])        
        # x = self.up1(x2, x1)
        # x = self.up2(x, x_j1)
        # x = self.dual_attention1(x)
        # x = self.dual_attention2(x)    
        
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # if hs is None, create a list of None values corresponding to the number of rnn layers
        #假设 x 经过多头自注意力机制后的形状为 [316, 64, 1312]
        # x, _ = self.multihead_attention(x, x, x)  # 输入和输出形状均为 [time_steps, batch_size, channels * feature_dim]
        # x, _ = self.multihead_attention1(x, x, x)
        # 将输出形状变回原来的形状
        #x = x.transpose(0, 1).transpose(1, 2).contiguous()  # [batch_size, channels * feature_dim, time_steps] -> [64, 1312, 316]
        #x = x.transpose(0, 1).contiguous()
        #x = x.view(sizes[0], sizes[1] , sizes[2], sizes[3])  # [64, 32, 41, 316]

        # x = self.dual_attention1(x)
        # x = self.dual_attention2(x)

        if hs is None:
            hs = [None] * len(self.rnns)

        new_hs = []
        for i, rnn in enumerate(self.rnns):
            x, h = rnn(x, output_lengths, hs[i])
            new_hs.append(h)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths, new_hs

    def training_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        out, output_sizes, hs = self(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)

        loss = self.criterion(out, targets, output_sizes, target_sizes)
        #focal_loss = self.focal_loss(out, targets)
        #loss = loss# + focal_loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(self.device)
        with autocast(enabled=self.precision == 16):
            out, output_sizes, hs = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_sizes)
        self.wer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.cer(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        if OmegaConf.get_type(self.optim_cfg) is SGDConfig:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                momentum=self.optim_cfg.momentum,
                nesterov=True,
                weight_decay=self.optim_cfg.weight_decay
            )
        elif OmegaConf.get_type(self.optim_cfg) is AdamConfig:
            optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=self.optim_cfg.learning_rate,
                betas=self.optim_cfg.betas,
                eps=self.optim_cfg.eps,
                weight_decay=self.optim_cfg.weight_decay
            )
        else:
            raise ValueError("Optimizer has not been specified correctly.")

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optim_cfg.learning_anneal
        )
        return [optimizer], [scheduler]

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)    
        return seq_len.int()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        #MFFE第一层
        self.myConv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #MFFE第二层
        self.myConv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #MFFE第三层
        self.myConv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        #MFFE第四层
        self.myConv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        #分别通过MFFE的不从尺度层，并将特征整合
        x1 = self.myConv1(x)
        x2 = self.myConv2(x)
        x3 = self.myConv3(x)
        x4 = self.myConv4(x)
        x_all = x + x1 + x2 + x3 + x4
        return x_all

import torch
import torch.nn as nn
 
class SEWeightModule(nn.Module):
 
    def __init__(self, channels, reduction=8):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
 
        return weight

 
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
 
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
class PSAModule(nn.Module):
 
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 2, 4, 8]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, int(planes/4), kernel_size=conv_kernels[0], padding=int(conv_kernels[0]/2),
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, int(planes/4), kernel_size=conv_kernels[1], padding=int(conv_kernels[1]/2),
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, int(planes/4), kernel_size=conv_kernels[2], padding=int(conv_kernels[2]/2),
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, int(planes/4), kernel_size=conv_kernels[3], padding=int(conv_kernels[3]/2),
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(int(planes / 4))
        self.split_channel = int(planes / 4)
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
 
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
 
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)
 
        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
 
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=int(in_dim / 8), kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=int(in_dim / 8), kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
class DualAttentionModule(nn.Module):
    """双注意力模块，结合 PSA 和位置注意力模块。
    
    参数:
        channels (int): 输入通道数。
    
    输出是通过 sum fusion 结合的两个模块的输出。
    """
    def __init__(self, channels):
        super(DualAttentionModule, self).__init__()
        self.psa = PSAModule(channels, channels)
        self.pam = PAM_Module(channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        #psa_out = self.psa(x)  # PSA 模块输出
        pam_out = self.pam(x)  # 位置注意力模块输出
        pout = self.up(pam_out)# + psa_out)  # 合并两个输出
        return pout

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DualAttentionModule(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class JPU(nn.Module):
    def __init__(self, in_channels, width=64, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #256
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
        #     # norm_layer(width),
        #     nn.BatchNorm2d(width),
        #     nn.ReLU(inplace=True))
        #128
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            # norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        #64
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            # norm_layer(width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(128, 32, 1)
        self.dilation1 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                    #    norm_layer(width),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                    #    norm_layer(width),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        # self.dilation3 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
        #                             #    norm_layer(width),
        #                                nn.BatchNorm2d(width),
        #                                nn.ReLU(inplace=True))
        # self.dilation4 = nn.Sequential(SeparableConv2d(2*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
        #                             #    norm_layer(width),
        #                                nn.BatchNorm2d(width),
        #                                nn.ReLU(inplace=True))
 
    def forward(self, *inputs):
        feats = [self.conv4(inputs[-1]), self.conv3(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bilinear', align_corners=False)
        #feats[-3] = F.interpolate(feats[-3], (h, w), mode='bilinear', align_corners=False)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat)],dim=1)#, self.dilation3(feat), self.dilation4(feat)], dim=1)
        feat = self.up(feat)        
        feat = self.conv(feat)
        #feat = self.conv(feat)
        return inputs[0], inputs[1], feat


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()
 
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        print("***************************************************")        
        print("inputs.size={}".format(inputs.size()))
        print("targets.size={}".format(targets.size()))
        print("***************************************************")        
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets)
            F_loss = alpha_factor * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss