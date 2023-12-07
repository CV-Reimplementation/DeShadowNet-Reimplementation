import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.functional import pad


class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class TransConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, transposed=True, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(TransConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            transposed, _pair(0), groups, bias)

    def forward(self, input):
        return transposeconv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def transposeconv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv_transpose2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


def maxpool2d_same(x, kernel_size, stride=None, dilation=1):
    """
    Provides the same padding as TensorFlow's 'SAME' for a maxpool operation.
    """
    if stride is None:
        stride = kernel_size

    # Calculate total padding for height and width
    pad_h = max((x.shape[2] - 1) * stride + (kernel_size - 1) * dilation + 1 - x.shape[2], 0)
    pad_w = max((x.shape[3] - 1) * stride + (kernel_size - 1) * dilation + 1 - x.shape[3], 0)
    
    # Split the padding to both sides evenly (or unevenly if odd padding)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding to the input and perform max pooling
    x_padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0)
    return F.max_pool2d(x_padded, kernel_size, stride, 0, dilation)


class ConvTranspose2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(ConvTranspose2dSame, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # The padding attribute is unused and only here for nn.ConvTranspose2d compatibility
        self.padding = 0

        # Define the transposed convolution layer without padding
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, self.kernel_size,
                                                 stride=self.stride, padding=self.padding,
                                                 dilation=self.dilation)

    def forward(self, x):
        # Compute the output size
        output_size = [
            (x.size(2) - 1) * self.stride + 1 - self.stride + self.dilation * (self.kernel_size[0] - 1) + 1,
            (x.size(3) - 1) * self.stride + 1 - self.stride + self.dilation * (self.kernel_size[1] - 1) + 1
        ]
        
        # Calculate the padding size
        pad_h = max((output_size[0] - x.size(2)) // 2, 0)
        pad_w = max((output_size[1] - x.size(3)) // 2, 0)
        
        # Apply padding to the input and perform transposed convolution
        x_padded = nn.functional.pad(x, [pad_w, pad_w, pad_h, pad_h])
        output = self.conv_transpose(x_padded)
        
        # Calculate the necessary cropping
        crop_h = output.size(2) - output_size[0]
        crop_w = output.size(3) - output_size[1]
        
        # Crop the output if necessary
        if crop_h > 0 or crop_w > 0:
            output = output[:, :, :output.size(2)-crop_h, :output.size(3)-crop_w]

        return output
    

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()

        # Define the VGG layers using PyTorch's nn.Conv2d and nn.MaxPool2d
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding='same')

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding='same')

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding='same')

        # Fully convolutional layers converted to conv2d with a kernel size of 1
        self.fcn1 = nn.Conv2d(512, 4096, kernel_size=1)
        self.fcn2 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.fcn3 = nn.Conv2d(4096, 1000, kernel_size=1)

        # Deconvolution layers
        self.deconv2_1 = TransConv2d(256, 256, kernel_size=4, stride=2)
        self.deconv3_1 = TransConv2d(1000, 256, kernel_size=4, stride=2)

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        pool1 = maxpool2d_same(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2_1(pool1))
        x = F.relu(self.conv2_2(x))
        pool2 = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3_1(pool2))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        pool3 = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv4_1(pool3))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        pool4 = maxpool2d_same(x, kernel_size=1, stride=1)

        x = F.relu(self.conv5_1(pool4))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        pool5 = maxpool2d_same(x, kernel_size=1, stride=1)

        x = F.relu(self.fcn1(pool5))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fcn2(x))
        x = F.dropout(x, p=0.5)
        fcn3 = F.relu(self.fcn3(x))
        x = F.dropout(fcn3, p=0.5)


        deconv2_1 = F.relu(self.deconv2_1(pool3))
        deconv3_1 = F.relu(self.deconv3_1(fcn3))

        return deconv2_1, deconv3_1


class A_Net(nn.Module):
    def __init__(self):
        super(A_Net, self).__init__()
        # Define the layers
        self.conv2_1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding='same')
        # self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        
        self.conv2_3 = nn.Conv2d(160, 64, kernel_size=5, stride=1, padding=2)
        self.conv2_4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv2_5 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv2_6 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        
        self.deconv2_2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        # PReLU layers with learnable parameters
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu2_3 = nn.PReLU()
        self.prelu2_4 = nn.PReLU()
        self.prelu2_5 = nn.PReLU()
        self.prelu2_6 = nn.PReLU()
        self.prelu_deconv2_2 = nn.PReLU()

        # Add more layers as needed

    def forward(self, x, G_input):
        # Forward pass
        conv2_1_out = self.prelu2_1(self.conv2_1(x))
        conv2_1_pool = F.max_pool2d(conv2_1_out, kernel_size=2, stride=2)

        conv2_2_out = self.prelu2_2(self.conv2_2(G_input))
        
        conv_concat = torch.cat((conv2_1_pool, conv2_2_out), 1)

        conv2_3_out = self.prelu2_3(self.conv2_3(conv_concat))
        conv2_4_out = self.prelu2_4(self.conv2_4(conv2_3_out))
        conv2_5_out = self.prelu2_5(self.conv2_5(conv2_4_out))
        conv2_6_out = self.prelu2_6(self.conv2_6(conv2_5_out))
        
        deconv2_2_out = self.prelu_deconv2_2(self.deconv2_2(conv2_6_out))

        return deconv2_2_out


class S_Net(nn.Module):
    def __init__(self):
        super(S_Net, self).__init__()
        # Define the layers here
        self.conv3_1 = nn.Conv2d(3, 96, kernel_size=9, stride=1, padding=4)
        # Assuming that G_input comes from another part of the network with 256 channels
        self.conv3_2 = nn.Conv2d(256, 64, kernel_size=1, stride=1)

        # Concatenation will happen in the forward method
        self.conv3_3 = nn.Conv2d(160, 64, kernel_size=5, stride=1, padding=2)
        self.conv3_4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv3_5 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv3_6 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

        # Define the deconvolution layer, assuming batch_size is known ahead of time
        self.deconv3_2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x, G_input):
        conv3_1_out = F.relu(self.conv3_1(x))
        conv3_1_pool = F.max_pool2d(conv3_1_out, kernel_size=3, stride=2, padding=1)

        conv3_2_out = F.relu(self.conv3_2(G_input))
        # Concatenate conv3_1_pool and conv3_2_out along the channel dimension
        conv_concat = torch.cat((conv3_1_pool, conv3_2_out), dim=1)

        conv3_3_out = F.relu(self.conv3_3(conv_concat))
        conv3_4_out = F.relu(self.conv3_4(conv3_3_out))
        conv3_5_out = F.relu(self.conv3_5(conv3_4_out))
        conv3_6_out = F.relu(self.conv3_6(conv3_5_out))

        deconv3_2_out = F.relu(self.deconv3_2(conv3_6_out))

        return deconv3_2_out
    

class ShadowMatte(nn.Module):
    def __init__(self):
        super(ShadowMatte, self).__init__()
        # Define the 1x1 convolution layer
        # Assuming the concatenated channel size is 6 (A_input channels + S_input channels)
        self.conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, stride=1)

    def forward(self, A_input, S_input):
        # Concatenate A_input and S_input along the channel dimension
        x = torch.cat([A_input, S_input], dim=1)
        # Pass the concatenated tensor through the 1x1 convolution layer
        x = self.conv(x)
        # Optionally clip the values to the range [0, 1]
        x = torch.clamp(x, min=0, max=1)
        return x
    

class DeShadowNet(nn.Module):
    def __init__(self):
        super(DeShadowNet, self).__init__()
        self.model_G = G_Net()
        self.model_A = A_Net()
        self.model_S = S_Net()
        self.matte = ShadowMatte()

    def forward(self, x):
        x_A, x_S = self.model_G(x)

        res_A = self.model_A(x, x_A)
        res_S = self.model_S(x, x_S)

        x = self.matte(res_A, res_S)
        return x
    

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = DeShadowNet().cuda()
    res = model(t)
    print(res.shape)

    

'''
convert the following tensorflow code to pytorch:

def conv_layer(x, filtershape, stride, name):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
            trainable = True)
        conv = tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding= 'SAME')
        conv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
                                trainable=True, name ='bias')
        bias = tf.nn.bias_add(conv, conv_biases)
        output = prelu(bias)
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        tf.summary.image('conv_filter',img_filt)
        return output

def deconv_layer(x, filtershape,output_shape, stride, name):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
            trainable = True)
        deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding ='SAME')
        #deconv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
        #                        trainable=True, name ='bias')
        #bias = tf.nn.bias_add(deconv, deconv_biases)
        #output = prelu(bias)
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        tf.summary.image('deconv_filter',img_filt)
        return prelu(deconv)

def max_pool_layer(x,filtershape,stride,name):
    return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding ='SAME',name = name)

def A_Net(self,x,G_input,keep_prob): # after conv3 in G_Net  256

        print('making a-network')
        sess=tf.Session()
        with tf.variable_scope('A_Net'):

            # conv2-1
            conv2_1 = conv_layer(x,[9,9,3,96],1,'conv2-1')

            # pool5
            conv2_1_output = max_pool_layer(conv2_1,[1,3,3,1],2,'pool2-1')
            print('conv2-1')
            print(sess.run(tf.shape(conv2_1_output)))

            # conv2-2
            conv2_2_output = conv_layer(G_input,[1,1,256,64],1,'conv2-2') 
            print('conv2-2')
            print(sess.run(tf.shape(conv2_2_output)))

            # concat conv2-1 and conv2-2
            conv_concat = tf.concat(axis=3, values = [conv2_1_output,conv2_2_output], name = 'concat_a_net')

            # conv2-3
            conv2_3 = conv_layer(conv_concat,[5,5,160,64],1,'conv2-3')
            print('conv2-3')
            print(sess.run(tf.shape(conv2_3)))

            # conv2-4
            conv2_4 = conv_layer(conv2_3,[5,5,64,64],1,'conv2-4')
            print('conv2-4')
            print(sess.run(tf.shape(conv2_4)))

            # conv2-5
            conv2_5 = conv_layer(conv2_4,[5,5,64,64],1,'conv2-5')
            print('conv2-5')
            print(sess.run(tf.shape(conv2_5)))

            # conv2-6
            conv2_6 = conv_layer(conv2_5,[5,5,64,64],1,'conv2-6')
            print('conv2-6')
            print(sess.run(tf.shape(conv2_6)))

            # deconv2_1
            deconv2_2 = deconv_layer(conv2_6,[4,4,3,64],[self.batch_size,224,224,3],2,'deconv2-2')
            print('deconv2-2')
            print(sess.run(tf.shape(deconv2_2)))
 
            print('finishing a-network')
            
            tf.summary.image('conv2_1',conv2_1_output[:,:,:,0:3])
            tf.summary.image('conv2_2',conv2_2_output[:,:,:,0:3])
            tf.summary.image('conv2_3',conv2_3[:,:,:,0:3])
            tf.summary.image('conv2_4',conv2_4[:,:,:,0:3])
            tf.summary.image('conv2_5',conv2_5[:,:,:,0:3])
            tf.summary.image('conv2_6',conv2_6[:,:,:,0:3])
            red = tf.reshape(deconv2_2[:,:,:,0], [-1,224,224,1])
            green = tf.reshape(deconv2_2[:,:,:,1], [-1,224,224,1])
            blue = tf.reshape(deconv2_2[:,:,:,2], [-1,224,224,1])
            tf.summary.image('deconv3_1',red)
            tf.summary.image('deconv3_1',green)
            tf.summary.image('deconv3_1',blue)
            #tf.summary.image('deconv2_2-1',deconv2_2[:,:,:,0:1])
            #tf.summary.image('deconv2_2-2',deconv2_2[:,:,:,1:2])
            #tf.summary.image('deconv2_2',deconv2_2[:,:,:,:])
            sess.close()
            return deconv2_2

'''