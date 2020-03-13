---
layout: post
title:  "Make your models smaller! (Part 2)"
description: "Create lean and mean models :zap: :muscle:"
date:   2020-02-20 10:55:06 +0530
categories: [ML, On-device]
excerpt_separator: <!--more-->
permalink: /ml-model-compression-part2/
---

This post is a direct continuation of [Part 1]({% post_url 2020-01-01-make-models-smaller %}), please try to go through it before proceeding. In this post, I will be going through Low rank transforms, efficient network architectures and knowledge distillation. Low rank transforms techniques decompose a convolution filter to lower rank parts decreasing the overall computational and storage complexity. Knowledge distillation or student-teacher models use techniques in which a larger model trains a smaller model. The smaller model inherits the *'knowledge'* of the larger model. <!--more-->

## Low Rank Transforms

Low-rank transform means representing a matrix or tensor as a product of some lower rank components. These components often only approximate the original matrix but benefit hugely in space and computational efficiency. For example, fully connected layers can be compressed using plain old [Truncated SVD][tsvd] as a fully connected layer can be represented as a matrix. In truncated SVD matrix $M$ of size ${n \times m}$ is approximated by $ \tilde{M} = U \Sigma V^T$, where $U$ is $n \times t$, $\Sigma$ is a diagonal matrix of size $t \times t$ and $V$ is $t \times m$ in size. A fully connected layer can be represented as $Wx +b$, where $W$ is the weight matrix and $b$ are the biases. We now represent the FC layer as
\$$ (U\Sigma V^Tx) + b = U(\Sigma V^Tx) + b \$$ 
hence we can split our FC layer into two; 
- The first layer with shape $n \times t$, having no biases and weights taken from $\Sigma V^T$.
- Second layer with shape $t \times m$, original biases and weights from $U$.
This drops the number of weights from $n \times m$ to $ t(n+m) $. Time complexity is also reduced by the same factor. 

This can be easily implemented as in *PyTorch* using the `torch.svd` method as depicted in the code snippet below. Here `vgg16` is a pre-trained model picked from `torchvision.models`. I have applied SVD on Linear layers after training. It can also be applied before training, but that involves calculating the gradient of the SVD operation which is a hassle.

~~~python
svd_classifier_layers = []
L = 50
for layer in vgg16.classifier:
  if isinstance(layer, nn.Linear):
    W = layer.weight.data
    U, S, V = torch.svd(W)
    W1 = U[:,:L]
    W2 = torch.diag(S[:L]) @ V[:,:L].t()
    layer_1 = nn.Linear(in_features=layer.in_features, 
                        out_features=L, bias=False)
    layer_1.weight.data = W2
    svd_classifier_layers.append(layer_1)

    layer_2 = nn.Linear(in_features=L, 
                        out_features=layer.out_features, bias=True)
    layer_2.weight.data = W1
    layer_2.bias.data = layer.bias.data
    svd_classifier_layers.append(layer_2)
  else:
    svd_classifier_layers.append(layer)

svd_vgg16.classifier = nn.Sequential(*svd_classifier_layers)
~~~
This results in size reduction from `528MB` to `195M` i.e. **~ 2.7x decrease**. <!--TODO: Talk about accuracy decrease--> This works so well works because the majority of the weights in a VGG16 are in Fully Connected layers. For much newer network e.g. ResNets majority of the weights lie in the Conv layers, therefore it makes more sense to apply Low rank transforms to Conv layers. Since conv layers are 4D tensors i.e `(batch, channels, width, height)`, SVD and its cousins will not work here. We need to apply specialized tensor decomposition techniques such as CP decomposition ([*Lebedev et.al.*][leb] in 2015) and Tucker Decomposition ([*Kim et. al.*][kim] in 2016). Not covering these papers in more detail because these techniques are now superseded by efficient architectures like SqueezeNet and MobileNet which are discussed in the next section.

## Efficient network architectures
Rather than applying size reducing techniques to existing architectures, we try to create novel architectures that decrease the model size and try to preserve the accuracy of the network over the time there have been many such architectures, prominent of them being SqueezeNet, MobileNet V1 and MobileNet V2.
### SqueezeNet
SqueezeNet by [*Iandola et.al.*][iandola] is presumably the first to explore a new architecture for smaller CNNs. At the core of SqueezeNet are **Fire Modules**. Fire modules use `1x1` filters rather than `3x3` filters as they have 9x lesser parameters and have a lesser number of channels than normal, which is called a *squeeze* layer. The lesser number of channels are recovered in the expand layer which consists of several zero-padded `1x1` filters and `3x3` filters. The number of filters in the squeeze layers and expand layers are hyper-parameters. If $e_{3 \times 3} + e_{1 \times 1}$ are the number of filters in expand layer and $s_{1 \times 1}$ is the number of filters in the squeeze layer. When using Fire module $s_{1 \times 1} < e_{3 \times 3} + e_{1 \times 1}$ works best.

{% 
    include image.html 
    file="/assets/fire_module.png" 
    caption="Fig 1. Fire module with $s_{1 \times 1} = 3$, $e_{1 \times 1} = 4$ and $e_{3 \times 3} = 4$."
    source="https://arxiv.org/pdf/1602.07360.pdf"
%}

Code for the Fire Module adapted from `torchvision.models`. Here `inchannels` are the number of input channels, `squeeze_planes` are the number of output channels, `expand1x1_planes` and `expand3x3_planes` are the output channel number for the expand layer. They are generally same.
~~~python
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
~~~
SqueezeNet also uses delayed-sampling to create larger activation maps towards the *end* layers, which in turn leads to greater accuracy. The full architecture can be visualized [*here*][squeezenet].

### MobileNets
MobileNets are specifically developed by Google to specifically run on mobile devices. MobileNets were first introduced in a paper by [*Howard et.al.*][mobv1] in 2017, subsequently, in 2018 an improved version was introduced called MobileNet v2 in [*Sandler et. al.*][mobv2]. The gist of optimization in MobileNet v1 lies in a special kind of convolution layer called **Depthwise separable convolutions**. For a simple convolution layer if $k$ is the dimension of the kernel, $N_k$ is the number of kernels, and the input is of size $ N_c \times W \times H$, where $N_c$ are the number of input channels. The total number of parameters and computations are $k^2N_kN_cWH$. MobileNet Convolutions work in two stages
1. Convolve a $k \times k$ for each channel of the input and stack $N_c$ of them, creating an output tensor of size $N_c \times W \times H$. Total number of ops in this layer is $k^2N_cWH$
2. Convolve with a $1 \times 1$ filter with $N_k$ channels to create the final output. Total number of computations in this stage is $N_cN_kWH$

Total computations in a MobileNet convulation are $k^2N_cWH + N_cN_kWH$. There total reduction in parameters in given by,
\$$\frac{k^2N_cWH + N_cN_kWH}{k^2N_kN_cWH} = \frac{1}{N_k} + \frac{1}{k^2}\$$
For $k = 3$, $N_k = 16$ we have a **~ 5.76x** reduction in number of parameters for a layer.

{% 
    include image.html 
    file="/assets/depthwise.svg" 
    caption="Fig 2. Depthwise seperable convolution followed by pointwise convolution"
    max-width="400"
    source="https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/"
%}

Implementing Depthwise conv. is quite simple. Checkout the code snippet below, `inp` donates the number of input channels and `oup` are the number of output channels.
~~~python
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )
~~~
**MobileNet v2** uses as an inverted residual block as its main convolutional layer. A Residual block taken from the [*ResNets*][resnet] includes bottleneck layers that decrease the number of channels followed by an expansion layer that restores the number of channels for the residual concat operation. The inverted block layer does the reverse of that it first expands the number of channels then reduce them. The last layer in the block is a bottleneck layer as it decreases the channels of the output. This layer has to non-linearity attached to it. This because authors found out that a linear bottleneck does not lose information when a feature-map is embedded into a lower dimension space i.e. reduced to a tensor with less number of channels. This is found to increase the accuracy of these networks. To calculate the number of parameters, presume $N_{in}$ is the number of input channels, $N_{out}$ the number of output channels and $t$ is the expansion ratio, the ratio between the size of the intermediate layer to the input layer. The number of computations and parameters are $WHN_{in}t(N_{in} + k^2 + N_{out})$. But there is an extra `1x1` convolution component, still, we have a computational advantage because due to the nature of the block we can now decrease the input and output dimensions e.g. a layer with dimensions `112x112` can have only `16` channels and retaining the accuracy as compared to `64` for MobileNet v1.

{% 
    include image.html 
    file="/assets/InvResidualBlock.png" 
    caption="Fig 3. MobileNet v2 primary convolution block."
    source=" https://machinethink.net/blog/mobilenet-v2/"
%}

The code for the `InvertedResidual` block is adapted from `trochvision.models` package.
~~~python
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, 
                       groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
~~~

## Knowledge Distillation
**MobileNet v2** uses as an inverted residual block as its main convolutional layer. A Residual block taken from the [*ResNets*][resnet] includes bottleneck layers that decrease the number of channels followed by an expansion layer that restores the number of channels for the residual concat operation. The inverted block layer does the reverse of Knowledge Distillation (KD) is a model compression technique by which the behaviour of a smaller (student) model is trained to replicate the behaviour of a larger (teacher) model. The term was first coined by none other than Geoffrey Hinton in his [2015 paper][distill1]. KD involves training a smaller network on the weighted average of soft target output of the larger model and the ground truth. Soft target output can be obtained by calculating the softmax on the logits of the larger model, but this creates large divides between the probabilities of the correct label and the wrong label, thus not creating much information other than the ground truth. To remedy this problem Hinton introduces *softmax with temperature* given by
\$$q_i = \frac{exp(\frac{z_i}{T})}{\sum_j exp(\frac{z_j}{T})}\$$
where $T$ the temperature parameter, $T =1$ gives the same result as a simple softmax. AS $T$ grows the probabilities grow softer, providing more information about the model. The overall loss function of the now student-teacher pair becomes
\$$\mathcal{L} = \lambda \mathcal{L_{gt}} + (1-\lambda) \mathcal{L_{temp}}\$$
where $\mathcal{L_{gt}}$ is the loss with ground truth outputs and $\mathcal{L_{temp}}$ is the softmax temperature loss. Both $\lambda$ and $T$ are tunable hyperparameters. The loss configuration is as in the image below.

{% 
    include image.html 
    file="/assets/kd.png" 
    caption="Fig 4. Knowledge distillation model configuration."
    source=" https://nervanasystems.github.io/distiller/knowledge_distillation.html"
%}

A major success story of KD is [DistillBERT][distillbert]. [Hugging Face :hugs:][huggingface] managed to use KD to reduce the size of the BERT from 110M parameters to 66M parameters, while still retaining 97% of the performance of the original model. DistillBERT uses various additional tricks to achieve this such as using KD loss instead of standard cross-entropy to retain the probability distribution of the teacher model. The code to train a KD model will go like below. This code is adapted from DistilBERT training sequence itself.
~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

celoss = nn.CrossEntropyLoss
lambda_ = 0.5

def kd_step(teacher: nn.Module,
            student: nn.Module,
            temperature: float,
            inputs: torch.tensor,
            optimizer: Optimizer):
    teacher.eval()
    student.train()
    
    with torch.no_grad():
        logits_t = teacher(inputs=inputs)
    logits_s = student(inputs=inputs)
    
    loss_gt = celoss(input=F.log_softmax(logits_s/temperature, dim=-1),
                     target=labels) 
    loss_temp = celoss(input=F.log_softmax(logits_s/temperature, dim=-1), 
                       target=F.softmax(logits_t/temperature, dim=-1))
    loss = lambda_ * loss_gt + (1 - lambda_) * loss_temp
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
~~~


This concludes the series on ML model compression :raised_hands:. There are many more methods to make ML models smaller which I cannot cover as the posts would become too long. More and more research is being done on this, to follow the research be sure to check to [arixv-sanity](https://www.arxiv-sanity.com/). Will try to introduce a further reading section in future.

[tsvd]: https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD
[leb]: https://arxiv.org/pdf/1412.6553.pdf
[kim]: https://arxiv.org/pdf/1511.06530.pdf
[jacob]: https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
[iandola]: https://arxiv.org/pdf/1602.07360.pdf
[squeezenet]:https://dgschwend.github.io/netscope/#/preset/squeezenet
[mobv1]:https://arxiv.org/pdf/1704.04861.pdf
[mobv2]:https://arxiv.org/pdf/1801.04381.pdf
[depth]:https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/
[resnet]: https://arxiv.org/abs/1512.03385
[invres]: https://machinethink.net/blog/mobilenet-v2/
[distill1]: https://arxiv.org/pdf/1503.02531.pdf
[distillbert]:https://medium.com/huggingface/distilbert-8cf3380435b5
[huggingface]:https://huggingface.co/