---
layout: post
title:  "Make your models smaller! (Part 2)"
date:   2020-02-20 10:55:06 +0530
categories: [ML, On-device]
excerpt_separator: <!--more-->
permalink: /ml-model-compression-part2/
---

This post is direct continuation of [Part 1]({% post_url 2020-01-01-make-models-smaller %}), please try to go through it before proceeding. In this post i will be going through Low rank transforms, efficient network architechtures and knowledge distillation. Low rank transforms techniques decompose a convolution filter to lower rank parts decreasing the overall computational and storage complexity. Knowledge distillation or student-teacher models use techniques in which a larger model trains a smaller model. The smaller model inherits the *'knowledge'* of the larger model. <!--more-->

## Low Rank Transforms

Low rank transform means reprsenting a matrix or tensor as a product of some lower rank components. These components often only approximate the original matrix, but benefit hugely in space and computational efficiency. For example, fully connected layers can be compressed using plain old [Truncated SVD][tsvd] as a fully connected layer can be represented as a matrix. In truncated SVD matrix $M$ of size ${n \times m}$ is approximated by $ \tilde{M} = U \Sigma V^T$, where $U$ is $n \times t$, $\Sigma$ is a diagonal matrix of size $t \times t$ and $V$ is $t \times m$ in size. A fully connected layer can be represented as $Wx +b$, where $W$ is the weight matrix and $b$ are the biases. We now reprsent the FC layer as
\$$ (U\Sigma V^Tx) + b = U(\Sigma V^Tx) + b \$$ 
hence we can split our FC layer into two; 
- First layer with shape $n \times t$, having no biases and weights taken from $\Sigma V^T$.
- Second layer with shape $t \times m$, original biases and weights from $U$.
This drops the number of weights from $n \times m$ to $ t(n+m) $. Time complexity is also reduced the same factor. 

This can be easily implemented as in *PyTorch* using `torch.svd` method as depicted in the code snippet below. Here `vgg16` is a pre-trained model picked from `torchvision.models`. I have applied SVD on Linear layers after training. It can also be applied before training, but that involves calculating gradient of the SVD operation which is a hassle.

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
This results in size reduction from `528MB` to `195M` i.e. **~ 2.7x decrease**. <!--TODO: Talk about accuracy decrease--> This so well works because majority of the weights in a VGG16 are in Fully Connected layers. For many newer network e.g. ResNets majority of the weights lie in the Conv layers, therefore it makes more sense to apply Low rank transforms to Conv layers. Since conv layers are 4D tensors i.e `(batch, channels, width, height)`, SVD and its cousins will not work here. We need to apply specialized tensor decomposition techniques such as CP decomposition ([*Lebedev et.al.*][leb] in 2015) and Tucker Decomposition ([*Kim et. al.*][kim] in 2016). Not covering these papers in more detail because these techniques are now superceded by efficient architechtures like SqueezeNet and MobileNet which are discussed in the next section.

## Efficient network architectures
Rather than applying size reducing techniques to existing architechtures, we try to create novel architechtures that decrease model size and try to preserve the accuracy of the network over the time there have been many such architechtures, prominent of them being SqueezeNet, MobileNet V1 and MobileNet V2.
### SqueezeNet
SqueezeNet by [*Iandola et.al. 2016*][iandola] is presumably the first to explore a new architechure for smaller CNNS. At the core of SqueezeNet are *Fire Modules*

[tsvd]: https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD
[leb]: https://arxiv.org/pdf/1412.6553.pdf
[kim]: https://arxiv.org/pdf/1511.06530.pdf
[jacob]: https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
[iandola]: https://arxiv.org/pdf/1602.07360.pdf
