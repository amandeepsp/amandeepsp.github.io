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

Convolutions are expensive opertains and they represent a bulk of computation and storage used up by a network. To convolve a 2D matrix $x \in \Bbb{R} ^ {H \times W}$ with $N$ number of filters each of size $d \times d$, producing a feature map $ y \in \Bbb{R} ^ {H_o \times W_o}$, we need $ \mathcal{O}(d^2NH_oW_o)$ number of operations. Since filters in CNNs have channels too. If $C$ are the the number of channels in a filter, the complexity of convolution rises to $ \mathcal{O}(d^2NCH_oW_o)$. Fully connected layers can be compressed using plain old [Truncated SVD][tsvd] as a fully connected layer can be represented as a matrix. In truncated SVD matrix $M$ of size ${n \times m}$ is approximated by $ \tilde{M} = U \Sigma V^T$, where $U$ is $n \times t$, $\Sigma$ is $t \times t$ and $V$ is $t \times m$ in size. A fully connected layer can be represented as $Wx +b$, where $W$ is the weight matrix and $b$ are the biases. We now reprsent the FC layer as
\$$ (U\Sigma V^Tx) + b = U(\Sigma V^Tx) + b \$$ 
hence we can split our FC layer into two; 
- First layer with shape $n \times t$, having no biases and weights taken from $\Sigma V^T$.
- Second layer with shape $t \times m$, original biases and weights from $U$.
This drops the number of weights from $n \times m$ to $ t(n+m) $. Time complexity is also reduced the same factor. 

All this is done after training the model. Since convolutional layers are 4D tensors, they use tensor decompositions to achieve the same effect. 
- **CP Decomposition**: A tensor is decomposed into vector pSpeeding up convolutional neural networks with low rank expansions.arts e.g. a 3D tensor $X$ is decomposed as
\$$ X \approx \sum_{r=1}^R u_r \circ v_r \circ w_r = \tilde{X}\$$
where $R > 0$ and $u_r, v_r, w_r$ are vectors, and the notaion $\circ$ donaties the outer product of tensors i.e.
\$$ x_{ijk} \approx \sum_{r=1}^R u_{ri} v_{rj} w_{rk} \$$
For $R = 1$ Optimizing using CP decomposition was first explored (as far as i can find) by [*Lebedev et.al.*][leb] in their 2014 paper.
![](https://www.alexejgossmann.com/images/CP_tensor_decomposition/rank-1_decomposition_cartoon.png)

## Efficient network architectures

### SqueezeNet

### MobileNet

## Knowledge distillation

[tsvd]: https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD
[leb]: https://arxiv.org/pdf/1412.6553.pdf
