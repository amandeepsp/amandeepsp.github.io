---
layout: post
title:  "Make your models smaller! (Part 2)"
date:   2020-02-20 10:55:06 +0530
categories: [ML]
excerpt_separator: <!--more-->
permalink: /ml-model-compression-part2/
---

This post is direct continuation of [Part 1]({% post_url 2020-01-01-make-models-smaller %}), please try to go through it before proceeding. In this post i will be going through Low rank transforms, efficient network architechtures and knowledge distillation. Low rank transforms techniques decompose a convolution filter to lower rank parts decreasing the overall computational and storage complexity. Knowledge distillation or student-teacher models use techniques in which a larger model trains a smaller model. The smaller model inherits the *'knowledge'* of the larger model. <!--more-->

## Low Rank Transforms

Convolutions are expensive opertains and they represent a bulk of computation and storage used up by a network. Convolution filters are 4D tensors and the fully connected layers can be expressed as a 2D matrix. Decomposing tensors into smaller low-rank parts is not a new idea, it has been around for a long time in Singal Processing Litrature, e.g. high dimentional DCT (Discrete Cosine Transforms) and wavelets are constructed using their 1D DCT and wavelets respectively.