---
layout: post
title:  "Make your models smaller!"
date:   2019-12-26 10:55:06 +0530
categories: ml
excerpt_separator: <!--more-->
permalink: /ml-model-compression/
---

Machine Learning models are getting bigger and expensive to compute. Embedded devices have restricted memory, computation power and battery. But we can optimize our model to run smoothly on these devices. By reducing the size of the model we decrease the number of operations that need to be done hence reducing the computation. Smaller model also trivially translates into less memory usage. Smaller models are also more power efficient. One must think that reduced number of computations is responsible for less power consumption, but on the contrary the power draw from a memory access is about *1000x* more costly than an addition or a multiplication. Now since, there are no free lunches i.e. everything comes at a cost, we loose some accuracy of our models here. Bear in mind these speedups are not for training but for inference only.

<!--more-->

## Pruning

Pruning is remove excess network connections that does not hugely contribute to the output. Ideas of pruning networks are very old dating back to 1990s namely Optimal Brain Damage[^obd] and Optimal Brain Surgeon[^obs]. 

[^obd]: Optimal Brain Damage [https://papers.nips.cc/paper/250-optimal-brain-damage.pdf]
[^obs]: Optimal Brain Surgeon [https://papers.nips.cc/paper/749-optimal-brain-surgeon-extensions-and-performance-comparisons.pdf]