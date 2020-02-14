---
layout: post
title:  "Make your models smaller! (Part 1)"
date:   2019-12-26 10:55:06 +0530
categories: ml
excerpt_separator: <!--more-->
permalink: /ml-model-compression/
---

Machine Learning models are getting bigger and expensive to compute. Embedded devices have restricted memory, computation power and battery. But we can optimize our model to run smoothly on these devices. By reducing the size of the model we decrease the number of operations that need to be done hence reducing the computation. Smaller model also trivially translates into less memory usage. Smaller models are also more power efficient. One must think that reduced number of computations is responsible for less power consumption, but on the contrary the power draw from a memory access is about *1000x* more costly than an addition or a multiplication.
<!--more--> Now since, there are no free lunches i.e. everything comes at a cost, we loose some accuracy of our models here. Bear in mind these speedups are not for training but for inference only.

## Pruning
<!--proof-read-->
Pruning is remove excess network connections that does not hugely contribute to the output. Ideas of pruning networks are very old dating back to 1990s namely [*Optimal Brain Damage*][obd] and [*Optimal Brain Surgeon*][obs]. These methods use Hessians to determine the importance of connections, which also makes them impractical to use with deep networks. Pruning methods use an interative training technique i.e. Train - Prune - Fine-tune. Fine-tuning after pruning restores the accuracy of the network lost after pruning. 
One method is to rank the weights in the network using the L1/L2 norm and remove the last x% of them. Other types of methods which also use ranking use the mean acivation of neurons, the number of times a neuron's activation is zero on a validation set and many other creative methods. This approch is pioneered by [Han et.al.][han] in thier 2015 paper.

![pruning_image]

Even more recently in 2019, the [Frankle et.al.][frankle] paper titled *The Lottery Ticket Hypothesis* the authors found out that within every deep neural network there exists a subset of it which gives the same accuracy for equal amount of training. These results hold true for unstructured pruning which prunes the whole network with gives us a sparse network. Sparse networks are inefficient on GPUs since there is no structure to their computation. To remedy this, structured pruning is done, which prunes a part of the network e.g. a layer or a channel. The Lottery Ticket discussed earlier is found no to work here by [Liu et.al.][liu] They instead discovered that it was better to retrain a network after pruning instead of fine-tuning.
Aside from performace is there any other use of sparse networks? Yes, sparse networks are more robust to noise input as shown in a paper by [Ahmed et.al.][ahmed] Pruning is supported in both TF (`tensorflow_model_optimization` package) and PyTorch (`torch.nn.utils.prune`).

To use pruning in PyTorch you can either select a technique class from `torch.nn.utils.prune` or implement your own subclass of `BasePruningMethod`.

~~~ python
from torch.nn.utils import prune
tensor = torch.rand(2, 5)
pruner = prune.L1Unstructured(amount=0.7)
pruned_tensor = pruner.prune(tensor)
~~~

To prune a module we can use pruning methods (basically wrappers on the classes discussed above) given in `torch.nn.utils.prune` and specify which module you want to prune, or even which parameter within that module.

~~~ python
conv_1 = nn.Conv(3, 1, 2)
prune.ln_structured(module=conv_1, name='weight', amount=5, n=2, dim=1)
~~~

This replaces the parameter `weight` with the pruned result and adds a parameter `weight_orig` that stores the upruned version of the input. The pruning mask is stored as `weight_mask` and saved as a module buffer. These can be checked by the `module.named_parameters()` and `module.named_buffers()`. To enable iterative pruning we can use just apply the pruning method for the next iteration and it just works, due to `PruningContainer` as it handles computation of final mask taking into account previous prunings using the `compute_mask` method.

[obd]: https://papers.nips.cc/paper/250-optimal-brain-damage.pdf
[obs]: https://papers.nips.cc/paper/749-optimal-brain-surgeon-extensions-and-performance-comparisons.pdf
[han]: https://arxiv.org/abs/1506.02626
[frankle]: https://arxiv.org/abs/1803.03635
[liu]: https://arxiv.org/abs/1810.05270
[ahmed]: https://arxiv.org/abs/1903.11257
[pruning_image]: https://miro.medium.com/max/1934/1*4dJE_vHfGpPBtXLLXLmnBQ.png