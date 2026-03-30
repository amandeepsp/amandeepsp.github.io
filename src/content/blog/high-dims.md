---
title: "The Boon of Dimensionality"
subTitle: "or: Why High Dimensions work for ML"
publishDate: "Mar 3 2026"
updatedDate: "Mar 30 2026"
tags: [ml]
toc: true
featured: false
seo:
  description: "Volume, equators, and near-orthogonality in high dimensions, and why embeddings and random projections work."
---

I recently watched Grant Sanderson's (3blue1brown) [video](https://youtu.be/fsLh-NYhOoU) about volume of high-dimensional spheres.
He made a note that the high-dimensional space is also of peculiar interest to the ML field. I knew of the [*curse of dimensionality*](https://en.wikipedia.org/wiki/Curse_of_dimensionality), but I wanted to dig deeper and here is what I found, the other side; the *boon of dimensionality*.

I would recommend you watch [Grant's video](https://youtu.be/fsLh-NYhOoU) for this. I will try my best here to explain the intuition behind the presented results.

## Volume of a high-dimensional ball

The result that Grant's video centers on is that the volume of a ball with radius $r$ in $d$ dimensions is:

$$ V_d = \cfrac{\pi^{d/2}}{\Gamma(d/2 + 1)}{r^d} $$

Where $\Gamma(x)$ is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function). Plotting $V_d$ for $d$ from $1$ to $50$ we get something like this.

![Volume of a unit ball in d dimensions](/blog/high-dims/volume.svg)

You can see the volume peaks at around $d=5$, roughly 5.26 and at $d=50$ becomes vanishingly small, since the Gamma function denominator grows much faster than the numerator. Even more strange is how this volume is distributed. Consider a ball of radius $(1-\epsilon)$ where epsilon is an infinitesimal (read: vanishingly small value). The ratio of its volume to the full ball is

$$  \cfrac{V(1-\epsilon)}{V(1)} = {(1-\epsilon)}^d \leq {e}^{-\epsilon d}$$

Since $(1-x) \leq e^{-x}$ from Taylor expansion of $ e^{-x} $. We can see that as $ d \rightarrow \infty $, the ratio $ \rightarrow 0 $. This tells us that most of the already small volume is concentrated near its surface. This is a special case of a general principle called **[concentration of measure](https://en.wikipedia.org/wiki/Concentration_of_measure)**, the tendency for high-dimensional probability distributions to concentrate their mass in thin regions.

## Equators

Yet another interesting thing is about the equators, but first what is the equator of a high-dimensional ball. we can pick a coordinate say $x_1$ and all the points with $ -1 \leq x_1 < 0 $ lie in one hemisphere and $ 0 < x_1 \leq 1$ in another; the $ x_1 = 0 $ boundary will be the equator. The other such slices will be $\sqrt{1 - {x_1}^2}$ (Pythagoras) and volume of such a slice will be $\propto {(1-{x_1}^2)}^{(d-1)/2} $, since we have fixed one dimension and are left with $d-1$ dimensions. This is again bounded by

$$ {(1-{x_1}^2)}^{(d-1)/2} \leq e^{-(d-1){x_1}^{2}/2} $$

This shows that the slice volume *decreases exponentially* as we move
away from $0$, which means most of the volume is also concentrated near $x_1 = 0$. From the shell result and this we can see most of the points are concentrated on the equator.

## Near Orthogonality

The value $x_1$ is just $\langle x, e_1 \rangle$, where $e_1$ is the unit vector across the chosen axis. If most points lie near the equator (small $x_1$), they are nearly orthogonal to $e_1$. Since balls are rotationally symmetric, this can be said about any direction. So pick one point, make it the north pole and we are *very certain* the next point we pick will lie on the equator.

![Pairwise dot products of random unit vectors](/blog/high-dims/orthogonality.svg)

## The Johnson-Lindenstrauss Lemma

The near-orthogonality result has a famous companion that makes the
capacity claim precise. In 1984, Johnson and Lindenstrauss proved the
following [^jl]:

> Take any $n$ points in a high-dimensional space. You can project them
> into a space of just $O(\epsilon^{-2} \log n)$ dimensions and preserve all pairwise
> distances up to a small factor $(1 \pm \epsilon)$.

While this is originally stated as a compression result, we can read it backwards to see that a $d$ dimensional space has the capacity to represent $e^{\Omega(d)}$ points with geometry intact. Note that this is exponential in $d$.

This is not just a theoretical curiosity. Embedding models like word2vec [^w2v] or sentence transformers [^sbert] pack millions of concepts into ~768 dimensions. The JL capacity result tells us why this works: even after accounting for the constant hidden in the $\Omega$, a 768-dimensional space has room for a huge number of near-orthogonal directions, far more than any vocabulary needs. Unrelated words get mapped to nearly orthogonal vectors and don't interfere with each other, which is why cosine similarity between embeddings tracks semantic similarity so well. The famous analogy arithmetic (king - man + woman ≈ queen) suggests that directions in this space can be meaningful, though in practice this only works for cherry-picked examples [^analogy]. What matters more is the geometric separation. Random projections exploit this directly too: you can project high-dimensional data down to $O(\log n)$ dims via a random matrix and preserve distances, which is the basis of [locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) and streaming sketches.

## Where next

The curse of dimensionality tells us that data gets sparse and distances lose meaning as dimensions grow. The boon is the flip side: exponential capacity, near-orthogonality for free, and distances that survive projection. Same geometry, two readings. These two operate at different scales though: the curse applies to the *intrinsic* dimensionality of the data (you still need samples proportional to the manifold dimension), while the boon applies to the *ambient* dimensionality (the room you have to embed things in). ML lives in the gap since real data is high-dimensional but structured (it sits on a manifold), and the boon wins over the curse.

There are two directions I want to explore from here. First is [Cover's theorem](https://en.wikipedia.org/wiki/Cover%27s_theorem) (1965), which says data that is not linearly separable in low dimensions becomes separable when mapped to higher dimensions. This is why kernel methods and neural network hidden layers work, they buy room. Second is [superposition](https://transformer-circuits.pub/2022/toy_model/index.html) in neural networks, the idea that a network with $m$ neurons can represent far more than $m$ features by packing them into near-orthogonal directions, which connects directly to the geometry above. But those are for another post.

[^jl]: W. B. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space," *Contemporary Mathematics*, 26, 1984.
[^w2v]: T. Mikolov et al., "Efficient Estimation of Word Representations in Vector Space," [arXiv:1301.3781](https://arxiv.org/abs/1301.3781), 2013.
[^sbert]: N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," [arXiv:1908.10084](https://arxiv.org/abs/1908.10084), 2019.
[^analogy]: O. Levy and Y. Goldberg, "Linguistic Regularities in Sparse and Explicit Word Representations," [CoNLL 2014](https://aclanthology.org/W14-1618/). See also Nissim et al., "Fair is Better than Sensational," [arXiv:1905.09866](https://arxiv.org/abs/1905.09866), 2019.
