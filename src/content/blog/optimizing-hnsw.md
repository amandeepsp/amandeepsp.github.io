---
title: "Optimizing HNSW: a Worklog"
subTitle: "Bottlenecks are a real pain in the neck!"
publishDate: "Feb 28 2026"
tags: [ml, algorithms, zig]
draft: true
toc: true
featured: true
---

>You can have a second computer once you’ve
shown you know how to use the first one. [^1]

In my [last post](/blog/hnsw), I implemented [Heirachical Navigable Small Worlds (HNSW)]() algorithm in Zig. While the performance was adequate on [GloVe]() dataset, we can do better. This posts is a worklog I will iterately improve the query and build time performance. The goal is not to build fully production ready implementation but to gain deep understanding of techinques used to optimize on CPUs.

A quick intro about HSNW.

First thing we need to do is to profile the run, currently the code includes build and query phases in a single go. For now I will run this as is since we will be doing a coarse pass to see hotspots where we are spending a lot of CPU time. I am on Linux so will be using [`perf`](). I will not be covering `perf` commands here, there enough info on the web about this, or just ask your friendly neighborhood LLM. Getting a list of functions my time spent on CPU we get this

```sh
# Overhead  Command  Shared Object         Symbol
# ........  .......  ....................  ...............................
#
    59.89%  hnsw     hnsw                  [.] distance.normCosine
    26.60%  hnsw     hnsw                  [.] hnsw.HnswIndex.searchLayer
     8.26%  hnsw     hnsw                  [.] benchmark.runBenchmark
     1.33%  hnsw     libc.so.6             [.] 0x000000000018618e
```


## SIMD Distance calculations


## Flat Layer 0
Currently our node access pattern is this

```zig
const Node = struct {
    neighbors: []std.ArrayListUnmanaged(NodeIdx),
};

const HnswIndex = struct {
    nodes: std.ArrayListUnmanaged(Node)
};

// To get the neighbors at a layer
self.nodes.items[idx].neighbors[layer].items;
```
Count the number of pointer chases we had to do. **3**, (how?) the chances of cache misses

[^1]: Attributed to Paul Barham in [Scalability! But at what COST?](https://www.usenix.org/system/files/conference/hotos15/hotos15-paper-mcsherry.pdf) paper.
