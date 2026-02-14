---
title: Exploring Clojure for Advent of Code
publishDate: "Jan 05 2022"
toc: true
tags:
  - programming-languages
  - clojure
featured: false
seo:
  description: |
    An account of experience using Clojure for Advent of Code 2021. Goes through Threading Macros, Clojure's collections
    and Zipper functionality
---

Very early in my career I had the absolute delight of reading [SICP](https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/book.html)
; this was my first introduction Lispy languages. I used [Racket](https://racket-lang.org/)
to go through the book at the time, and found it overall enjoyable to use. Although, I found the all the parentheses to be
an acquired taste. Eventually could visually cut through the parentheses.

[Clojure](https://clojure.org/) was one other language I wanted to explore in this genre. Aside form being Lispy, some
of its selling points that intrigued me were that a) it is targeted to compiled to JVM bytecode b) had persistent data structures;
that can change with mutating the original memory (by creating new structures efficiently).

I gave it a go for [Advent of Code 2021](https://adventofcode.com/2021). Here are my solutions, [github/aoc2021](https://github.com/amandeepsp/aoc2021/tree/master).
My overall experience was again also good, here are some of the language features I found very fun to use.

## Threading Macros
These are a very convenient way to compose functions. I like the ergonomics of using them over the manual way of composing
`(c (b (a x)))` which is difficult to understand when going through the program. There are two of them in Clojure;
thread-first `->` and thread-last `->>` owing to the fact that the former threads into the first argument of each function
and the latter the last argument.

Here is an example for file parsing. I also appreciate, when languages have simple IO APIs when sometimes
10-20% of AoC problem code is just input parsing.

```clojure
(ns aoc.shared
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

(defn read-lines [file-resource]
  (->> (io/resource file-resource)
       (slurp)
       (str/split-lines)))
```

This was also my most overused feature when solving AoC since there is a huge correlation between composition and breaking
a problem into smaller pieces, for example in Day 14 [Problem](https://adventofcode.com/2021/day/14), we apply a function 11 times
and then calculate the difference of min frequency and max frequency of the characters of the resulting string.
[Complete Solution](https://github.com/amandeepsp/aoc2021/blob/master/src/aoc/day14.clj)

```clojure
;Part-1
(->> template
     (iterate apply-subs)
     (take 11)
     (last)
     (frequencies)
     (vals)
     (min-max-diff))
```

## Collections
Clojure collections are *persistent data structures*, when mutating them we get a new structure, that may or may not
share memory with the original. They provide $\approx\mathcal{O}(log_{32}N)$ operations. They are based on
Hash Array Mapped [from Phil Bagwell's 2001 paper Ideal Hash Trees](https://lampwww.epfl.ch/papers/idealhashtrees.pdf).[^1]. Here is
an implementation Dijkstra's shortest path algorithm for [Day 15](https://adventofcode.com/2021/day/15). Note the
`assoc` operator, this would be mutating key `node` in a map `cost` with `curr-dist`, but here you get the appearance
of a returning a new map with the changed value. [Complete Solution](https://github.com/amandeepsp/aoc2021/blob/master/src/aoc/day15.clj).

```clj
(ns aoc.day15
  (:require [clojure.data.priority-map :refer [priority-map]]))

(defn dijkstra [graph start-coords h w]
  (loop [q (priority-map start-coords 0)
         costs {}]
    (if (empty? q)
      costs
      (let [[node curr-dist] (peek q)
            dist (->> (valid-neighbors node h w)
                      (filter (complement costs))
                      (map #(vector % (+ (graph %) curr-dist)))
                      (into {}))]
        (recur
         (merge-with min (pop q) dist)
         (assoc costs node curr-dist))))))
```

## Zippers
Input for [Day 18](https://adventofcode.com/2021/day/18) are nested vectors, which are tree like. This problem also wants us to edit the tree structure.
Seems like a good place to use the Zipper APIs. A good deep dive for zippers is provided by this
[Ivan Grishaev post](https://grishaev.me/en/clojure-zippers/). But in short: a zipper is a data structure that represents not just a tree,
but also a cursor into that tree; a focused position plus all the context needed to rebuild the whole structure after changes.
In Clojure, (`clojure.zip/vector-zip tree`) turns a nested vector into such a navigable structure.
Each zipper location contains/supports:
 - the current node (`zip/node`),
 - information about its siblings and parent path, and
 - functions to move (`zip/down`, `zip/up`, `zip/left`, `zip/right`, `zip/next`, `zip/prev`) or edit in place (`zip/edit`, `zip/replace`).

Because all of this is purely functional, edits return new zipper locations instead of mutating the original tree.
You can freely descend, modify, and then climb back to the top with `zip/root` to recover the updated structure;
exactly what we need for repeated tree rewrites in the problem.

### Navigating the tree
Lets add a few helpers deal with moving through the nested vector and finding specific locations (like leaves or the root).
```clj
(ns aoc.day18
  (:require [clojure.zip :as z]
            [clojure.walk :as walk]))

(defn leaves-seq [loc step]
  (->> (iterate step loc)
       rest
       (take-while (complement z/end?))
       (remove z/branch?))) ;filter out branches

(defn next-leaves [number] (leaves-seq number z/next))
(defn prev-leaves [number] (leaves-seq number z/prev))

(defn root-loc [loc]
  (->> (iterate z/up loc)
       (take-while identity)
       last))
```

`leaves-seq` walks a zipper in depth-first order, collecting only the leaf nodes (the regular numbers, not pairs).
`next-leaves` and `prev-leaves` specialize this for scanning forward or backward.
Meanwhile, `root-loc` ensures we can always “rewind” to the top of the tree after edits since zipper edits keep you
at the modified node.

### Exploding deeply nested pairs

Exploding is the most complex operation: when a pair is nested inside four pairs, it “explodes”
its left and right values are distributed to the nearest regular numbers on the left and right,
and it’s replaced by `0`.

```clj
(defn explode [number]
  (if-let [explode-loc
           (->> number
                next-leaves
                (filter #(> (count (z/path %)) 4))
                first
                z/up)]
    (let [[left-val right-val] (z/node explode-loc)
          explode-loc (z/replace explode-loc 0)
          explode-loc (if-let [left-loc (first (prev-leaves explode-loc))]
                        (-> left-loc (z/edit + left-val) next-leaves first)
                        explode-loc)
          explode-loc (if-let [right-loc (first (next-leaves explode-loc))]
                        (-> right-loc (z/edit + right-val) prev-leaves first)
                        explode-loc)]
      [explode-loc :continue])
    [number :done]))
```


Here we locate the leftmost pair nested deeper than 4 levels `((> (count (z/path %)) 4))`.
Once found, we destructure its values, replace the pair with 0, and then use `prev-leaves` and `next-leaves`
to locate and increment the nearest neighboring numbers. Everything is done immutably,
`z/edit` returns a new zipper each time.

To run explosions repeatedly until stable, we wrap it in full-explode:

```clj
(defn full-explode [number]
  (->> [number :continue]
       (iterate (fn [[number _]] (explode number)))
       (map-indexed vector)
       (filter (fn [[_ [_ state]]] (= state :done)))
       first
       ((fn [[i [number _]]]
          [number (if (> i 1) :changed :done)]))))
```


This repeatedly applies explode until it signals `:done`, then reports whether anything changed.

### Reducing to a stable form

To fully *normalize* a snailfish number, we must apply explosions and splits repeatedly, explosions first,
then splits, until neither changes the tree.

```clj
(defn normalize [number]
  (let [[exploded e-state] (full-explode number)
        [split-n s-state] (split (root-loc exploded))]
    (if (= :done e-state s-state)
      (z/root split-n)
      (recur (root-loc split-n)))))
````

This recursive loop ensures the number reaches its reduced form.
We always start each pass from the `root-loc`, so the next traversal covers the entire tree correctly.
`z/root` extracts the final value once normalization is complete.


### Combining numbers and Magnitude

Adding two snailfish numbers simply wraps them in a new vector and normalizes:

```clj
(defn add [n1 n2]
  (normalize (z/vector-zip [n1 n2])))
```

Finally, we compute the *magnitude*; a recursive formula:
`magnitude([x, y]) = 3 * magnitude(x) + 2 * magnitude(y)`.

```clj
(defn magnitude [number]
  (walk/postwalk
   (fn [node]
     (if (number? node)
       node
       (+ (* 3 (first node)) (* 2 (second node)))))
   number))
```

`clojure.walk/postwalk` is perfect here, it processes the tree bottom-up, collapsing pairs into scalar magnitudes along the way.

In Part 1, we sum all snailfish numbers in sequence and compute the final magnitude:

```clj
(magnitude (reduce add input))
```

Each call to `add` wraps, normalizes, and returns the reduced result, just like the problem statement.
The zipper operations handle all the nested mutations elegantly, keeping the code declarative and readable.
[Complete Solution](https://github.com/amandeepsp/aoc2021/blob/master/src/aoc/day18.clj)

## Chinks in the Armor?
So far I have been gushing praises for the language; but there are a lot of sharp edges. Errors have been a big pain
for me and the [Clojure community at large](https://ericnormand.me/article/clojure-error-messages-accidental).
Another one for me is the function discoverability, like how I am supposed to know [`remove`](http://clojuredocs.org/clojure.core/remove)
exists? [Clojure Docs](http://clojuredocs.org/) help but this is not great; since all these are
just clubbed in `clojure.core`.

Aside from these Clojure has been a joy to work with and I look forward to using it more in my projects. I have not yet
explored a lot of tooling around dependency management, but I have seem people on the web complaining about it.


[^1]: Some great resources to understand the inner workings
[Understanding Clojure's Persistent Vectors, pt. 1](https://hypirion.com/musings/understanding-persistent-vector-pt-1)
and later posts from [Jean Niklas](https://hypirion.com/category/clojure)
