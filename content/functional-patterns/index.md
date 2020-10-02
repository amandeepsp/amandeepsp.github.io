---
title: "Fantastical FP and how to use it"
description: ""
date: 2020-09-20 10:00:00 +0530
categories: [Functional Programming]
path: /functional-patterns
art_type: cubic_disarray
---

In my [last post](/using-functional) I talked about core concepts of functional programming, one of
them was purity. Being pure our functions could not produce any side effects, but side effects are
necessary. Without side-effects, by definition, we could never any output out of our program. So,
functional programmers achieve working programs by delaying side-effect execution to the outskirts
of their programs, maintaining a _functional core with an imperative shell_ ⭕. In this post we look
at _Algebraic data types and structures_ and their compositions to create abstractions over
effect-ful stuff.

## Algebraic data types

Algebraic data types (ADTs, not to be confused with Abstract data types) are types that are composed
of other types. The _algebraic_ here refers to the property that these types are constructed using
_sum_ and _product_. In Haskell, we can define things like

```haskell
data Bool = True | False
```

## Wat da Functor?

## YAMT (Yet another monad tutorial)
