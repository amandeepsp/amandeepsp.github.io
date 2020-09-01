---
layout: blog-post
title: "Upgrading the tool-belt with functional programming"
description: "A Journey to the Ivory tower"
date: 2020-08-12 10:55:06 +0530
categories: [Functional Programming]
image: /assets/ivory_tower.jpg
path: /using-functional
redirects:
    - /fp-is-awesome/
---

<!-- I have a few years of programming experience under my belt and until recently functional programming paradigm was completely unbeknownst to me. I don't have a formal degree in Computer Science, so I was looking to fill sometimes glaring gaps in my knowledge, when I found the amazing [Teach Yourself CS][teach] list of resources to follow.

The first book in the list I started reading was [SICP (Structure and Interpretation of Computer Programs)][sicp] and it was a wild ride, my notions of was can and can't be done were dispelled by the book. I even didn't realize that for the first two chapters no assignments statements were anywhere in the whole code. SICP never tells you, you just get introduced to the functional paradigm of programming and it amazing how naturally it comes together in the book.
more -->

Functional Programming is a programming paradigm that uses functions and composition of those functions to create programs. It make programs declarative. A subset of functional programming is "pure" functional programming, the here idea is that your functions should be pure, i.e. they should follow two simple rules.

1. Functions must return the same value for the same arguments passed, regardless of the fact when or where the function was called. i.e. no internal random stuff.
2. Functions evaluation could not have any side-effects. You might ask what side-effects? 🤔 functions.. side-effects? _Side-effects_ is just jargon for the fact that your functions should not make any changes to the things that outside of its scope, e.g. changing a mutable state, performing I/O and calling other functions that have side effects.

These facts combined make your functions more composable, enhancing your ability to compose large systems out of very simple components.❗ **The code here is written in [Haskell][haskell], but the concepts are language agnostic. Different languages choose and implement these concepts differently.**

## Why Functional?

You might feel like it is overly restrictive and time-consuming to have pure functions. You are not alone. While starting out I thought it was overkill, but now since the benefits are more clearer to me.. it makes sense.

> "The functional programmer sounds rather like a
> medieval monk, denying himself the pleasures of life in the hope that it will
> make him virtuous."
> &mdash; quote from [Hughes et.al. 1990][whyfp]

The benefits might not seem obvious at first and are hugely language dependent but knowing these concepts is also a win. Other times functional programming's inspirations from [Category Theory][cattheo] is a big turnoff for developers. This is aptly summed up in this video adapted from the movie _Downfall_.

<style>
    .embed-container {
        position: relative; 
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
        max-width: 100%; 
    } 
    .embed-container iframe, .embed-container object, .embed-container embed {
        position: absolute; 
        top: 0;
        left: 0;
        width: 100%;
        height: 100%; 
    }
</style>
<div class='embed-container'>
    <iframe src="https://www.youtube-nocookie.com/embed/ADqLBc1vFwI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
    </iframe>
</div>
<br/>
I would argue that functional concepts are still worth learning as it disciplines you to think a program from a different perspective than that of imperative or OOP paradigm and makes you write programs which are not necessarily functional but contains elements of FP.
<!-- The no side-effects rule essentially eliminates the worry of managing the order of execution as nothing now depends on it. This a huge blessing for concurrency as it does away with *shared mutable state*. -->

FP provides _referential transparency_ to programs, means that an _expression_ can be replaced by its corresponding _value_, and this operation doesn't change the execution of the program. To achieve this your expressions must be _pure_, you cannot guarantee this in presence of a mutable state as the value in now dependent on a state.
Also, functions can be run at any time without any consequences and they don't have any external dependency aside from parameters that are passed in. This can result in super easy unit testing of your code as there is no need to mock anything or add dependencies to your tests since there is no outside state that is modified by the function.
It is theorized that a person can only hold 7 ± 2 items in their mind a time, an extensive state and dataflow system can easily overwhelm a developer, this is a problem that FP helps with a lot as functions are their own units and don't affect the outside world.

Here is a tweet by "Uncle" Bob Martin about functional programming, that is a little more assuring than all the mumbo-jumbo I dropped above.

https://twitter.com/unclebobmartin/status/1214173706018312193?ref_src=twsrc%5Etfw

Functional languages try to provide two kinds of glues Higher-order functions and Lazy evaluation, well generally. More kinds of glues exist but they not currently in scope, I just want to emphasize the goodness of functional programming. Higher-order functions are nowadays present in most modern programming languages. Lazy evaluation is a kind of controversial topics, some functional languages have it, some choose not to and some implement it partially (e.g. Scheme providing only lazy lists).
But all of them agree that lazy evaluation cannot be easily fit into an imperative perspective.

### Glues

#### Higher-order functions

Higher-order functions are functions that receive other functions as their arguments

First some [Haskell][haskell] basics... Here obviously I am going at light speed. If you want to learn Haskell, I would recommend [Learn You a Haskell for Great Good!](http://learnyouahaskell.com/). Haskell lists can be constructed as pairs with the `:` operator i.e `[1,2,3]` is `1:2:3:[]` which is also same as `1:2:[3]` or `1:[2,3]` etc. We can write functions on lists such as below. Haskell's functions can be defined using pattern matching so rather having if-else we can define functions in a more mathematical piecewise function kinda way. You can see in the functions below, `sum` of `[]` is 0, otherwise, it is `head` of the list plus the sum of the rest of elements.

```haskell
sum [] = 0
sum (x:xs) = x + sum xs

prod [] = 1
prod (x:xs) = x * prod xs

and [] = True
and (x:xs) = x && and xs

length [] = 0
length (x:xs) = 1 + length xs
```

We can see a kind of abstraction emerging here, the operator and the starting value can be abstracted out. This is not possible in languages that do have higher-order functions. Here `foldr` accumulates values form right.
We could also create a `foldl` that accumulates values form left by just recursing before applying `f` rather than after.

```haskell
foldr f id [] = id
foldr f id (x:xs) = f x (foldr f id xs)

sum     = foldr (+) 0
product = foldr (*) 1
all     = foldr (&&) True
any     = foldr (||) False
length  = foldr (\x acc -> 1 + acc) 0
append xs ys = foldr (:) ys xs

foldl f id [] = []
foldl f id (x:xs) = foldl f (f id x) xs
```

Here `\x acc -> 1 + acc` is lambda function, they start with `\` in Haskell.
We can think of ways of defining `map` and `filter` using `fold`, these are described below. If you think that it is neat that we can express `map` and `filter` in terms of `fold`, it is not a coincidence, there's a bunch of maths involved that you can check [here][universal]

```haskell
map f = foldr (\x acc -> f x : acc) []
filter f = foldr (\x acc -> if p x then x : acc else acc) []
```

You might think that yeah, these are nice to have when we have to deal with lists, be we can conform these functions to many other applications.
A more real-world example of this would be [Redux's][redux] reducers which are pure functions that essentially take a `state` object and an `action` object and returns a new `state` object that represents the state after applying the `action`. If you think actions as a stream you can see how reducers are very much the same thing as `fold`.

We could also glue two functions by composing them to make another function using Haskell's `.` operator where `(f . g) x` is the same as `\x -> f (g x)` aside from the fact that `(f . g)` is now a new function, given the return type of `g` matches with the input of `f`. Some of the neat stuff can be done using this.

```haskell
map f = foldr ((:) . f) []

summatrix = (sum . map) sum
```

If we closely examine the first example `: . f` evaluates to `\x acc-> f x : acc`. Similarly, we can sum a list of lists by glueing together `sum` and `map`, which gives us `\f x -> sum (map f x)` where f is `sum` and x is a list of lists.

#### Lazy evaluation

Lazy evaluation is an evaluation strategy in which an expression is not evaluated until it is needed. Another name for this is call-by-need. Due to this strategy, we can implement things like infinite data structures and the ability to define control structures using as functions. Think of this, if you were to implement a version of if/else using a function with if and else branches as parameters, it wouldn't be able to do it as both branches will be evaluated at the time of function call. But in a language with lazy evaluation, this is do-able as the if or the else statements will only be executed depending on the result of the if-predicate. Infinite data structures can be created by lazily evaluating the arguments of `:` constructor, hence in an infinite list, initially we only have the head of the list and continue to discover the elements of a list are accessed.

Let's go through an example...

```haskell
factors n = filter (\x -> n `mod` x == 0) [1..n]
isprime n = factors n == [1,n]
primes = filter isprime [1..]
take 4 primes -- > [2,3,5,7]
```

This generates the list of all the primes [^1] and you easily apply operations on it e.g primes between a and b, sum of primes until x etc.

### More!!

For a more thorough example, we can look into an implementation of the [Newton-Raphson method][newton] for finding square roots, provided in [SICP][sicp]. This boils down to iterating a function mentioned below, which gives you the square root of $n$. We also check for the difference in subsequent values and stop if the desired accuracy is reached.

$$
x_{i+1} = (x_i + n/x_i)/2
$$

Here's a C program to do this. Pretty straightforward.

```c
const double eps = 0.001;
double sqrt(double n, double x0) {
    double x = x0; // assume x0 != 0
    double y = x + 2*eps; // just for initialization
    while (abs(x-y) > eps) {
        y = x;
        x = (x + n/x)/2;
    }
    return x;
}
```

Ok. What if you need to change the stopping criteria?. In languages with no higher-order function, we would have to write another function with duplicated logic for next value calculation application. Let's take a look inside functional land. First `next` is defined which produces the next `x` value based on the current value. Then `repeat`, generates a sequence of repeated function applications i.e. it generates `[x0, f x0, f (f x0), f (f (f x0)), … ]` and lastly, a stopping criterion which goes though the sequence and returns the apt value. These fit like small lego blocks to generate new functions. Here we can swap out `within` with `relative` it all works out. [^2]

```haskell
next n x = (x + n/x)/2

repeat f x = x:(repeat f (f x))

within eps (a:b:rest) =
   if abs (a-b) <= eps
      then b
      else within eps (b:rest)

sqrt n x0 eps =
   within eps (repeat (next n) x0)

relative eps (a:b:rest) =
   if abs (a-b) <= eps * abs b
      then b
      else relative eps (b:rest)

sqrt n x0 eps =
    relative eps (repeat (next n) x0)
```

This repeated application of a function gets us to the fixed point of a function. More things can be done using this fixed point finder program. We can use it to find $\phi$, the golden ratio for which the function application will be something like $x_{i+1} = 1 + 1/x_i$. We can just plug a new next function and we get the value of $\phi$.

```haskell
next_phi x = 1 + 1/x
phi x0 eps = within eps (repeat next_phi x0)
```

This can even be taken to the next level in which we could also abstract out the process of constructing the next function. You can read more in SICP [Section 1.3][sect1.3].

## Not all sunshine and rainbows

Eventually, the ride comes to an end. You almost absolutely require side effects to do something meaningful say fetch data from the network, query a database, write to a file/console, all these things are side-effects and we need them. Pure functional languages usually push all the side-effects to the edges of the program and still maintain a pure FP core. 100% functional code is very difficult to achieve in a commercial system. "Purely" functional code to almost always less performing that an imperative implementation even with tail-call optimization. FP was invented in the 1950s before the advent of OO, the fact that it never rose to popularity then, is mainly because memory was not that cheap then. Runtime performance also takes a hit across the board. If you care about the bare-metal performance of the code a functional style is difficult to adapt. If we dial down the percentage to a 70-80% range of functional code it is much more doable and can deliver the proposed benefits. For it is not about having no side-effects, but controlled and isolated side-effects. Lazy evaluation in languages is still a hot topic for debate from its power to represent to its runtime and implementation costs.

I will leave you with a profound quote by Micheal Feathers which is just a great summary of what I just talked about in the post.

https://twitter.com/mfeathers/status/29581296216?ref_src=twsrc%5Etf

[^1]: I know this is not the most speedy implementation. We could also have implemented Sieve of Eratosthenes pretty neatly but it comes with its own set of problems. Read more [here][oneil].
[^2]: I am here not commenting about performances of these implementations, the aesthetics and modularity is stressed here. In real world programming readability generally comes first. The C code will generally be faster and consume less memory.

[teach]: https://teachyourselfcs.com/
[sicp]: https://mitpress.mit.edu/sites/default/files/sicp/index.html
[whyfp]: https://www.cs.kent.ac.uk/people/staff/dat/miranda/whyfp90.pdf
[cattheo]: https://en.wikipedia.org/wiki/Category_theory
[universal]: https://jeremykun.com/2013/09/30/the-universal-properties-of-map-fold-and-filter/
[redux]: https://redux.js.org/
[newton]: https://en.wikipedia.org/wiki/Newton%27s_method
[oneil]: https://www.cs.hmc.edu/~oneill/papers/Sieve-JFP.pdf
[sect1.3]: https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/book-Z-H-12.html#%_sec_1.3
[haskell]: https://www.haskell.org/
