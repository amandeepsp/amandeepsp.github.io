---
layout: blog-post
title: "Upgrading the tool-belt with functional programming"
description: ""
date: 2020-08-12 10:55:06 +0530
categories: [Functional Programming]
image: /assets/ivory_tower.jpg
path: /using-functional
redirects:
    - /fp-is-awesome/
art_type: circle_packing
---

Functional Programming is, putting it plainly, a programming paradigm that uses functions and
composition of those functions to create programs. Thought leaders and proponents claim that it
makes programs more declarative, have fewer bugs and easier to debug and test. This paradigm when
taken to the extreme is called _pure functional programming_ where all the functions are "pure".
Functional programming hasn't achieved much of traction as OOP (Object Oriented Programming) has,
but almost all popular languages are incorporating constructs to enable a functional style of
programming. Let's dabble into a few core concepts of functional programming that may help to
understand what the fuss is all about. At the end of the article you might feel like the guy in the
meme.

![WOW]

## Core concepts

### Higher Order functions

> Higher Order functions are functions that take other functions as arguments or return them as
> results.

Higher-Order functions can enable us to build nifty abstractions. In JavaScript consider the `map`,
`reduce` and `filter` functions which are just abstractions for looping structures. The `sum`
function implemented below is nothing special we can replace the starting value of `acc` to 1,
replace + with \* and we can get a function for the product. This idea when extended gives the
`reduce` function. We have separated the process of reducing from how data is reduced.

```js
function sum(arr) {
    let acc = 0 //intial value
    for (let i = 0; i < arr.length; i++) {
        acc = acc + arr[i] // acc = func(acc, arr[i])
    }
    return acc
}
```

```js
const sum = arr.reduce((acc, val) => acc + val, 0)
const product = arr.reduce((acc, val) => acc * val, 0)
const all = arr.reduce((acc, val) => acc && val, true)
const any = arr.reduce((acc, val) => acc || val, false)
const length = arr.reduce((acc, val) => 1 + acc, 0)

const append = (xs, ys) => ys.reduce((acc, val) => acc.concat(val), xs)

const map = (func, arr) => arr.reduce((acc, val) => acc.concat(func(val)), [])
const filter = (predicate, arr) =>
    arr.reduce((acc, val) => (predicate(val) ? acc.concat(val) : acc), [])

const summatrix = (matrix) => sum(map(sum, matrix))
```

The same thing applies when we pass a comparison function to `sort`. `sort` doesn't make assumptions
of what kind of an array are we passing in, unless we tell it how to compare it. Had higher-order
functions not been at our disposal we would have to write separate `sort` for every type of array
passed in.

You might think that yeah, these are nice to have when we have to deal with lists, be we can conform
these functions to many other applications. A more real-world example of this would be
[Redux's][redux] reducers which are pure functions that essentially take a `state` object and an
`action` object and returns a new `state` object that represents the state after applying the
`action`. If you think actions as a stream you can see how reducers are very much the same thing as
`reduce`. Which is also why these functions are called _reducers_.

Let's look at another example. Suppose you want to log events on a page. We might do it something
like this

```js
function logEvent(pageName, eventName) {
    //send logging event...
}

logEvent("SAMPLE_PAGE", "A_CLICKED")
logEvent("SAMPLE_PAGE", "B_CLICKED")
logEvent("SAMPLE_PAGE", "C_CLICKED")
```

Here `js>"SAMPLE_PAGE"` is repeated unnecessarily, when we know that `pageName` is the same for a
page. To avoid this we could write a function that takes in `pageName` and returns a function to log
the event.

```js
function logWithPage(pageName) {
    return function (eventName) {
        logEvent(pageName, eventName)
    }
}

const logPageEvent = logWithPage("SAMPLE_PAGE")
logPageEvent("A_CLICKED")
logPageEvent("B_CLICKED")
logPageEvent("C_CLICKED")
```

Here `logWithPage` is an example of a function that returns a function. `logWithPage` returns a
function that can access `pageName` argument due to closure. This technique has a name... _Partial
application_ when we return a new function by providing some of the arguments of the function.
Functions returning functions are very common in the React ecosystem. An example of this would be
the `connect` function from [Redux][redux]. `connect` takes in arguments that tell in how data
should be mapped and returns a function that takes in the React component that needs connection.

#### Currying and Partial application

The idea of partial application can be extended by converting a function that takes multiple
arguments into a series of functions that take only one argument. Let's add another argument to our
`logEvent` function.

<!--TODO: make diff-[language] in next release of gatsby-remark-prismjs-->

```diff
- function logEvent(pageName, eventName){
+ function logEvent(pageName, referrer, eventName){
```

We can now curry the function like this

<!-- prettier-ignore -->
```js
const cLogEvent = (pageName) => (referrer) => (eventName) =>
    logEvent(pageName, referrer, eventName)

const logWithPageAndReferrer = cLogEvent("THE_PAGE")("OTHER_PAGE")

["EVENT_1", "EVENT_2", "EVENT_3"].forEach(logWithPageAndReferrer)
```

Here `cLogEvent` is a function that takes in `pageName` and returns a function that takes in
`referrer`, that returns a function... you get the gist. In purely functional languages like
_Haskell_ all functions are pre-curried. Currying a function also helps to delay function evaluation
to a time when all the arguments are available and for time being we can apply only available
arguments. In the example above `pageName` and `referrer` are known to us at the time of page load,
but events are only known when they happen. Calling this with all the arguments is a bit ugly in
this form `js>cLogEvent(x)(y)(z)` There are nicer ways of doing this using library like
[lodash][lodash] and [Rambda][rambda], using which you can call it normally like
`js>cLogEvent(x, y, z)`, even though it is internally curried.

### Pure functions and Immutability

Pure functions follow two rules.

1. Functions must return the same value for the same arguments passed, regardless of the fact when
   or where the function was called.
2. Functions evaluation could not have any observable side-effects. _Side-effects_ means that your
   functions should not make any changes to the things that outside of its scope, e.g. changing a
   mutable state, performing I/O and calling other functions that have side effects.

Take the case of `slice` and `splice`, an example from [Mostly Adequate Guide to FP][mostly].
`slice` gives you a new array without changing the original but `splice` changes the original array.
Both have the same effect i.e select a range, but `splice` is impure as it removes the selected
elements from the original array, thus performing a side-effect.

```js
const xs = [1, 2, 3, 4, 5]

// pure
xs.slice(0, 3) // [1,2,3]
xs.slice(0, 3) // [1,2,3]
xs.slice(0, 3) // [1,2,3]

// impure
xs.splice(0, 3) // [1,2,3]
xs.splice(0, 3) // [4,5]
xs.splice(0, 3) // []
```

#### Case for purity

It is theorized that a person can only hold 7 ± 2 items in their mind a time[^1], an extensive state
and dataflow system can easily overwhelm a developer, this is a problem that purity helps with a lot
as functions are their own units and don't affect the outside world. Also, functions can be run at
any time without any consequences and they don't have any external dependency aside from parameters
that are passed in. This can result in super easy unit testing of your code as there is no need to
mock anything or add dependencies to your tests since there is no outside state that is modified by
the function. We could list some other benefits

-   Functions can be easily memoized by input arguments since return values are always the same.
-   Things are now easier to test as there is no external dependency to mock. We can just give
    inputs and asserts outputs.
-   Pure functions provide _referential transparency_ to programs, means that an _expression_ can be
    replaced by its corresponding _value_, and this operation doesn't change the execution of the
    program. Being able to do this replacement, we can easily figure out how our code works.
-   Code gets highly parallelized as we have eliminated the pesky _shared mutable state_.

### Composition

Composition is a more abstract idea of combining smaller things to make bigger and complex things.
The most basic idea here is function composition. In mathematics, given functions $f: X \to Y$ and
$g: Y \to Z$, then function composition $ (g \circ f)(x) = g(f(x))$. In programming,

```js
const compose2 = (g, f) => (x) => g(f(x))
```

`compose2` composes two _compatible_ functions into one. A more generalized version of compose takes
in multiple functions

```js
const compose = (...fns) => (x) => fns.reduceRight((acc, fn) => fn(acc), x)
```

Composing functions create a new function that can be used as its own standalone thing. Here, the
catch is we can compose functions with only one argument, except the first one, we can have multiple
arguments for the first function in the compose sequence. We can easily solve this by partially
applying the functions with more arguments. Let's go through an example, here we read a file to
extract `HOST` variable in it as `HOST=www.abc.xyz`. We could do this by reading a file[^2], finding
the line with `HOST` and splitting it to get the hostname. Take a look at the `get_host` function,
it is constructed using nested function calls.

```js {21,22}
const fs = require("fs");
import * as R from "rambda"

const { curry } = R

function cat(filepath) {
  return fs.readFileSync(filepath, "utf-8");
}

function grep = curry((pattern, content) => {
  const exp = new RegExp(pattern);
  const lines = content.split('\n');

  return lines.find(line => exp.test(line));
})

function cut = curry(({ delimiter, fields }, str) => {
  return str.split(delimiter)[fields - 1];
})

const get_host = (filepath) =>
    cut({ delimiter: "=", fields: 2 }, grep("^HOST=", cat(filepath)))
```

Using `compose` this can be simplified into a pipeline on functions which is much easier to
understand what is going on. But the ordering is a bit awkward, data flows from right to left.

<!-- prettier-ignore -->
```js
const get_host = compose(
    cut({ delimiter: "=", fields: 2 }), 
    grep("^HOST="), 
    cat
)
```

There's a cousin to `compose` called `pipe` which just reverses the direction of data flow which is
much natural to look at.

<!-- prettier-ignore -->
```js
const { pipe } = R

const get_host = pipe(
    cat,
    grep("^HOST="),
    cut({ delimiter: "=", fields: 2 })
)
```

Composition helps a lot in improving the readability and ease of understanding of our code. Some
purely functional languages have first-class support for composition. In _Haskell_ `.` operator is
used to compose e.g. `f . g`. The parallels with LEGOs are much overused here, people often say that
composition is the same as building using small, atomic LEGOs to build complex things, and I kinda
agree with this.

Functional programming is a very different style, which is not how most programmers are taught
programming. It offers many advantages to an imperative style. But imperative programming has its
place in the programming as eventually, we need to cause side-effects. DB queries, DOM renders, API
calls all are side effects, but we can learn from functional principles and incorporated them to
make our code more declarative.

[^1]:
    George A. Miller,
    [The Magical Number Seven, Plus or Minus Two: Some Limits on our Capacity for Processing Information](https://web.archive.org/web/20100619202020/http://psychclassics.asu.edu/Miller/)

[^2]:
    Here I am assuming file reading operation always succeeds and ignoring the impurity of a file IO
    operation, there are ways to handle these using _Monads_, for more curious you could also check
    out Either and IO Monads.

[redux]: https://redux.js.org/
[lodash]: https://lodash.com/docs/#curry
[rambda]: https://ramdajs.com/docs/#curry
[mostly]: https://mostly-adequate.gitbooks.io/mostly-adequate-guide/content/ch03.html
[wow]:
    assets/wow.png
    "Image source[@impurepics](https://impurepics.com/posts/2020-06-21-fp-is-wow.html)"
