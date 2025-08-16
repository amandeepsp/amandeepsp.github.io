---
title: "Intuition behind Power of 2 Load balancing"
publishDate: "17 Aug 2025"
tags:
  - load balancing
draft: true
---

Power of 2 load balancing is that from a set of targets we choose 2 at random and then compare the load on those two and assign
the request the target that has less load. Load here can any metric you want to choose, e.g. CPU usage, number of requests
in flight, latency stats of requests served by the target. This is also not even the best way to balance requests;
there are far better ways than this.

But the result that compared to just random selection of targets the expected
max load on a target is exponentially better with 2 choices, i.e. $ \approx \frac{\log n}{\log\log n}$ for random and
$ \approx \frac{\log\log n}{\log d}$ for $d$ random choices, for 2 this will be $\log_2 \log n$.
This result is exponentially better than random allocation.
While if you are just given the results it's easy to see why that is since Power of 2 choices approach is approximately
slow by a factor of $ \ln n $. But it is not obvious intuitively, since we are still selecting the two targets randomly.

Here is a visualization of how the Power of 2 choices approch perform better, as you can see in the gif the load appears
more uniform as the number of requests increase.

Say in our servers, $t$ servers are already at max load out of total $n$ server, say max load is $k$.
The probability that the max load grows is just $t/n$ since we can only choose one target. But when we get to choose
two targets, what is the probability that the max load grows? What would take it to make $k+1$?
We will have to choose two target that are already at $k$ since if only one is at $k$ the other target will be
selected and max will not increase. Now what is the probability that two targets are at max and we choose both of
them is $ (t/n)^2 $.

Here $t$ would be small constant. Furthermore to increase the max even more from $k+1$ to $k+2$, the number of servers
with max load would have fallen even more, call it $t'$, the probability will be $(t'/n)^2$ which is event tinier than
moving from $k$ to $k+1$. This is why the tail of max load in case of 2 choices fall very rapidly since with
every iteration the probability of increasing the max falls faster than the single choice method.

