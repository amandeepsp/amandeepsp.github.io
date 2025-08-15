---
title: The Rabbit Hole of Building a Filesystem Watcher
publishDate: "Aug 17 2025"
tags:
  - bpf
  - linux
featured: true
draft: true
---

Some of the systems I work with are highly customized environments, and often need support engineers to maintain.
A lot of automation exists; but sometimes they would need to manually go into a VM and change things, this is normal
but with these manual tasks mistakes are inevitable. One such case is a service that would only work if all the files
and directories it manages are owned by a special user. But sometimes people run commands in the service directories
as root. This doesn't impact the service as its running but it won't restart. While the fix is simple, just `chown -R`
the service directory. There are many easy ways to prevent this e.g. setting file permissions,
[File ACLs](https://linux.die.net/man/1/setfacl). These are less strict _Discretionary Access Control_ since `root`
user can override these. Setting SELinux policies would be much a much strict solution and will be a
_Mandatory Access Control_. These are very sensible solutions. But what is the fun it that? How about we build an
entire filesystem event watcher ourselves?

## Attempt 1 - `fanotify`

[`fanotify`](https://www.man7.org/linux/man-pages/man7/fanotify.7.html) is set of APIs in the Linux kernel by when we
could get filesystem events sent to userspace. Let's dive in, according to [man page](https://www.man7.org/linux/man-pages/man7/fanotify.7.html),
we first need to call [`fanotify_init(2)`](https://www.man7.org/linux/man-pages/man2/fanotify_init.2.html) with proper flags; this sets up a kernel-space
notification group. We can add

## Attempt 2 - eBPF

[eBPF](https://ebpf.io/what-is-ebpf/) is a way to run sandboxed programs inside the Linux kernel without writing kernel modules or rebooting your system.
It started life as a packet filter for networking, but these days it’s more like a programmable Swiss Army knife for the kernel —
capable of tracing syscalls, profiling applications, enforcing security rules, and yes, watching the filesystem. It is now ubiquitous and every major
cloud provider and Linux distro supports it.
