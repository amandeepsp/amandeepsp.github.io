---
title: The Rabbit Hole of Building a Filesystem Watcher
publishDate: "Sep 17 2025"
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
[File ACLs](https://linux.die.net/man/1/setfacl). These are less strict as `root`
user can override these. Setting SELinux policies would be a much strict solution. These are very sensible solutions. But what is the fun it that? How about we build an entire filesystem event watcher ourselves?

## Attempt 1 - `fanotify`

[`fanotify`](https://www.man7.org/linux/man-pages/man7/fanotify.7.html) is a set of APIs in the Linux kernel by when we
could get filesystem events sent to userspace. Let's dive in, according to [man page](https://www.man7.org/linux/man-pages/man7/fanotify.7.html),
we first need to call [`fanotify_init`](https://www.man7.org/linux/man-pages/man2/fanotify_init.2.html) with proper flags; this sets up a kernel-space notification group. 
We can setup the directories we need to watch via [`fanotify_mark`](https://www.man7.org/linux/man-pages/man2/fanotify_mark.2.html). 
`fanotify_init` sets up a file descriptor for the event queue which can be accessed by reading the file descriptor. This is a very nice API but we will have two problems.

1. We have to do event filtering in userspace, i.e all events need to be delivered via read syscall and then processed. On a busy system with high event volumes, this can introduce significant overhead in userspace processing, though the kernel-side impact on the filesystem is designed to be minimal.

2. Another limitation is that `fanotify` only gives us the PID of the process that triggered the event (`metadata->pid`),
not the full credentials. If we want to know *who* (which UID/GID) actually performed the operation, we must do an extra
lookup in `/proc/<pid>` (for example, reading `/proc/<pid>/status`) to fetch the task's credentials. That
means for every single event, we would need to, open and parse a `/proc` file, and then apply
our filtering logic. This adds significant overhead and contention, especially since PIDs can be reused
quickly and the task might even exit before we look it up. [^1]

## Attempt 2 - eBPF

This was when I put this idea on the back-burner, but then I stumbled on [eBPF](https://ebpf.io/what-is-ebpf/) while working on another project with [Falco](https://falco.org/).

With eBPF, we can hook directly into the kernel code paths where these events occur.
For example we can directly hook into kernel VFS functions such as  `vfs_mkdir` and `vfs_create`. We could read the arguments and filter out the events shipped to userspace saving
on a lot of context switches.

This method again has its own slew of annoyances.

1. Using kprobes on functions like `vfs_*` does not guarantee a stable API, i.e the arguments can change anytime or functions themselves can disappear across kernel releases. 
While for my case this is not a big deal since I would be running this a standardized environment with consistent kernel
versions. But this is a solvable problem, though requiring more engineering effort.
See [this section about handling kernel change in the BPF-CORE reference](https://nakryiko.com/posts/bpf-core-reference-guide/#dealing-with-kernel-changes-and-feature-detection)

2. We will have to write the path filtering logic in kernelspace using eBPF, since `vfs_*` probes will trigger for *all* events. We will have to walk the filesystem tree up
and see if some dir matches our monitored dir. Aside from the complexity of writing this,
each eBPF program is statically verified. It must not contain unbounded loops and we have a limited stack size (typically 512 bytes).

### Walking the Tree in BPF
With the generous help of [Andrii Nakryiko's excellent BPF CO-RE reference guide](https://nakryiko.com/posts/bpf-core-reference-guide/), I was able come up with a good
enough solution. We can use the `dentry` struct to walk up the tree. But since we can't
have unbounded loops in BPF, I had to truncate the walk at `MAX_DEPTH`, this is acceptable for my problem statement since the expected depth of the directory I want to 
monitor is known.

```c
static __always_inline bool is_monitored_dir(struct dentry *dentry,
                                             __u64 target_ino) {
  struct dentry *curr_dentry = BPF_CORE_READ(dentry, d_parent);

  struct inode *curr_inode;
  __u64 curr_ino;

  #pragma unroll
  for (int i = 0; i < MAX_DEPTH; i++) {
    curr_inode = BPF_CORE_READ(curr_dentry, d_inode);
    if (!curr_inode) {
      // Should not happen in a valid FS hierarchy, but check anyway
      return false;
    }

    curr_ino = BPF_CORE_READ(curr_inode, i_ino);
    if (curr_ino == target_ino) {
      // We have found our montiored directory
      return true;
    }

    struct dentry *parent_dentry = BPF_CORE_READ(curr_dentry, d_parent);
    if (curr_dentry == parent_dentry) {
      return false; // curr_dentry is its own root, we have reached the top of
                    // the tree.
    }

    curr_dentry = parent_dentry;
  }
  
  return false;
}
```

For a complete, working example of this approach, please refer to the fs-watcher [GitHub repository](https://github.com/amandeepsp/fs-watcher)
. This repository contains the full source code, including the BPF program, user-space components, and build instructions.

### Better Probes
LSM hooks provide a more stable and semantically meaningful API for monitoring filesystem events, since they are part of the kernel’s [Linux Security Module framework](https://www.kernel.org/doc/html/latest/security/lsm.html). They can reduce the number of events you need to filter and eliminate some of the brittleness associated with probing low-level VFS functions.
However, these hooks were not available in the kernel I was working with. 
You can check if BPF LSM hooks are available in your kernel using [`bpftool`](https://man.archlinux.org/man/bpftool.8.en) with:

```bash
bpftool feature probe
```

[Quentin Monnet's blog post about bpftool](https://qmonnet.github.io/whirl-offload/2021/09/23/bpftool-features-thread/) is also great starting point since the man pages are not very fleshed out.

## Fin.

This has been fun exercise in exploring kernel APIs and pushing BPF to its limits.
The performance trade offs that came about during research were also very interesting. 
Also, it should be very obvious that this is nowhere near production ready code, just a
proof of concept I cobbled together. Please do benchmark everything before deploying.

[^1]: There might be better ways to get uid/gid from pid, I have not searched well enough.