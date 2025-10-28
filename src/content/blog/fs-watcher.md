---
title: The Rabbit Hole of Building a Filesystem Watcher
publishDate: "Sep 23 2025"
tags:
  - linux-kernel
  - systems-programming
featured: true
seo:
  description: |
    A deep dive into building a Linux filesystem watcher. We compare fanotify vs. a powerful eBPF solution to solve in-kernel monitoring challenges.
  keywords:
    - eBPF
    - Linux filesystem watcher
    - BPF CO-RE
    - filesystem monitor
    - Kernel programming
    - fanotify
    - VFS hooks
    - kprobes
    - Linux system programming
    - LSM hooks
    - dentry
    - BTF
---

Some of the systems I work with are highly customized environments, and often need support engineers to maintain them.
A lot of automation exists, but sometimes they need to manually go into a VM and change things. This is normal
, but with these manual tasks, mistakes are inevitable. One such case is a service that would only work if all the files
and directories it manages are owned by a special user. But sometimes people run commands in the service directories
as root. This doesn't impact the service as it's running, but it won't restart. While the fix is simple, just `chown -R`
the service directory. There are many easy ways to prevent this, e.g. setting file permissions,
[File ACLs](https://linux.die.net/man/1/setfacl). These are less strict as `root`
user can override these. Setting SELinux policies would be a much stricter solution. These are very sensible solutions.
But what is the fun it that? How about we build an entire filesystem event watcher ourselves?

## Attempt 1 - `fanotify`

[`fanotify`](https://www.man7.org/linux/man-pages/man7/fanotify.7.html) is a set of APIs in the Linux kernel by which we
could get filesystem events sent to userspace. Let's dive in, according to [man page](https://www.man7.org/linux/man-pages/man7/fanotify.7.html),
we first need to call [`fanotify_init`](https://www.man7.org/linux/man-pages/man2/fanotify_init.2.html) with proper flags;
This sets up a kernel-space notification group. We can set up the directories we need to watch via
[`fanotify_mark`](https://www.man7.org/linux/man-pages/man2/fanotify_mark.2.html).
`fanotify_init` sets up a file descriptor for the event queue, which can be accessed by reading the file descriptor.
This is a great built-in API, but we have a few issues.

1. We cannot monitor a directory recursively. This feature is only available for whole filesystem
   mounts.

2. Another limitation is that `fanotify` only gives us the PID of the process that triggered the event (`metadata->pid`),
   not the full credentials. If we want to know _who_ (which UID/GID) actually performed the operation, we must do an extra
   lookup in `/proc/<pid>` (for example, reading `/proc/<pid>/status`) to fetch the task's credentials. That
   means for every single event, we would need to open and parse a `/proc` file, and then apply our filtering logic.

## Attempt 2 - eBPF

This was when I put this idea on the back burner, but then I stumbled on [eBPF](https://ebpf.io/what-is-ebpf/)
while working on another project with [Falco](https://falco.org/).

eBPF enables running programs in kernel space. Programs are first compiled into bytecode, then verified by an in-kernel static
verifier, then run using JIT for native execution performance. To communicate with the user-space, we can instantiate various forms of
data structures too. The official intro docs do a great job of explaining this, see [What is eBPF?](https://ebpf.io/what-is-ebpf/).

I have been fortunate enough to be writing this at a time when tooling around eBPF has evolved a lot. Earlier tools had to include
kernel headers by either a) compiling the program with the exact kernel source present locally or b) compiling the program on the
server where it will run. Thanks to improvements around adding lightweight type info [BTF(BPF Type Format)](https://nakryiko.com/posts/bpf-portability-and-co-re/)
, [CO-RE (Compile Once - Run Everywhere)](https://nakryiko.com/posts/bpf-portability-and-co-re/) and `libbpf` loader. The user interface
for writing eBPF programs is a bit easier.

Now the question comes, what do you hook into? We can directly hook into [kernel VFS layer](https://www.kernel.org/doc/html/latest/filesystems/vfs.html)
functions such as `vfs_mkdir` and `vfs_create`, which abstract out various filesystem implementations and expose a single filesystem interface to user-space.
We could read the arguments and filter out the events shipped to userspace, saving on a lot of context switches.

This method again has its own slew of annoyances.

1. Using kprobes on functions like `vfs_*` does not guarantee a stable ABI,
   i.e the arguments can change anytime, or functions themselves can disappear across kernel releases.
   In my case, this is not a big deal since I would be running this in a standardized environment with consistent kernel
   versions. But this is a solvable problem, though requiring more engineering effort.
   See [this section about handling kernel change in the BPF-CORE reference](https://nakryiko.com/posts/bpf-core-reference-guide/#dealing-with-kernel-changes-and-feature-detection)

2. We will have to write the path filtering logic in kernelspace using eBPF,
   since `vfs_*` probes will trigger for _all_ events. We will have to walk the filesystem tree up
   and see if some dir matches our monitored dir. Aside from the complexity of writing this,
   each eBPF program is statically verified. It must not contain unbounded loops, and we have a limited stack size (typically 512 bytes).

### Walking up the file tree in eBPF

With the generous help of [Andrii Nakryiko's excellent BPF CO-RE reference guide](https://nakryiko.com/posts/bpf-core-reference-guide/),
I was able to come up with a good enough solution. We can use the `dentry` struct to walk up the tree. But since we can't
have unbounded loops in BPF, I had to truncate the walk at `MAX_DEPTH`,
which is acceptable for my problem statement since the expected depth of the directory I want to monitor is known.

```c
static bool is_monitored_dir(struct dentry *dentry, __u64 target_ino) {
  bpf_rcu_read_lock();
  struct dentry *curr_dentry = BPF_CORE_READ(dentry, d_parent);
  struct inode *curr_inode;
  __u64 curr_ino;
  bool result = false;

  #pragma unroll
  for(int i=0; i < MAX_DEPTH; i++) {
    if (!curr_dentry) {
      break;
    }

    curr_inode = BPF_CORE_READ(curr_dentry, d_inode);
    curr_ino = BPF_CORE_READ(curr_inode, i_ino);
    if (curr_ino == target_ino) {
      result = true;
      break;
    }

    struct dentry *parent_dentry = BPF_CORE_READ(curr_dentry, d_parent);
    if (curr_dentry == parent_dentry) {
      break; // curr_dentry is its own root, we have reached the top of
                    // the tree.
    }

    curr_dentry = parent_dentry;
  }

  bpf_rcu_read_unlock();
  return result;
}
```

Note the [kernel RCU (Read, Copy, Update)](https://www.kernel.org/doc/html/latest/RCU/whatisRCU.html) locks are needed since the `dentry` tree
can change while we are traversing it. The RCU mechanism lets the readers safely traverse without blocking the writers.

For a complete, working example of this approach, please refer to the fs-watcher [GitHub repository](https://github.com/amandeepsp/fs-watcher)
. This repository contains the full source code.

### Better Probes

LSM hooks provide a more stable and semantically meaningful API for monitoring filesystem events, since they are part of the kernel’s
[Linux Security Module framework](https://www.kernel.org/doc/html/latest/security/lsm.html).
They can reduce the number of events you need to filter and eliminate some of the brittleness associated with probing low-level VFS functions.
However, these hooks were not available in the kernel I was working with. With LSM hooks, we have access to the `path` struct with which we can resolve
the name into a buffer using [`bpf_path_d_path`](https://docs.ebpf.io/linux/kfuncs/bpf_path_d_path/). Then we can do a substring search to see if the
path is monitored or not. I will be sure to try this out after our next infra update.

## Wrapping Up

This little experiment turned out to be a great deep dive into Linux kernel internals, eBPF and various trade-offs of running kernel-space programs.
eBPF is a very powerful tool, but also has very sharp edges if you are not careful. This has also been my most rigorous exercise in RTFM'ing.
A lot of information about these tools exists, but it’s scattered across kernel docs, blog posts, and reference guides. Piecing it all together was
a journey in itself.
