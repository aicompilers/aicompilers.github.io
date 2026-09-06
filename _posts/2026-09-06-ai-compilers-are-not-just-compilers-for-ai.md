---
layout: post
title: "AI Compilers Are Not Just Compilers for AI"
date: 2026-09-06
---

*This post touches on some of the questions we'll be digging into at [CODAI 2027](https://www.aicompilers.org) — "Where AI Models Meet Modern Hardware," the annual meeting of the AI compiler community, January 18, 2027 in Glasgow alongside HiPEAC. If any of this is close to your work, consider submitting.*

At first glance, an AI compiler is just a compiler with unusual input and unusual hardware. A model goes in, optimized code comes out.

But this view misses some of the most interesting differences between AI compilation and classical compiler design.

AI compilers operate in a world where there are surprisingly few important programs, where transformations may deliberately change numerical results, where moving data can cost more than computing on it, and where the boundary between compiler, runtime, and hardware is still being defined.

These differences change what the compiler knows, what it is allowed to change, and even what it means for the compiled program to be correct.

## There Are Surprisingly Few Important Programs

A classical compiler has to be prepared for almost anything.

Consider what Clang might encounter. A database, a browser, a game engine, an operating system kernel, a numerical simulation, or a small command-line utility can all be C++ programs. Two programs written in the same language may have almost nothing in common.

AI compilers see a very different workload population.

A large fraction of important AI workloads belongs to a relatively small number of model families. Transformers dominate language models. Convolutional networks remain important in vision. Diffusion models follow recognizable structures. Mixture-of-experts models add another recurring pattern.

And inside these models, an even smaller vocabulary appears repeatedly:

matrix multiplication, convolution, attention, normalization, reductions, element-wise operations, and data movement between them.

This is an unusual situation for compiler design.

If thousands of models contain variations of the same computational structures, recognizing those structures becomes extremely valuable. An AI compiler can identify attention, fuse familiar operator sequences, specialize matrix multiplications for known shapes, or optimize an entire subgraph as a unit.

The distinction between optimizing a program and optimizing a model architecture starts to blur.

It also makes high-level information unusually valuable.

At the beginning of compilation, the system might know that a group of operations implements attention. Later it may only see matrix multiplications, reductions, loops, memory accesses, and eventually instructions.

Every lowering step makes some optimizations possible, but can make others impossible.

One of the central questions in an AI compiler is therefore not simply *how do I lower this operation?*

It is:

**What information should I preserve, and when can I afford to lose it?**

This is one reason multi-level intermediate representations have become so important in AI compilation.

## Correctness Is Not Simply Binary

The second difference is more fundamental.

A compiler transformation is normally expected to preserve the semantics of a program. The generated instructions can be completely different, but the observable behavior must remain consistent with the rules of the source language.

Now take a neural network trained using FP32 and quantize its weights to INT8.

The results change.

Replace FP32 computation with BF16.

The results can change.

Change the order in which thousands of values are accumulated.

Again, the result can change.

Yet all of these may be perfectly acceptable transformations of an AI workload.

AI compilation therefore has several useful notions of correctness:

**bit-exact → numerically close → accuracy-equivalent → unacceptable**

These are not the same thing.

Two implementations can produce different tensor values while the model produces effectively identical predictions. Conversely, a numerical difference that appears tiny locally can propagate through a model and cause a measurable accuracy regression.

Quantization makes this particularly obvious.

Converting a model from FP32 to INT8 or INT4 deliberately throws information away. The purpose is not to preserve every intermediate value. The purpose is to retain sufficient model quality while reducing memory consumption, bandwidth, and computation cost.

This means that numerical representation can itself become a compilation decision.

The compiler and deployment stack may choose precision, scaling, accumulation behavior, clipping, or alternative implementations of an operation. They are no longer changing only *how* a computation executes. They can change the numerical computation itself.

That creates a debugging problem that is characteristic of AI systems.

Suppose a model achieves the expected accuracy in PyTorch but loses several percentage points after deployment on an accelerator.

Where is the bug?

It might be a compiler lowering error.

But it might also be quantization. Or an inappropriate accumulation precision. Or a numerical approximation. Or a runtime problem. Or a hardware implementation issue.

The final symptom is the same: the model is less accurate.

Compiler correctness remains essential, but bit-for-bit comparison alone cannot answer every question.

## Moving Data Can Matter More Than Computing

The hardware creates another shift in priorities.

AI accelerators can perform extraordinary amounts of arithmetic. A matrix engine can execute huge numbers of multiply-accumulate operations every second.

Keeping it busy is another matter.

Weights and activations have to move through a memory hierarchy before the arithmetic can happen. Intermediate results have to go somewhere afterwards. Moving all of this data can consume more time and energy than the arithmetic itself.

The compiler therefore spends much of its effort answering questions about data:

Where does a tensor live?

When should it be moved?

How should it be tiled?

Can part of it remain in scratchpad SRAM?

Can two operations be fused so that an intermediate tensor never has to leave on-chip memory?

Can data transfers overlap with computation?

This changes the optimization problem.

Instead of asking only

**How can I compute this faster?**

an AI compiler often asks

**How can I avoid moving this data at all?**

Fusion, tiling, tensor layouts, memory planning, buffering, and asynchronous transfers are consequences of this question.

Scheduling computation and scheduling data movement become difficult to separate.

## The Compiler Is Part of the Architecture

There is also no single equivalent of the mature CPU software/hardware contract across AI accelerators.

Different accelerators expose different matrix engines, memory hierarchies, synchronization mechanisms, numerical formats, sparsity features, and data-movement engines.

Some expose their memory hierarchy explicitly. Others hide more of it. Some depend heavily on compiler-managed transfers. Some introduce operations or data types specifically because they can be implemented efficiently in hardware.

As a result, the compiler is often part of the architecture itself.

A hardware feature that cannot be expressed through the compiler stack is difficult for applications to use. A large matrix engine is of limited value if the compiler cannot keep it supplied with data. A new numerical format only helps if software knows where it can be used without destroying model quality.

For AI accelerators, hardware/compiler co-design is not just an optimization opportunity. It is frequently necessary to make the machine programmable in the first place.

## Compilation Does Not Always End at Compile Time

Finally, some important information does not exist until the model executes.

Tensor dimensions can be dynamic. Batch sizes change. Mixture-of-experts models route tokens depending on their inputs. LLM serving systems maintain KV caches whose size and placement evolve over time. The best kernel can depend on shapes or other runtime conditions.

Not every interesting decision can therefore be made ahead of time.

The compiler may generate multiple implementations and let the runtime choose between them. It may specialize code once shapes become known. Some systems compile parts of a model lazily or trigger recompilation when execution conditions change.

The boundary between compiler and runtime becomes less clear.

Both participate in optimizing the same computation.

## So, What Is an AI Compiler?

None of this means that classical compiler theory has become irrelevant.

Quite the opposite.

AI compilers build on decades of work on intermediate representations, data-flow analysis, SSA, dependence analysis, loop transformations, instruction selection, scheduling, register allocation, and code generation.

But they apply these ideas under a different set of assumptions.

**AI programs are unusually structured.**

**Correctness includes numerical behavior and model accuracy.**

**Performance is often determined by data movement.**

**The hardware/compiler interface is still evolving.**

**And optimization increasingly spans compilation and runtime.**

Together, these differences change what the compiler knows, what transformations it can make, and how we decide whether the result is good.

That is why AI compilers deserve to be studied as more than just another backend for machine learning.

They are an interesting new chapter in compiler design.

## Related Reading

None of these make the argument above end to end, but each covers a piece of it well.

- Chip Huyen, [A Friendly Introduction to Machine Learning Compilers and Optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html) (2021). Still the best on-ramp to the ecosystem itself: computation graphs, local vs. global optimization, why TVM-style autotuning exists. It stays at the level of *how ML compilers work*, not *why they are a different kind of compiler*.
- ["Deep Learning Compilers: A First Visit"](https://github.com/connglli/blog-notes/issues/128), notes on the DL-compiler survey literature. Closest in structure to the "few important programs" argument above: static, domain-specific graph input, tensor-oriented multi-level IR, layout transforms, autotuning. Descriptive rather than argumentative, and it does not touch correctness or the runtime/compile-time boundary.
- [MLSysBook, "Introduction to Machine Learning Systems"](https://mlsysbook.ai/vol1/frameworks/frameworks.html), the frameworks chapter. The clearest published version of the data-movement argument in this post: fusion, layout transforms, whole-graph memory preallocation. Worth reading directly if that section is what you came here for.
- Stephen Diehl, [Introduction to MLIR](https://www.stephendiehl.com/posts/mlir_introduction/). Motivates multi-level IR from the hardware side: LLVM was not built for this much accelerator diversity. Good technical companion to "the compiler is part of the architecture," though it stays a tutorial rather than making that claim explicitly.
