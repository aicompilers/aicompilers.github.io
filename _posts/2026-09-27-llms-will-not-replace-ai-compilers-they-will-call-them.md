---
layout: post
title: "LLMs Will Not Replace AI Compilers. They Will Call Them."
date: 2026-09-27
---

Michael J. Klaiber · September 27, 2026

Lately, I hear one claim more and more often: *LLMs will replace AI compilers.*

It is usually said casually, at a conference coffee break or in a comment thread, as if it were already obvious. It deserves a closer look. Depending on how you read it, it is either clearly wrong or almost certainly right, and the difference between the two readings is where the interesting questions are.

My view in short:

- **Taken literally, the claim means the transformer itself performs compilation.** A model goes in, and the LLM's forward passes produce optimized code or a binary. For production AI compilation, this is very unlikely to be efficient.
- **We don't do video encoding in the matmul either.** LLMs today call ffmpeg; they don't emulate it token by token. Compilation is no different.
- **Correctness makes in-weights compilation even harder.** AI compilers already live on a spectrum of numerical correctness. Adding a non-deterministic, unverifiable translator does not help.
- **The realistic question is orchestration.** Will LLMs drive existing compiler passes, search loops, memory planners, and profilers in tool mode, and save what they learn as hints for the next compilation, the way autotuners already do? Increasingly, yes.
- **Writing low-level code is where LLMs will change the most.** Kernels, lowerings, and backends for new hardware will partly be written by LLMs. That makes compiler infrastructure more important, not less.

![Three roles for LLMs in AI compilation: compiler, orchestrator, author]({{ '/assets/img/llm_three_roles.svg' | relative_url }})

None of this means machine learning in compilers is new. Learned cost models, ML-guided autotuning, and ML-based inlining and register-allocation heuristics have been around for years. What these systems share is instructive: in every case, the model is a component inside the compiler, not a replacement for it.

## What the Literal Claim Implies

If an LLM replaces the compiler, the translation from model to machine code happens inside the transformer.

The LLM reads a computation graph, reasons about it, and emits the lowered result: IR, assembly, or a binary. Fusion decisions, tiling, memory planning, scheduling, and instruction selection all happen implicitly, somewhere in billions of parameters.

Consider what that costs.

A memory planner that assigns buffers to on-chip SRAM runs in milliseconds. A tiling search over a few hundred candidates, evaluated with an analytical cost model, might take seconds. An LLM producing the same decisions token by token spends billions of floating-point operations per token, across thousands of tokens, to reach a result that a purpose-built algorithm computes almost for free.

Compilers are not slow because nobody thought hard enough about them. They are fast because decades of work went into turning hard problems into tractable algorithms: dataflow analysis, polyhedral scheduling, graph coloring, dynamic programming over fusion groups. Replacing those algorithms with next-token prediction discards that work and pays heavily for it.

There is a certain irony here. Large language models are among the most intensively compiled programs in existence. Serving them efficiently depends on exactly the fusion, quantization, memory planning, and kernel specialization that AI compilers provide. An LLM acting as a compiler would itself run on top of one.

## Nobody Encodes Video in the Matmul

A useful comparison is video.

Ask a modern LLM agent to convert a video to H.265 at a given bitrate. It will not attempt to produce the encoded bitstream token by token. It will write a command line for ffmpeg, run it, check the result, and perhaps adjust the parameters.

Nobody considers this a limitation. It is the correct division of labor. The encoder implements decades of signal processing knowledge efficiently and deterministically. The LLM understands the intent, chooses the tool, sets the parameters, and handles the exceptions.

![In the weights vs. tool mode: video encoding and AI compilation]({{ '/assets/img/llm_tool_mode_ffmpeg.svg' | relative_url }})

**Tool mode is not a workaround. It is the architecture.**

There is no reason compilation should be different. Compiler passes are tools. Autotuners are tools. Profilers, simulators, and verifiers are tools. An LLM that uses them well is far more valuable than an LLM that tries to replicate them internally.

## Correctness Makes It Worse, Not Better

In a [previous post](blogpost_ai_compilers.md) I argued that correctness in AI compilation is not binary. It is a spectrum:

**bit-exact → numerically close → accuracy-equivalent → unacceptable**

Debugging an accuracy regression after deployment is already hard. Was it a lowering bug, quantization, accumulation precision, the runtime, or the hardware?

Now add a translator that is non-deterministic, cannot explain its decisions in a checkable way, and may produce a different result when asked twice.

A classical compiler pass can be tested, reviewed, bisected, and reasoned about. When it is wrong, it is wrong reproducibly. An in-weights translation offers none of these properties. Every compiled model would need to be verified end to end, and at that point the verification infrastructure is doing much of the work a compiler would have done anyway.

For a demo, this may be acceptable. For a deployment stack shipping millions of devices, it is not.

## The Right Question: Orchestration

So let's replace the literal claim with a better question:

**Will LLMs orchestrate AI compilers?**

Here the answer looks increasingly like yes.

A lot of what an experienced compiler engineer does is not algorithmic. It is judgment. Which pass pipeline fits this model? Why did fusion fail on this subgraph? Should this layer use a different quantization scheme? What does this profile say about where the time goes? Which flag, which tile size, which layout?

These are exactly the tasks LLMs handle well in tool mode. An agent can read the IR, call a pass, inspect the result, run the profiler, compare against a baseline, and iterate. The heavy lifting stays inside the tools: the search loop, the memory planner, the cost model. The LLM decides what to run and interprets what comes back.

There is one more thing an orchestrating LLM can borrow from existing tools: memory.

Autotuners have done this for years. A tuning run explores tile sizes, schedules, and kernel variants, then writes the best configurations to a log. The next compilation of the same operator on the same hardware skips the search and reuses the result. The expensive exploration is paid once and amortized across every later build.

An LLM orchestrator can work the same way. It can record which pass pipeline worked for a model family, why a fusion failed, which quantization scheme held accuracy, and which flags mattered on which target, then save these as hints the compiler consumes next time. The next compilation starts from what was learned instead of from scratch.

This also answers the cost argument. The LLM's reasoning becomes a one-time investment rather than a per-compile expense. What it produces is data: a tuning record, a pass configuration, a hint file. That output can be versioned, reviewed, and replayed deterministically, without calling the model again.

The same caveat applies as for autotuning logs. Hints must be keyed to the model, shapes, hardware, and compiler version, or yesterday's good decision quietly becomes today's regression.

LLMs can also improve heuristics. They can propose a better cost model, write a new fusion rule, or find a better search strategy for a tiling space. But notice what happens in that case. The improvement is expressed as code, and that code becomes part of the compiler. The LLM is used in tool mode again, the same way it uses ffmpeg today.

**Why route a decision through the transformer when a good heuristic already exists? The better use of the transformer is to improve the heuristic.**

## Where LLMs Will Change the Most: Writing the Low-Level Code

The version of the claim I find most credible is not about compilation at all. It is about authorship.

A large part of the cost of AI compiler stacks lies in writing low-level code: hand-tuned kernels, lowering patterns, target-specific backends, support for each new operator, data type, and hardware generation. This work is expensive, specialized, and never finished. Every new accelerator and every new model architecture restarts part of it.

This is where LLMs are already making progress. Generating a Triton or CUDA kernel, writing an MLIR lowering pattern, or porting an operator to a new backend are well-scoped tasks with a clear success criterion. It must be correct, and it must be fast.

That criterion is the key point. Generated kernels only become useful inside a harness that checks correctness against a reference, measures performance on real hardware, and rejects everything else. Early work on automated kernel generation has already shown how easily a generator can game a weak evaluation harness. Without rigorous verification, "faster" can simply mean "wrong in a way the test didn't catch."

So LLM-generated low-level code does not remove the need for compiler infrastructure. It increases it:

- **IR becomes the interface.** A well-defined intermediate representation is what gives generated code a place to plug in, and what makes it checkable.
- **Verification becomes central.** Numerical comparison, accuracy evaluation, and equivalence checking become the gate every generated artifact must pass.
- **Search and autotuning become more valuable.** An LLM that proposes a hundred kernel variants needs infrastructure to evaluate them.
- **Compiler knowledge moves from writing code to specifying it.** Someone still has to decide what the kernel must do and what makes it correct.

This is an evolution of AI compilers, not their replacement. Code generation gets a new, very capable author. The compiler becomes the environment that author works in.

## So, Will LLMs Replace AI Compilers?

Not in the sense the claim usually implies.

**Compiling in the weights is inefficient and hard to verify.**

**Orchestrating compiler tools is realistic and useful, and gets cheaper as it learns.**

**Writing kernels, lowerings, and heuristics is where LLMs will have the largest impact.**

**All three depend on compiler infrastructure becoming more rigorous, not less.**

The more interesting future is not one where LLMs make compilers obsolete. It is one where compilers expose clean interfaces, verifiable IR, and fast search, so that LLMs can use them effectively. In that world, the AI compiler engineer does not disappear. The job shifts toward building the system that LLM-written code runs inside, and toward deciding what "correct" means.

That is not a smaller role. It may well be a bigger one.

## Related Reading

- Trofin et al., [MLGO: a Machine Learning Guided Compiler Optimizations Framework](https://arxiv.org/abs/2101.04808) (2021). Machine-learning-guided optimization in LLVM. An example of ML as a component inside a production compiler.
- Zheng et al., [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762) (OSDI 2020). TVM auto-scheduling: learned cost models, search, and tuning logs that make later compilations faster.
- Ouyang et al., [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) (2025). Evaluating LLM-generated kernels, including the problem of verification.

---

*These questions are close to what we'll be discussing at [CODAI 2027](https://www.aicompilers.org) — "Where AI Models Meet Modern Hardware," the annual meeting of the AI compiler community, January 18, 2027 in Glasgow alongside HiPEAC. If your work touches LLM-driven compilation, kernel generation, or verification, consider submitting.*
