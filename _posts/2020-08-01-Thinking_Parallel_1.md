---
title: Thinking Parallel, Part I: Collision Detection on the GPU
tags: Collision
---

本篇对blog[《Thinking Parallel, Part I: Collision Detection on the GPU》](https://developer.nvidia.com/blog/thinking-parallel-part-i-collision-detection-gpu/) 进行机器翻译（google)，以便学习其原理及做笔记备忘。

<!--more-->
本系列文章旨在强调传统编程和并行编程在算法层面上的一些主要区别，以宽相碰撞检测为例。第一部分将给出一些背景知识，讨论两种常用的方法，并介绍散度的概念。第二部分将切换到分层树遍历，以展示一个好的单核算法如何在并行设置中变成一个糟糕的选择，反之亦然。第三部分也是最后一部分将讨论并行树的构造，介绍占用的概念，并介绍最近发布的算法，该算法专门设计用于考虑大规模并行性。

#### 为什么要并行？
计算世界正在发生变化。过去，摩尔定律意味着集成电路的性能大约每两年翻一番，您可以期望任何程序在更新的处理器上自动运行得更快。然而，自从处理器架构在 2002 年左右撞上Power Wall以来，提高单个处理器内核的原始性能的机会就变得非常有限。今天，摩尔定律不再意味着你可以获得更快的内核——它意味着你可以获得更多的内核。因此，除非程序能够有效利用不断增加的内核数量，否则程序不会变得更快。

在当前的消费级处理器中，GPU 代表了这一发展的一个极端。例如，NVIDIA GeForce GTX 480 可以并行执行 23,040 个线程，实际上需要至少 15,000 个线程才能达到全部性能。这个设计点的好处是单个线程非常轻量级，但它们一起可以实现极高的指令吞吐量。

有人可能会争辩说，GPU 是一种有些深奥的处理器，只有从事专门应用程序的科学家和性能爱好者才会感兴趣。虽然这在某种程度上可能是正确的，但朝着越来越平行的方向发展似乎是不可避免的。学习编写高效的 GPU 程序不仅可以帮助您获得显着的性能提升，而且还突出了一些基本的算法考虑因素，我相信这些考虑因素最终将与所有类型的计算相关。

许多人认为并行编程很难，他们部分正确。尽管近年来并行编程语言、库和工具在质量和生产力方面取得了巨大飞跃，但它们仍然落后于单核同行。这并不奇怪。这些对应物已经发展了几十年，而主流的大规模并行通用计算只出现了短短几年。

更重要的是，并行编程感觉很难，因为规则不同。我们程序员已经学习了算法、数据结构和复杂性分析，并且我们已经发展出直觉，知道为了解决特定问题需要考虑什么样的算法。在为大规模并行处理器编程时，其中一些知识和直觉不仅不准确，甚至可能完全错误。这一切都是关于学习“并行思考”。

####  排序和扫描
碰撞检测是几乎所有物理模拟中的重要组成部分，包括游戏和电影中的特殊效果，以及科学研究。问题通常分为两个阶段，广义阶段和狭义阶段。宽阶段负责快速找到可能相互碰撞的 3D 对象对，而窄阶段更仔细地查看在宽阶段中发现的每个潜在碰撞对，以查看是否确实发生了碰撞。我们将专注于广泛的阶段，其中有许多众所周知的算法。最直接的一种是[sort and sweep](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)。
![sort and sweep](/img/assets/Collision/fig01-sort-and-sweep.png)

排序和扫描算法的工作原理是为每个 3D 对象分配一个轴对齐边界框 (AABB)，并将边界框投影到选定的一维轴上。投影后，每个对象将对应轴上的一维范围，这样两个对象只有在它们的范围相互重叠时才能发生碰撞。为了找到重叠的范围，该算法将范围的起点（S1，S2，S3）和终点（E1，E2，E3）收集到一个数组中，并沿轴排序. 对于每个对象，它会从对象的起点和终点（例如 S2和 E2）扫描列表，并识别起点位于它们之间的所有对象（例如 S3）。对于以这种方式找到的每一对对象，该算法会进一步检查它们的 3D 边界框是否重叠，并将重叠对报告为潜在的碰撞。

排序和扫描很容易在并行处理器上通过三个处理步骤实现，在每个步骤之后同步执行。
1. 第一步，我们为每个对象启动一个线程来计算其边界框，将其投影到选定的轴上，并将投影范围的起点和终点写入输出数组中的固定位置。
2. 第二步，我们对数组进行升序排序。并行排序本身就是一个有趣的话题，但底线是我们可以使用[并行基数排序](https://code.google.com/archive/p/back40computing/wikis/RadixSorting.wiki)。产生相对于对象数量的线性执行时间（假设有足够的工作来填充 GPU）。
3. 第三步，我们为每个数组元素启动一个线程。如果元素指示结束点，则线程简单地退出。如果元素指示起点，则线程向前遍历数组，执行重叠测试，直到遇到相应的终点。

该算法最明显的缺点是它可能需要在第三步中执行多达 O(n^2) 的重叠测试，而不管 3D 中实际重叠的边界框有多少。即使我们尽可能智能地选择投影轴，最坏的情况仍然非常缓慢，因为我们可能需要将任何给定对象与任意远的其他对象进行测试。虽然这种影响同样会损害排序和扫描算法的串行和并行实现，但在分析并行实现时，我们还需要考虑另一个因素：Divergence(分歧)。


#### Divergence 
Divergence 是衡量附近线程是在做同样的事情还是在做不同的事情的度量。有两种风格：执行分歧意味着线程正在执行不同的代码或做出不同的控制流决策，而数据分歧意味着它们正在读取或写入内存中的不同位置。两者都不利于并行机器的性能。此外，这些不仅仅是当前 GPU 架构的产物——任何足够并行的处理器的性能必然会在一定程度上存在差异。

在传统 CPU 上，我们已经学会了依赖于在执行管道旁边构建的大型数据缓存。在大多数情况下，这使得内存访问实际上是免费的：我们想要访问的数据几乎总是已经存在于缓存中。但是，将并行量增加多个数量级会完全改变情况。由于功率和面积限制，我们不能真正添加​​更多的片上内存，因此我们必须从类似大小的缓存中为更大的线程组提供服务。如果线程在同一时间或多或少地做同样的事情，这不是问题，因为它们的组合工作集仍然可能保持相当小。但如果他们在做完全不同的事情，工作集就会爆炸，

排序和扫描算法的第三步主要受到执行分歧的影响。在上面描述的实现中，碰巧落在端点上的线程将立即终止，其余线程将在数组中遍历可变步数。负责大对象的线程通常比负责小对象的线程执行更多的工作。如果场景包含不同大小的对象的混合，附近线程的执行时间将有很大的不同。换句话说，执行分歧会很大，性能会受到影响。

#### Uniform Grid
![Uniform Grid](/img/assets/Collision/fig02-uniform-grid.png)
另一种适合并行实现的算法是Uniform Grid碰撞检测。在其基本形式中，该算法假定所有对象的大小大致相等。这个想法是构建一个统一的 3D 网格，其单元格至少与最大对象的大小相同。
1. 第一步，我们根据边界框的**质心**将每个对象分配给网格的一个单元格。
2. 第二步，我们查看网格中的 3x3x3 邻域（突出显示）。对于我们在相邻单元格中找到的每个对象（绿色复选标记），我们检查相应的边界框是否重叠。

如果所有对象的大小确实大致相同，它们或多或少是均匀分布的，网格不会太大而无法放入内存，并且 3D 中彼此靠近的对象恰好被分配给附近的线程，这个简单的算法实际上非常高效的。每个线程执行的工作量大致相同，因此执行分歧很小。每个线程还访问与附近线程大致相同的内存位置，因此数据差异也很低。但它需要许多假设才能走到这一步。虽然这些假设可能在某些特殊情况下成立，例如模拟盒子中的流体粒子，但该算法存在许多现实世界场景中常见的所谓体育场内茶壶问题。

当您有一个带有小细节的大场景时，就会出现体育场内的茶壶问题。有各种不同大小的物体，还有很多空旷的空间。如果在整个场景上放置一个统一的网格，单元格将太大而无法有效地处理小对象，并且大部分存储空间将浪费在空白空间上。为了有效地处理这种情况，我们需要一种分层方法。这就是事情变得有趣的地方。

#### 讨论
到目前为止，我们已经看到了两种简单的算法，它们很好地说明了并行编程的基础知识。考虑并行算法的常用方法是将它们分成多个步骤，以便每个步骤针对大量项目（对象、数组元素等）独立执行。凭借同步，后续步骤可以自由地以任何他们喜欢的方式访问先前步骤的结果。

我们还将分歧确定为在比较并行化给定计算的不同方法时要牢记的最重要的事情之一。根据目前提供的示例，似乎最小化散度会使尺度倾斜，有利于“愚蠢”或“蛮力”算法——达到低散度的均匀网格确实必须依赖许多简化假设才能去做。

在我的下一篇文章中，我将重点介绍分层树遍历作为执行碰撞检测的一种手段，旨在更多地阐明优化低散度的真正含义。
关于作者

关于 Tero Karras
Tero Karras 于 2009 年加入 NVIDIA 研究部，此前他曾在 NVIDIA 的 Tegra 业务部门工作了 3 年。他的研究兴趣包括实时光线追踪、详细 3D 内容的表示、并行算法和 GPU 计算。

---
If you like it, don't forget to give me a star.

[![Star This Project](/img/assets/github.svg)](https://github.com/fwzhuang/fwzhuang.github.io)