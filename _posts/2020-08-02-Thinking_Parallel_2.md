---
title: Thinking Parallel, Part II -> Tree Traversal on the GPU
tags: Collision
---

本篇对blog[《Thinking Parallel, Part II: Tree Traversal on the GPU》](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/) 进行机器翻译（google)，以便学习其原理及做笔记备忘。
<!--more-->

在本系列的第一部分中，我们研究了 GPU 上的碰撞检测，并讨论了两种常用算法，它们使用轴对齐边界框 (AABB) 在一组 3D 对象中找到潜在的碰撞对。这两种算法中的每一种都有其弱点：排序和扫描存在高执行分歧，而统一网格依赖于过多的简化假设，限制了其在实践中的适用性。

在这一部分中，我们将把注意力转向一种更复杂的方法，分层树遍历，它在很大程度上避免了这些问题。在此过程中，我们将进一步探讨发散在并行编程中的作用，并展示几个如何改进它的实际示例。

#### 包围体层次结构(BVH)
我们将围绕[包围体层次结构(BVH) ](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy)构建我们的方法，这是光线追踪中常用的加速结构（例如）。包围体层次本质上是 3D 对象的[层次分组](https://developer.nvidia.com/discover/cluster-analysis)，其中每个组都与一个保守的边界框相关联。
![bvh](/img/assets/Collision/fig03-bvh.png)

假设我们有八个对象，O1 - O8，即上图中的绿色三角形。在 BVH 中，单个对象由叶节点（图中的绿色球体）表示，对象组由内部节点（N1 - N7，橙色球体）表示，整个场景由根节点（N1）表示。每个内部节点（例如 N2）有两个子节点（N4和 N5），并与完全包含所有底层对象（O1 - O4）的边界体积（橙色矩形）相关联。边界体基本上可以是任何 3D 形状，但为了简单起见，我们将使用轴对齐边界框 (AABB)。

我们的总体方法是首先在给定的 3D 对象集上构建一个 BVH，然后使用它来加速搜索潜在碰撞对。我们将把有效层次结构的讨论推迟到本系列的第三部分。现在，让我们假设我们已经有了 BVH。

#### 独立遍历
给定特定对象的边界框，可以直接制定递归算法来查询其边界框重叠的所有对象。以下函数在参数中使用 BVH，并在参数中bvh使用 AABB 来查询它queryAABB。它递归地针对 BVH 测试 AABB 并返回list潜在的冲突。

```C++
void traverseRecursive( CollisionList& list,
                        const BVH&     bvh, 
                        const AABB&    queryAABB,
                        int            queryObjectIdx,
                        NodePtr        node)
{
    // Bounding box overlaps the query => process node.
    if (checkOverlap(bvh.getAABB(node), queryAABB))
    {
        // Leaf node => report collision.
        if (bvh.isLeaf(node))
            list.add(queryObjectIdx, bvh.getObjectIdx(node));

        // Internal node => recurse to children.
        else
        {
            NodePtr childL = bvh.getLeftChild(node);
            NodePtr childR = bvh.getRightChild(node);
            traverseRecursive(bvh, list, queryAABB, 
                              queryObjectIdx, childL);
            traverseRecursive(bvh, list, queryAABB, 
                              queryObjectIdx, childR);
        }
    }
}
```
这个想法是以自上而下的方式遍历层次结构，从根开始。对于每个节点，我们首先检查其边界框是否与查询重叠。如果不是，我们知道底层的叶子节点也不会与它重叠，所以我们可以跳过整个子树。否则，我们检查该节点是叶子节点还是内部节点。如果它是叶子，我们报告与相应对象的潜在碰撞。如果它是一个内部节点，我们继续以递归方式测试它的每个子节点。

要查找所有对象之间的冲突，我们可以简单地对每个对象并行执行一个这样的查询。让我们把上面的代码变成CUDA C++，看看会发生什么。
```C++
 void traverseRecursive( CollisionList& list,
                                   const BVH&     bvh, 
                                   const AABB&    queryAABB,
                                   int            queryObjectIdx,
                                   NodePtr        node)
{
    // same as before...
}

__global__ void findPotentialCollisions( CollisionList list,
                                         BVH           bvh, 
                                         AABB*         objectAABBs,
                                         int           numObjects)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numObjects)
        traverseRecursive(bvh, list, objectAABBs[idx], 
                          idx, bvh.getRoot());
}
```
在这里，我们在__device__的声明中添加了关键字traverseRecursive()，表示代码要在 GPU 上执行。我们还添加了一个__global__ 可以从 CPU 端启动的内核函数。BVH和对象是方便的CollisionList包装器，用于存储访问 BVH 节点和报告冲突所需的 GPU 内存指针。我们在 CPU 端设置它们，并通过值将它们传递给内核。

内核的第一行计算当前线程的线性一维索引。我们不对块和网格大小做出任何假设。至少numObjects以一种或另一种方式启动线程就足够了——任何多余的线程都将被第二行终止。第三行获取相应对象的边界框，并调用我们的函数执行递归遍历，在最后两个参数中传递对象索引和指向 BVH 根节点的指针。

#### 最小化分歧
我们的递归实现最明显的问题是高执行分歧。是否跳过给定节点或递归到其子节点的决定由每个线程独立做出，并且无法保证附近的线程一旦做出不同的决定就会保持同步。我们可以通过以迭代方式执行遍历并显式管理递归堆栈来解决此问题，如下面的函数所示。

```c++
__device__ void traverseIterative( CollisionList& list,
                                   BVH& bvh, 
                                   AABB& queryAABB, 
                                   int queryObjectIdx)
{
    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    NodePtr stack[64];
    NodePtr* stackPtr = stack;
    *stackPtr++ = NULL; // push

    // Traverse nodes starting from the root.
    NodePtr node = bvh.getRoot();
    do
    {
        // Check each child node for overlap.
        NodePtr childL = bvh.getLeftChild(node);
        NodePtr childR = bvh.getRightChild(node);
        bool overlapL = ( checkOverlap(queryAABB, 
                                       bvh.getAABB(childL)) );
        bool overlapR = ( checkOverlap(queryAABB, 
                                       bvh.getAABB(childR)) );

        // Query overlaps a leaf node => report collision.
        if (overlapL && bvh.isLeaf(childL))
            list.add(queryObjectIdx, bvh.getObjectIdx(childL));

        if (overlapR && bvh.isLeaf(childR))
            list.add(queryObjectIdx, bvh.getObjectIdx(childR));

        // Query overlaps an internal node => traverse.
        bool traverseL = (overlapL && !bvh.isLeaf(childL));
        bool traverseR = (overlapR && !bvh.isLeaf(childR));

        if (!traverseL && !traverseR)
            node = *--stackPtr; // pop
        else
        {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR; // push
        }
    }
    while (node != NULL);
}
```
对于与查询框重叠的每个内部节点，循环执行一次。我们首先检查当前节点的子节点是否重叠，如果其中一个是叶子节点，则报告交集。然后我们检查重叠的子节点是否是需要在后续迭代中处理的内部节点。如果只有一个孩子，我们只需将其设置为当前节点并重新开始。如果有两个孩子，我们将左孩子设置为当前节点，并将右孩子压入堆栈。如果没有要遍历的子节点，我们会弹出一个先前被压入堆栈的节点。当我们 pop 时遍历结束NULL，这表明没有更多的节点要处理。

这个内核的总执行时间是0.91毫秒——比递归内核的3.8 毫秒 有了相当大的改进！改进的原因是每个线程现在只是一遍又一遍地执行相同的循环，而不管它最终做出哪些遍历决定。这意味着附近的线程彼此同步执行每次迭代，即使它们正在遍历树的完全不同的部分。

但是如果线程确实在遍历树的完全不同的部分呢？这意味着他们正在访问不同的节点（数据分歧）并执行不同数量的迭代（执行分歧）。在我们当前的算法中，无法保证附近的线程会真正处理 3D 空间中附近的对象。因此，散度的大小对指定对象的顺序非常敏感。

幸运的是，我们可以利用我们想要查询的对象与我们构造 BVH 的对象相同的事实。由于 BVH 的分层特性，在 3D 中彼此靠近的对象也可能位于附近的叶节点中。因此，让我们以相同的方式对查询进行排序，如下面的内核代码所示。
```C++
__global__ void findPotentialCollisions( CollisionList list,
                                         BVH           bvh)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < bvh.getNumLeaves())
    {
        NodePtr leaf = bvh.getLeaf(idx);
        traverseIterative(list, bvh, 
                          bvh.getAABB(leaf), 
                          bvh.getObjectIdx(leaf));
    }
}
```
我们不是像以前那样为每个对象启动一个线程，而是现在为每个叶节点启动一个线程。这不会影响内核的行为，因为每个对象仍将只被处理一次。然而，它改变了线程的顺序以最小化执行和数据分歧。总执行时间现在是0.43毫秒——这个微不足道的变化将我们算法的性能提高了 2 倍！

我们的算法还有一个小问题：每个潜在的碰撞都会报告两次——每个参与对象一次——并且对象也会报告与自己的碰撞。报告两倍的碰撞也意味着我们必须执行两倍的工作。幸运的是，可以通过对算法进行简单修改来避免这种情况。为了让对象 A 报告与对象 B 的碰撞，我们要求 A 在树中必须出现在 B 之前。

为了避免遍历层次结构一直到叶子以确定是否是这种情况，我们可以为每个内部节点存储两个额外的指针，以指示可以通过其每个子节点到达的最右边的叶子。在遍历过程中，当我们注意到一个节点不能用于到达任何位于树中查询节点之后的叶子时，我们就可以跳过它。

```
__device__ void traverseIterative( CollisionList& list,
                                   BVH&           bvh, 
                                   AABB&          queryAABB, 
                                   int            queryObjectIdx, 
                                   NodePtr        queryLeaf)
{
    ...

    // Ignore overlap if the subtree is fully on the
    // left-hand side of the query.

    if (bvh.getRightmostLeafInLeftSubtree(node) <= queryLeaf)
        overlapL = false;

    if (bvh.getRightmostLeafInRightSubtree(node) <= queryLeaf)
        overlapR = false;

    ...
}
```
修改后，算法运行时间为 0.25毫秒。这比我们的起点提高了 15 倍，我们的大多数优化只是为了最小化分歧。

#### 同时遍历
在独立遍历中，我们独立遍历每个对象的 BVH，这意味着我们为给定对象执行的任何工作都不会被其他对象使用。我们可以对此进行改进吗？如果在 3D 中碰巧有许多小对象位于附近，那么它们中的每一个最终都会执行几乎相同的遍历步骤。如果我们将附近的对象分组在一起并为整个组执行单个查询会怎样？

这种思路导致了一种称为[同时遍历](http://gamma.cs.unc.edu/GPUCOL/)的算法。这个想法不是查看单个节点，而是考虑节点对。如果节点的边界框不重叠，我们知道它们各自的子树中的任何地方也不会重叠。另一方面，如果节点确实重叠，我们可以继续测试其子节点之间所有可能的配对。以递归方式继续此操作，我们最终将到达成对的重叠叶节点，它们对应于潜在的冲突。

在单核处理器上，同时遍历非常有效。我们可以从根开始，与自身配对，进行一次大遍历，一次性找到所有潜在的碰撞。与独立遍历相比，该算法执行的工作量要少得多，而且它确实没有缺点——一个遍历步骤的实现在两种算法中看起来大致相同，但是在同时遍历中执行的步骤更少（在我们的算法中减少了 60%）例子）。这是一个更好的算法，对吧？

为了并行化同时遍历，我们必须找到足够的独立工作来填满整个 GPU。实现此目的的一种简单方法是开始遍历层次结构中更深的几个级别。例如，我们可以识别根附近的 256 个节点的适当切割，并为每对节点（总共 32,896 个）启动一个线程。这将导致足够的并行性，而不会过多地增加工作总量。额外工作的唯一来源是我们需要为每个初始对执行至少一个重叠测试，而单核实现将完全避免一些对。

所以，同时遍历的并行实现比独立遍历做的工作少，也不缺乏并行性。听起来不错，对吧？错误的。它实际上比独立遍历要差很多。这怎么可能？

答案是——你猜对了——分歧。在同时遍历中，每个线程都在树的完全不同的部分上工作，因此数据分歧很大。附近线程做出的遍历决策之间没有相关性，因此执行分歧也很大。更糟糕的是，各个线程的执行时间差异很大——给定非重叠初始对的线程将立即退出，而给定与其自身配对的节点的线程可能执行时间最长。

也许有一种方法可以以不同的方式组织计算，以便同时遍历会产生更好的结果，类似于我们对独立遍历所做的那样？已经有很多尝试在其他情况下完成类似的事情，使用巧妙的工作分配、数据包遍历、warp-同步编程、动态负载平衡等。长话短说，您可以非常接近独立遍历的性能，但实际上很难击败它。

#### 讨论
我们已经研究了通过并行遍历分层数据结构来执行广泛阶段碰撞检测的两种方法，并且我们已经看到通过相对简单的算法修改来最小化分歧可以导致显着的性能改进。

比较独立遍历和同时遍历很有趣，因为它突出了关于并行编程的重要一课。独立遍历是一种简单的算法，但它执行的工作比必要的要多。全面的。另一方面，同时遍历对其执行的工作更智能，但这是以增加复杂性为代价的。复杂的算法往往更难并行化，更容易出现分歧，并且在优化方面提供的灵活性较低。在我们的示例中，这些影响最终完全抵消了减少整体计算的好处。

并行编程通常不是关于程序执行多少工作，而是关于该工作是否发散。算法复杂性往往会导致发散，因此首先尝试最简单的算法很重要。很有可能，经过几轮优化后，算法运行得非常好，以至于更复杂的替代方案很难与之竞争。

在我的下一篇文章中，我将专注于并行 BVH 构建，讨论占用问题，并展示最近发布的明确旨在最大化它的算法。

---
If you like it, don't forget to give me a star.

[![Star This Project](/img/assets/github.svg)](https://github.com/fwzhuang/fwzhuang.github.io)