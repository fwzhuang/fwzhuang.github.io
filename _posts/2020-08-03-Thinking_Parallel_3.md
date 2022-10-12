---
title: Thinking Parallel, Part III-> Tree Construction on the GPU
tags: Collision
---

本篇对blog[《Thinking Parallel, Part III: Tree Construction on the GPU》](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/) 进行机器翻译（google)，以便学习其原理及做笔记备忘。
<!--more-->

在本系列的第二部分 中，我们将分层树遍历视为一种快速识别可能发生冲突的 3D 对象对的方法，并演示了如何针对低散度进行优化可以在大规模并行处理器上带来显着的性能提升。然而，拥有一个快速遍历算法并不是很有用，除非我们也有一棵树来配合它。在这一部分中，我们将通过查看树的构建来闭合圆圈；具体来说，并行包围体层次结构 (BVH) 构造。我们还将看到一个算法优化的例子，它在单核处理器上完全没有意义，但在并行设置中会带来显着的收益。

BVH 有很多用例，也有很多构建它们的方法。在我们的案例中，施工速度至关重要。在物理模拟中，对象不断地从一个时间步移动到下一个时间步，因此我们需要为每一步使用不同的 BVH。此外，我们知道我们将只花费大约 0.25 毫秒来遍历 BVH，因此在构建它上花费更多是没有意义的。处理动态场景的一种众所周知的方法是一遍又一遍地循环使用相同的 BVH。基本思想是只根据新的对象位置重新计算节点的边界框，同时保持层次结构节点结构相同。还可以进行小的增量修改，以改进移动最多的对象周围的节点结构。然而，困扰这些算法的主要问题是，随着时间的推移，树会以不可预测的方式恶化，这在最坏的情况下会导致任意糟糕的遍历性能。为了确保可预测的最坏情况行为，我们选择在每个时间步从头开始构建一棵新树。让我们看看如何。

#### 利用 Z 阶曲线
当前最有希望的并行 BVH 构造方法是使用所谓的线性 [BVH (LBVH)](https://luebke.us/publications/eg09.pdf)。这个想法是通过首先选择叶子节点（每个对应一个对象）出现在树中的顺序来简化问题，然后以尊重该顺序的方式生成内部节点。我们通常希望在 3D 空间中彼此靠近的对象也位于层次结构中的附近，因此合理的选择是沿着[空间填充曲线](https://en.wikipedia.org/wiki/Space-filling_curve)对它们进行排序。为简单起见，我们将使用Z 阶曲线。

![Z-curve](/img/assets/Collision/fig04-z-curve.png)

Z阶曲线是根据Morton 代码定义的. 为了计算给定 3D 点的 Morton 码，我们首先查看其坐标的二进制定点表示，如图左上部分所示。首先，我们获取每个坐标的小数部分，并通过在每个位之后插入两个“间隙”来扩展它。其次，我们将所有三个坐标的位交织在一起以形成一个二进制数。如果我们以递增顺序逐步执行以这种方式获得的 Morton 码，我们实际上是沿着 3D 中的 Z 阶曲线步进（图右侧显示了 2D 表示）。在实践中，我们可以通过为每个对象分配一个 Morton 码，然后对对象进行相应的排序来确定叶节点的顺序。正如在第一部分的排序和扫描的上下文中提到的，并行基数排序正是这项工作的正确工具。为给定对象分配 Morton 码的一个好方法是使用其边界框的质心点，并相对于场景的边界框表示它。然后可以通过利用整数乘法的[位混合特性](https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/)来有效地执行位的扩展和交织，如下面的代码所示。


```C++
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}
```
在我们包含 12,486 个对象的示例数据集中，在GeForce GTX 690上以这种方式分配 Morton 代码需要0.02毫秒，而对对象进行排序需要0.18毫秒。到目前为止一切顺利，但我们还有一棵树要建。

#### 自上而下的层次结构生成
LBVH 的一大优点是，一旦我们确定了叶节点的顺序，我们就可以将每个内部节点视为它们的线性范围。为了说明这一点，假设我们总共有N个叶节点。根节点包含所有这些，即它覆盖范围[0，N-1]。然后，根的左孩子必须覆盖范围 [0,γ]，以选择适当的 γ，而右孩子必须覆盖范围 [γ+1, N-1]。我们可以一直这样下去，得到下面的递归算法。

```C++
Node* generateHierarchy( unsigned int* sortedMortonCodes,
                         int*          sortedObjectIDs,
                         int           first,
                         int           last)
{
    // Single object => create a leaf node.

    if (first == last)
        return new LeafNode(&sortedObjectIDs[first]);

    // Determine where to split the range.

    int split = findSplit(sortedMortonCodes, first, last);

    // Process the resulting sub-ranges recursively.

    Node* childA = generateHierarchy(sortedMortonCodes, sortedObjectIDs,
                                     first, split);
    Node* childB = generateHierarchy(sortedMortonCodes, sortedObjectIDs,
                                     split + 1, last);
    return new InternalNode(childA, childB);
}
```

![fig05-top-down](/img/assets/Collision/fig05-top-down.png)

我们从一个覆盖所有对象的范围开始（first=0，last= N -1），并确定一个合适的位置将范围一分为二（split=γ）。然后，我们对生成的子范围重复相同的操作，并生成一个层次结构，其中每个这样的拆分对应一个内部节点。当我们遇到一个只包含一个项目的范围时，递归终止，在这种情况下，我们创建一个叶节点。剩下的唯一问题是如何选择 γ。LBVH 根据给定范围内的Morton码之间的最高位不同来确定 γ。换句话说，我们的目标是对对象进行分区，以使 中的所有对象的最高差异位为零childA，而 中的所有对象则为 1 childB。这是一个好主意的直观原因是按其 Morton 码中的最高不同位对对象进行分区对应于在 3D 中轴对齐平面的任一侧对它们进行分类。实际上，找出最高位变化的最有效方法是使用二进制搜索。这个想法是保持对位置的当前最佳猜测，并尝试以指数递减的步骤推进它。在每一步，我们都会检查提议的新职位是否会违反 的要求childA，并相应地接受或拒绝它。以下代码说明了这一点，该代码使用 __clz() NVIDIA Fermi 和 Kepler GPU 中可用的内在函数来计算 32 位整数中前导零位的数量。

```C++
int findSplit( unsigned int* sortedMortonCodes,
               int           first,
               int           last)
{
    // Identical Morton codes => split the range in the middle.

    unsigned int firstCode = sortedMortonCodes[first];
    unsigned int lastCode = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int commonPrefix = __clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            unsigned int splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}
```

我们应该如何并行化这种递归算法？一种方法是使用[Garanzha 等人提出的方法](https://scholar.google.fi/scholar?q=Simpler+and+faster+HLBVH+with+work+queues)。，它从根开始依次处理节点的级别。这个想法是以广度优先顺序维护不断增长的节点数组，以便层次结构中的每个级别对应于节点的线性范围。在给定的级别上，我们为落入此范围的每个节点启动一个线程。线程首先从节点数组中读取first和调用lastfindSplit(). 然后，它使用原子计数器将生成的子节点附加到同一节点数组，并写出它们相应的子范围。这个过程迭代，以便每个级别输出包含在下一个级别上的节点，然后在下一轮中进行处理。

#### Occupancy
当有数百万个对象时，刚刚描述的算法（Garanzha 等人）速度惊人。该算法将大部分执行时间花在树的底层，其中包含足够多的工作来充分利用 GPU。由于线程正在访问 Morton 代码数组的不同部分，因此在更高级别存在一定数量的数据分歧。但考虑到整体执行时间，这些级别也不太重要，因为它们开始时不包含那么多节点。然而，在我们的例子中，只有 12K 个对象（回想一下上一篇文章中看到的示例）。请注意，这实际上少于填充 GTX 690 所需的线程数，即使我们能够完美地并行化所有内容。GTX 690 是双 GPU 卡，两个 GPU 中的每一个都可以并行运行多达 16K 线程。即使我们只使用其中一个 GPU（例如，另一个可以在我们进行物理处理时处理渲染），我们仍然面临并行度不足的危险。

自顶向下算法处理我们的工作负载需要1.04毫秒，这是所有其他处理步骤所用总时间的两倍多。为了解释这一点，我们需要考虑除散度之外的另一个指标：占用率。占用率是相对于处理器理论上可以支持的最大线程数而言，在任何给定时间平均执行多少线程的度量。当占用率低时，它直接转化为性能：将占用率降低一半将使性能降低 2 倍。随着活动线程数量的增加，这种依赖性会逐渐减弱。原因是当占用率足够高时，整体性能开始受到其他因素的限制，例如指令吞吐量和内存带宽。

为了说明，考虑 12K 对象和 16K 线程的情况。如果我们为每个对象启动一个线程，我们的占用率最多为 75%。有点低，但绝不是灾难性的。自上而下的层次结构生成算法与此相比如何？第一层只有一个节点，所以我们只启动一个线程。这意味着第一层将以 0.006% 的占用率运行！第二级有两个节点，因此它以 0.013% 的占用率运行。假设一个平衡的层次结构，第三级以 0.025% 运行，第四级以 0.05% 运行。只有到了13层，才有希望达到25%的合理入住率。但在那之后，我们就已经没有工作了。这些数字有点令人沮丧——由于入住率低，第一层的成本大约与第 13 层一样多。

#### 完全并行的层次结构生成
如果不以某种方式从根本上改变算法，就无法避免这个问题。即使我们的 GPU 支持动态并行（如 NVIDIA Tesla K20 那样），我们也无法避免每个节点都依赖于其父节点的结果这一事实。我们必须在知道它的孩子覆盖哪些范围之前完成对根的处理，我们甚至不能希望在我们这样做之前开始处理它们。换句话说，无论我们如何实现自上而下的层次结构生成，第一层都注定要以 0.006% 的占用率运行。有没有办法打破节点之间的依赖关系？

事实上，我最近在 High Performance Graphics 2012 上展示了该解决方案（论文、幻灯片）。这个想法是以一种非常具体的方式对内部节点进行编号，这使我们能够找出任何给定节点对应的对象范围，而无需了解树的其余部分。利用任何具有N个叶节点的二叉树总是恰好有N-1 个内部节点这一事实，我们可以生成整个层次结构，如下面的伪代码所示。

```C++
Node* generateHierarchy( unsigned int* sortedMortonCodes,
                         int*          sortedObjectIDs,
                         int           numObjects)
{
    LeafNode* leafNodes = new LeafNode[numObjects];
    InternalNode* internalNodes = new InternalNode[numObjects - 1];

    // Construct leaf nodes.
    // Note: This step can be avoided by storing
    // the tree in a slightly different way.

    for (int idx = 0; idx < numObjects; idx++) // in parallel
        leafNodes[idx].objectID = sortedObjectIDs[idx];

    // Construct internal nodes.

    for (int idx = 0; idx < numObjects - 1; idx++) // in parallel
    {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)

        int2 range = determineRange(sortedMortonCodes, numObjects, idx);
        int first = range.x;
        int last = range.y;

        // Determine where to split the range.

        int split = findSplit(sortedMortonCodes, first, last);

        // Select childA.

        Node* childA;
        if (split == first)
            childA = &leafNodes[split];
        else
            childA = &internalNodes[split];

        // Select childB.

        Node* childB;
        if (split + 1 == last)
            childB = &leafNodes[split + 1];
        else
            childB = &internalNodes[split + 1];

        // Record parent-child relationships.

        internalNodes[idx].childA = childA;
        internalNodes[idx].childB = childB;
        childA->parent = &internalNodes[idx];
        childB->parent = &internalNodes[idx];
    }

    // Node 0 is the root.

    return &internalNodes[0];
}
```

```
__device__
int2 determineRange(unsigned int* sortedMortonCodes, int numTriangles, int idx)
{
   //determine the range of keys covered by each internal node (as well as its children)
    //direction is found by looking at the neighboring keys ki-1 , ki , ki+1
    //the index is either the beginning of the range or the end of the range
    int direction = 0;
    int common_prefix_with_left = 0;
    int common_prefix_with_right = 0;

    common_prefix_with_right = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx + 1]);
    if(idx == 0){
        common_prefix_with_left = -1;
    }
    else
    {
        common_prefix_with_left = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - 1]);

    }

    direction = ( (common_prefix_with_right - common_prefix_with_left) > 0 ) ? 1 : -1;
    int min_prefix_range = 0;

    if(idx == 0)
    {
        min_prefix_range = -1;

    }
    else
    {
        min_prefix_range = __clz(sortedMortonCodes[idx] ^ sortedMortonCodes[idx - direction]); 
    }

    int lmax = 2;
    int next_key = idx + lmax*direction;

    while((next_key >= 0) && (next_key <  numTriangles) && (__clz(sortedMortonCodes[idx] ^ sortedMortonCodes[next_key]) > min_prefix_range))
    {
        lmax *= 2;
        next_key = idx + lmax*direction;
    }
    //find the other end using binary search
    unsigned int l = 0;

    do
    {
        lmax = (lmax + 1) >> 1; // exponential decrease
        int new_val = idx + (l + lmax)*direction ; 

        if(new_val >= 0 && new_val < numTriangles )
        {
            unsigned int Code = sortedMortonCodes[new_val];
            int Prefix = __clz(sortedMortonCodes[idx] ^ Code);
            if (Prefix > min_prefix_range)
                l = l + lmax;
        }
    }
    while (lmax > 1);

    int j = idx + l*direction;

    int left = 0 ; 
    int right = 0;
    
    if(idx < j){
        left = idx;
        right = j;
    }
    else
    {
        left = j;
        right = idx;
    }

    printf("idx : (%d) returning range (%d, %d) \n" , idx , left, right);

    return make_int2(left,right);
}

```

该算法简单地分配一个由N -1 个内部节点组成的数组，然后并行处理所有这些节点。每个线程首先确定其节点对应的对象范围，有点神奇，然后像往常一样继续分割范围。最后，它根据各自的子范围为节点选择子节点。如果子范围只有一个对象，则子范围必须是叶子，因此我们直接引用相应的叶子节点。否则，我们从数组中引用另一个内部节点。

![fig06-numbering](/img/assets/Collision/fig06-numbering.png)

内部节点的编号方式从伪代码中已经很明显了。根的索引为 0，每个节点的子节点位于其拆分位置的任一侧。由于排序后的 Morton 码的一些不错的特性，这种编号方案永远不会导致任何重复或间隙。此外，事实证明，我们可以determineRange()以findSplit()与 有关其工作原理和原因的更多详细信息，请参阅[论文](https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf)。

该算法与递归自顶向下方法相比如何？它显然执行了更多的工作——我们现在每个节点需要三个二进制搜索，而自上而下的方法只需要一个。但它完全并行完成所有工作，在我们的示例中达到 75% 的全部占用率，这产生了巨大的差异。并行层次结构生成的执行时间仅为0.02毫秒，比自顶向下算法提高了 50 倍！

您可能会认为，当对象数量足够多时，自上而下的算法应该开始获胜，因为缺乏占用不再是问题。然而，实际情况并非如此——并行算法在所有问题规模上始终表现得更好。与往常一样，对此的解释是分歧。在并行算法中，附近的线程总是访问附近的 Morton 代码，而自上而下的算法将访问分散到更广泛的区域。

#### 边界框计算
现在我们已经有了节点的层次结构，剩下要做的就是为每个节点分配一个保守的边界框。我在论文中采用的方法是进行并行自下而上的归约，其中每个线程从单个叶节点开始并走向根节点。为了找到给定节点的边界框，线程只需查找其子节点的边界框并计算它们的并集。为了避免重复工作，这个想法是使用每个节点的原子标志来终止进入它的第一个线程，同时让第二个线程通过。这确保每个节点只被处理一次，而不是在其两个子节点都被处理之前。边界框计算的执行发散度很高——处理一个节点后只有一半的线程保持活动状态，处理两个节点后有四分之一的线程保持活动状态，处理三个节点后的八分之一，依此类推。然而，由于两个原因，这在实践中并不是真正的问题。首先，bounding box计算只需要0.06 ms，与例如对对象进行排序相比，这仍然是相当低的。其次，处理主要是读写bounding box，计算量最小。这意味着执行时间几乎完全由可用内存带宽决定，减少执行分歧并没有太大帮助。

#### 概括
我们在一组 3D 对象之间寻找潜在碰撞的算法包括以下 5 个步骤（时间是针对上一篇文章中使用的 12K 对象场景）。

0.02 ms，每个对象一个线程：计算边界框并分配 Morton 代码。
0.18 ms，并行基数排序：根据对象的 Morton 代码对对象进行排序。
0.02 ms，每个内部节点一个线程：生成 BVH 节点层次结构。
0.06 ms，每个对象一个线程：通过将层次结构推向根来计算节点边界框​​。
0.25 ms，每个对象一个线程：通过遍历 BVH 查找潜在的冲突。
完整的算法耗时0.53 ms，其中 53% 用于树构建，47% 用于树遍历。

#### 讨论
我们已经在宽相碰撞检测的背景下提出了许多不同复杂度的算法，并确定了在大规模并行处理器上设计和实现它们时的一些最重要的考虑因素。独立遍历和同时遍历之间的比较说明了发散在算法设计中的重要性——最好的单核算法很容易变成并行设置中最差的算法。依赖时间复杂度作为一个好的算法的主要指标有时会产生误导甚至是有害的——如果它有助于减少分歧，做更多的工作实际上可能是有益的。

BVH 层次生成的并行算法提出了另一个有趣的观点。在传统意义上，算法是完全没有意义的——在单核处理器上，节点之间的依赖关系一开始就不是问题，每个节点做更多的工作只会让算法运行得更慢。这表明并行编程确实与传统的单核编程有根本的不同：​​与其说是将现有算法移植到并行处理器上运行，不如说是在并行处理器上运行。它是关于重新思考我们通常认为理所当然的一些事情，并提出专门设计的新算法，考虑到大规模并行性。

---
If you like it, don't forget to give me a star.

[![Star This Project](/img/assets/github.svg)](https://github.com/fwzhuang/fwzhuang.github.io)