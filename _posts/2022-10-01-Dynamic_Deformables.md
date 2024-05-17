---
layout: post
title: Dynamic Deformables:Implementationand Production Practicalities (Now With Code!)
categories: [Simulation]
description: 翻译<Dynamic Deformables:Implementationand Production Practicalities>
keywords: Sim, CG
mermaid: false
sequence: false
flow: false
mathjax: true
mindmap: false
mindmap2: false
---

本文整理机器翻译《[Dynamic Deformables:Implementationand Production Practicalities (Now With Code!)](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf)》，用以学习之用。
<!--more-->

##  摘要
自从Boo在Monsters,Inc.(2001)中穿上衬衫以来，模拟动态变形一直是Pixar讲故事的一个组成部分。最近，皮克斯的核心模拟器Fizt应用了几项关键转换，以提高其速度、鲁棒性和通用性。从Coco(2017)开始，改进的碰撞检测和响应被整合到布料求解器中，然后在Cars3(2017)中引入
了3D刚体，在Onward(2020)中，允许服装与角色的身体双向交互耦合。

3D刚体是基于我们过去几年在SIGGRAPH上发布的快速、紧凑和强大的新公式。在这个公式下，力梯度的构造和特征分解，长期以来被认为是实现中最繁重的部分，变得快速而简单。我们在这里提供了技术论文中没有的详细、独立和统一的处理。我们还首次提供了许多上述算法的开源C++实现。


这种新的公式只是创建能够应对生产环境挑战的模拟器的起点。一个挑战是性能：我们讨论了我们当前加速系统组装和求解器性能的最佳实践。另一个需要相当关注的挑战是鲁棒的碰撞检测和响应。已经写了很多关于碰撞检测方法的文章，例如邻近查询、连续碰撞和全局交叉点分析。我们讨论了使用这些技术的策略，这些技术为我们提供了处理具有挑战性的场景所需的有价值的信息。

# 第一章 介绍
## 1.1 本课程的目标
自2001年Monsters,Inc.以来，Fizt1一直是Pixar的核心模拟器。虽然它所基于的经过实战考验的Baraff和Witkin(1998)模型的许多基础知识多年来保持不变，但还引入了各种关键改进和提升。其中许多改进已在过去两年在SIGGRAPH上发表的技术论文和演讲中传播：
*  Stable Neo-Hookean Flesh Simulation, Breannan Smith, Fernando de Goes, Theodore Kim(Technical Paperpresentedat SIGGRAPH2018)
* Better Collisions and Faster Cloth for Pixar’s Coco, David Eberle (Talk presented at SIGGRAPH 2018)
* Clean Cloth Inputs: Removing Character Self-Intersections with Volume Simulation,Audrey Wong, David Eberle,Theodore Kim (Talk presentedatSIGGRAPH2018)
*  RobustSkinSimulationinIncredibles2,RyanKautzman,GordonCameron,Theodore Kim(Talk presentedatSIGGRAPH 2018)
* Analytic Eigensystems for Isotropic Distortion Energies, Breannan Smith, Fernando de Goes,TheodoreKim(Technical Paper presented at SIGGRAPH2019)
* AnisotropicElasticityforInversion-SafetyandElementRehabilitation,TheodoreKim, Fernando deGoes,HayleyIben (Technical PaperpresentedatSIGGRAPH2019)

这些改进共同构成了模拟器的下一个主要版本，内部称为Fizt2。本课程的目标之一是以一种以前没有任何一篇论文或演讲能够做到的方式对这些发展进行统一的概述。


除了这些之外，我们还根据其中一位作者在过去十年中教授的一门大学课程，对变形力学的基础知识进行了新的介绍（第2章）。这种处理特别侧重于引入特定的基于张量的公式，该公式用于获得上述技术论文中的结果。Matlab和C++实现的几段微妙的代码也被内联，例如四面体的矩阵$\frac{\partial{F}}{\partial{x}}$（图E.1）和六面体的形状导数矩阵（图D.3）

## 1.2 现在有代码了
本着Pharr(2016)等人的pbrt渲染器的精神。我们提供了HOBAK，这些注释中描述的许多算法的原型C++实现。就像pbrt不是RenderMan一样，HOBAK也不是Fizt。它是完全独立编写的，旨在用作研究测试平台、参考实现和教学工具。每当我们必须在代码清晰度和运行时性能之间做出选择时，我们都会选择清晰度。
HOBAK不是生产系统。因此，它几乎没有依赖关系，应该立即构建和运行。如果您想直接进入C++，请参阅附录J中的入门指南

## 1.3 作者传记
Theodore Kim是耶鲁大学计算机科学副教授，此前，他是皮克斯动画工作室的高级研究科学家，在那里他设计并实现了这里描述的许多固体力学算法。他拥有康奈尔大学计算机科学学士学位以及硕士学位和博士学位。来自北卡罗来纳大学教堂山分校。2012年，他因其在流体模拟方面的工作获得了科学技术学院奖。
David Eberle是皮克斯动画工作室的模拟工具研发主管。他曾在Disney、Havok、 PDI/Dreamworks和Pixar开发过许多用于动画和视觉效果的模拟器。他拥有南卡罗来纳大学的数学学士学位和德克萨斯 A&M 大学的数学硕士学位。他的电影作品包括：《火之王朝》、《史瑞克2》、 《史瑞克3》、 《功夫熊猫》、《马达加斯加》、《马达加斯加2》、《超级头脑》、 《内翻》、《海底总动员》、《汽车总动员3》、《可可》、《前进》、 《灵魂》、《卢卡》、 《变红》和《光年》。

## 1.4 致谢
这些笔记是一个正在进行的草稿，因此作者要感谢以下人员在文本中发现错误和遗漏：JC Chang、Towaki Takikawa、Peng Yu、Kai Jia、Kirill Klimov、Samaksh Ajay (Avi) Goyal，特别是须田进次郎。如果您发现其他错误（肯定有很多），请随时与作者联系。


# 第二章 变形基础
## 2.1 我需要阅读本章吗
如果您是变形模拟的新手，或者一段时间后又重新开始使用，答案是：是的。

现有几种针对计算机图形学的变形介绍，包括经典的Witkin和Baraff(1997)以及更现代的Sifakis和Barbic(2012)或Bargteil和Shinar(2018)。如果您是已经熟悉这些材料的从业者，您可能可以跳过本章的大部分内容。需要注意的是，我们稍后展示的许多快速、紧凑的结构将取决于Golub和 Van Loan(2013)的高阶张量表示法。这在计算机图形学中并不经常出现，因此您可能仍想阅读第3章。

## 2.2 我们在谈论什么样的挤压？
![F2-1](/images/posts/Sim/F2_1.png)

我们将研究一种特定形式的固体变形，称为超弹性变形。假设我们拿一只兔子并用100磅的重量将它压扁（图2.1）。当我们移除重量时，它会恢复到原来的形状。这就是超弹性的本质：当所有的外力都被移除时，超弹性物体会恢复到原来的形状。兔子被拉长到原来长度的一百万倍，或者被压成薄饼都没有关系。移除力后，它会弹回原来的兔子形状。

因此，原始形状被认为是非常特殊的，因为它是对象“记住”并且总是“尝试”返回的形状，并被分配了一个特殊的名称：**rest shape**。事实上，当我们用100磅重压扁一只兔子时，它会呈现出在当前条件下它可以保持的最佳形状或最接近其他形状的形状。

但我们所说的最好或最接近是什么意思？与什么相比最好，还是与什么最接近？这真的是两个问题。
1. 我受到怎样的挤压？或者，我离原来的形状还有多远？或者：与我原来的形状相比，我的变形有多严重？最具体地说：我可以计算出一个定量的变形分数，它可以准确地告诉我相对于我的原始形状，我的拉伸或挤压程度如何？
2. 我应该推回多少？一旦我知道我到底有多畸形，我应该如何处理这些知识？真的变形算不算特别坏，所以我应该加倍努力才能恢复原来的形状？

让我们依次看看这些问题。

## 2.3 我受到怎样的挤压
首先，让我们看一个明显的测量变形的方法，它会被证明是错误的。从那里，我们可以思考哪里出了问题并想出更好的办法。
### 2.3.1测量错误的方法

![F2_2](/images/posts/Sim/F2_2.png)

让我们看一个三角形, 我们将使用上横线将其在**rest shape** _(材料空间)_ 中的顶点表示为$\overline{x}_0$,$\overline{x}_1$, $\overline{x}_2$。在三角形从其原始形状被压
扁后，我们将其**deformed shape** _（世界空间）_ 表示为$x_0$, $x_1$, $x_2$。计算我们变形程度的分数的一种明显方法是
$$\begin{align}\Psi_{wrong} = \sum_{i=0}^{2}\parallel\overline{x}_i - x_i \parallel_2 \tag{2.1} \end{align} $$

在这里，我们使用$\parallel\cdot\parallel_2$来表示2‑norm。如果您以前没有见过这种符号，请不要担心；我们在附录C.1.1中对其进行了描述。它是表示“距离”的通用简写。我们还将可互换地将评分函数称为能量函数。对于我们的目的，评分和能量函数具有完全相同的含义。

评分函数$\Psi_{wrong}$，看起来当然是合理的。当您向下挤压$x_1$或向上拉伸更多时，分数（能量）肯定会增加。当没有发生变形时，我们绝对希望score函数等于0，而当发生大量变形时，我们希望分数函数变成一个大数。如果这就是所需要的，$\Psi_{wrong}$似乎可以完成工作.

![F2_3](/images/posts/Sim/F2_3.png)

是吗？问题是我们不会直观地考虑变形的一堆东西也被归为分数。在图2.3中，我们将三角形平移到右上角一小段距离。如果我们在这个新三角形上使用$\Psi_{wrong}$，那么它将显示为非零分数.

然而，根据任何合理的定义，这个三角形并没有变形。相反，它已被移动或平移。图2.4显示了另一个问题案例。如果我们旋转三角形，那么$\Psi_{wrong}$将返回一个非零分数，错误地将这个三角形视为“变形”。但是，根据任何合理的定义，三角形已经旋转。它没有变形。

![F2_4](/images/posts/Sim/F2_4.png)

我们想要的是一个平移和旋转不变的评分函数$\Psi$。如果一个三角形仅仅被平移或旋转，它的变形分数应该显示为零。

### 2.3.2  移除平移（简单）
编码一个三角形所能经历的变换的一种方法是使用Aﬃne映射
$$\begin{aligned} \phi(\overline{x}) &= F\overline{x} + t \\
&= x  \end{aligned} \tag{2.2}$$



这里 在2D空间中$t\in \Re^2$, $F \in \Re^{2\times2}$ ，在3D空间中$F\in \Re^{3\times3} $ 参见附录A中的符号表。仿射变换只能编码一些东西：
* 旋转
* 缩放
* 反射
* 平移

我们看到旋转和平移不断出现并污染我们的变形分数，尽管我们不直观地将它们视为“变形”。然而，缩放与我们的变形概念非常吻合。我们可以隔离并衡量它吗？

作为第一步，让我们看看我们是否可以将平移切掉，这相对容易。平移完全存在于t中，所以如果我们执行一些操作来删除该组件，我们就完成了。这样做的规范方法是对其**rest shape**求导:
$$\begin{aligned} \frac{\partial \phi(\overline{x})}{\partial\overline{x}} &= \frac {\partial}{\partial\overline{x}}(F\overline{x} + t) \\ 
&= F \end{aligned} \tag{2.3}$$

由于我们取的是仿射映射（变形）的导数（梯度），所以矩阵$F$称为形变梯度。该矩阵现在包含旋转、缩放和反射，但不包含平移。如果我们将某种变形分数基于$F$，那么平移现在就不可能污染结果。

### 2.3.3 形变梯度：非常重要
很难夸大形变梯度的重要性。我们在本章其余部分所做的几乎所有事情都将处理$F$。变量有一个笨拙的名称，而且每个人（Bonet和Wood（2008）；Bower（2009）；Marsden和Hughes（1994）；Belytschk等人（2013）))使用$F$来表示它，即使“变形”和“梯度”都不是以“f”开头的。理想情况下，变量应该有一个时髦的名称和一个直观的符号，但这不是我们生活的世界。所以请记住$F$很重要。
您可能会问自己：如果$F$如此重要，您会告诉我如何计算它吗？绝对地。它有点复杂，而且占用一点篇幅，所以它在附录D中有自己的部分。理所当然地我们知道如何计算这个数量，让我们来看看它的含义。

### 2.3.4 去除旋转（不是那么容易）
我们现在有了$F$，其中平移已被仁慈地减去，但缩放、旋转和反射仍然纠缠不清。我们怎样才能消除旋转？这会比删除平移更困难，而且实际上会有不止一个答案。

让我们尝试用$F$构建一个简单的得分函数。将矩阵归结为标量得分的最简单方法是取其Frobenius范数的平方，记为$\parallel \cdot \parallel F$。与第2.3.1节中的2-norm一样，如果您不熟悉这个范数，请不要担心。我们在附录C.1.2中对其进行了详细记录，但由于这是第一次出现，我们将明确地编写它出来。
$$
\begin{aligned}
\Psi_{Dirichlet}  &= \parallel F \parallel^2 _F \\
 &= \sum_{i=0}^{2} \sum_{j=0}^{2}f^2 _{ij} \tag{2.4}
\end{aligned}
$$
我们对$F$的条目进行了编号:
$$
F = \left[
    \begin{matrix}
    f_{00} & f_{01} & f_{02} \\
    f_{10} & f_{11} & f_{12} \\
    f_{20} & f_{21} & f_{22} \\
    \end{matrix}
\right] \tag{2.5}
$$
这个分数函数也被称为狄利克雷能量，它不能完成工作。请记住，当变形为零时，我们希望变形分数返回零。然而，当F的所有项都为零时，狄利克雷能量将返回零，即$F=0$。
如果我们将$F=0$代入我们的原始变形函数$\phi(\overline x)=F\overline x+t$，它对应于所有点$\overline x$被压碎成以t为中心的零体积黑洞的情况。作为一个评分函数，狄利克雷能量可能是一个很棒的黑洞探测器，但它是一个糟糕的形变度量。
#### 2.3.4.1第一次尝试（错误）
当$F=0$时，狄利克雷能量返回零。在零变形下返回零的能量（分数）看起来像什么？好吧，如果$F=I$，那么肯定不会发生变形。在这种情况下，$\phi(\overline x)= F \overline {x} + t = I \overline{x} +t = \overline{x} + t$, 这单纯是的一个平移。

我们可以设计一个当$F=I$时返回零的函数吗？好吧，如果$F=I$，那么$\parallel F \parallel ^2_F = 3$那么，我们来尝试分数函数
$$ \Psi_{Neo-Hookean} = \parallel F \parallel ^2_F -3 \tag{2.6}$$

这不是一个好主意。虽然当$F=I$时，分数确实为零，但我们也希望零是可能的最小分数。否则，分数变得难以解释。零分意味着零变形，大于零的分数意味着一些变形，但小于零的分数意味着什么？小于零变形正在发生吗？
在$\Psi_{Neo‑Hookean}$的情况下，如果F=0，则$\Psi_{Neo‑Hookean} = -3$。所以，这是一个非常糟
糕的变形度量。如果情况F定义如下，则变得更糟
$$
F = \left[
    \begin{matrix}
    0 & 0 & 0 \\
    0 & \sqrt \frac{3}{2} & 0 \\
    0 & 0 & \sqrt \frac{3}{2} \\
    \end{matrix}
\right] \tag{2.7}
$$
这里指的是Wile‑E‑Coyote式的pancake state，绝对不是零变形状态。然而，$\Psi_{Neo‑Hookean}$分数返回零。显然，有许多鬼鬼祟祟的配置可以使这个评分函数误以为没有发生变形。我们可以尝试应用Band-Aid，例如$\Psi_{Neo‑Hookean,Band-Aid} = (\parallel F \parallel ^2_F-3)^2$, 但这只会解决负分数问题。分数对于同样鬼鬼祟祟的形变状态，仍然返回零。

然而，由于某种原因，它有一个花哨的名字“Neo‑Hookean”，所以它一定很重要。在过去的80年里，它在机械工程和材料科学中得到了广泛的研究（Mooney(1940);Rivlin(1948)），因为我们稍后会看到，如果你添加一些重要的组件，它就会成为一个非常有趣的变形测量。但是，它目前的形式并不是很好，所以我们暂时搁置它。

#### 2.3.4.2第二次尝试（仍然有误）
接下来，让我们尝试一些反复无常且看起来很愚蠢的东西（C&SL），我保证它最终会走向某个地方。我们让$F$和$I$相减并取其范数。然后，我们至少可以保证当$F=I$时，我们的评分函数将为零：
$$\Psi_{C\&SL} = \parallel F-I \parallel ^2_F \tag{2.8}$$
此外，由于$I$在范数内，所有内容都会平方，因此我们的函数不可能产生无意义的负分。我们可以使用附录B中的恒等式B.16将其扩展为：
$$\begin{aligned}
\Psi_{C\&SL} & = \parallel F-I \parallel ^2_F  \\ &=  \parallel F \parallel ^2_F + \parallel I \parallel - 2 tr F \\
&= \parallel F \parallel ^2_F + 3 - 2trF
\end{aligned} 
$$
我假设我们正在考虑3D情形，这就是$\parallel I \parallel ^2_F=3$的原因。查看展开，分数仍然不是很好。第一项是的黑洞偏爱的狄利克雷能量，它对我们没有任何好处。然后，我们有常数3，这看起来与我们不太好的Neo-Hookean能量有点相似，然后是一个新的$tr F$项。如果您有一段时间没有看到矩阵的迹，请参阅附录C.2中的复习。由于这是它第一次出现在这里，我们将明
确地写出来：
$$tr F = \sum_{i=0}^2f_{ii} = f_{00} + f_{11} + f_{22}  \tag{2.9}$$
当$F = I$时，这个评分函数将为零，但当$F\neq I$时，会发生奇怪的事情。例如，如果$F$恰好是某个旋转矩阵，
$$
F = \left[
    \begin{matrix}
    1 & 0 & 0 \\
    0 & cos\theta & -sin\theta \\
    0 & sin\theta & cos\theta \\
    \end{matrix}
\right] \tag{2.10}
$$
那么当θ不为零或2π的某个倍数时，$\Psi_{C\&SL}$将返回一个非零分数。但是，这些仍然明显对应于无变形状态！它比黑洞探测器要好，但也好不了多少。

总的来说，奇怪的是$trF$完全出现在这个公式中。由于$F$可以是任意的非对称矩阵，因此在过于特殊的情况$F=I$, 在抵消$\parallel F \parallel ^2_F +3$之外，它的迹线是否非常有意义尚不清楚。

#### 2.3.4.3第三次尝试：St.Venant Kirchhoff
当$F$是纯旋转时，我们如何制作一个版本的$\Psi_{C\&SL}$，而不是等于0？一种方法是利用一个好的线性代数恒等式。如果$F$是一个纯旋转，那么我们知道它一定是$F^TF=I$。当$F$是正交时，$F^{-1}=F^T$。
那么，我们为什么不计算$F^TF$, 减去旋转部分, 例如$I$部分,看看还剩下什么？这些残留物可能以某种合理的方式表征了变形。我们可以把它写成：
$$ \Psi_{StVK,stretch} = \frac{1}{2} \parallel F^TF -I \parallel ^2_F  \tag{2.11}$$
这是个好主意！这个想法足够好，人们给它起了一个特殊的名字：St.Venant‑Kirchhoff拉伸能量，简称“StVK，stretch”。$F^TF -I$的各自组成部分很受欢迎，并且他们有特殊的名字跟符号 ：
![F2_4_1](/images/posts/Sim/F2_4_1.png)
使用这些，能量可以写成高度紧凑的形式:
$$ \Phi_{StVK,stretch} = \parallel E \parallel ^2_F \tag{2.12}$$
同样，这是一种度量形变的好方法。我们正在使用$F$, 平移已经减去了，我们使用$F^TF$恒等式分解旋转。我们使用$F$会抱怨一件事，那就是评分函数是非线性的，执行与$\Psi_{C\&SL}$相同的展开，我们可以得到
$$ \begin{aligned}
\Psi_{StVK,stretch} &= \parallel F^TF-I \parallel ^2_F \\
&= \parallel F^TF \parallel ^2_F + trI - 2tr(F^TF) 
\end{aligned} \tag{2.13}
$$
第一项，$\parallel F^TF\parallel$,是4阶的，或者是$F$的4次方，$F$的平方通过$F^TF$，再取范数$\parallel \cdot \parallel ^2_F$, 再一次平方，所以整项是4阶的。我们将在后面看到，这个四阶性增加了额外的计算复杂性，这实际上是不可取的。

#### 2.3.4.4 第四次尝试：As-Rigid-As-Possible
我们利用了$F^TF=I$等式的特性，但是我们可以使用其他特性吗？如果我们特别关心如何区分矩阵的旋转和非旋转分量，极坐标分解会很有帮助：
$$F = RS \tag{2.14}$$
这里，矩阵$R$是$F$的旋转部分，而$S$是$F$的非旋转（缩放）部分。这种分解似乎正是为了解决我们感兴趣的问题而定制的。
然而，你在使用极坐标分解时需要小心一点，因为其经典定义与我们想要的版本略有不同。通常$R$只被定义为酉矩阵，而不是旋转矩阵，所以它只需要满足$R^TR=I$。这意味着在$R$的某个地方可能潜伏着一个反射，而我们希望$R$是一个纯旋转。在我们的例子中，如果反射必须潜伏在某个地方，我们更希望它潜伏在$S$中。我们将把这种特殊的风味称为极坐标分解的旋转变体。关于如何计算的细节，见附录F。

简而言之：如果你最终在某个库中调用了极坐标分解代码，请确保它是在计算旋转变体，而不是传统版本。如果不是这样，你需要按照图F.1和F.2的风格为库的调用写一个小的包装器。

说完这些，让我们假设你能计算出旋转变体的极坐标分解，并且现在有了$R$和$S$在手。你能用它们做什么？在StVK中，我们把$F^TF$，然后减去它的旋转分量$I$。好了，现在我们有
一个不同的旋转表示，$R$，但我们能做同样的事情吗？从$F$中减去旋转分量，并假设剩下的部分必须以某种方式描述变形？

最基本的，天哪，我想知道这是否会成功的写法是:
$$\Psi_{ARAP}=\parallel F-R \parallel ^2_F \tag{2.15}$$
这种公式有几个看起来令人担忧的地方。通常，当你有两个矩阵时，将它们相乘是首先要尝试的事情，因为你可以通过偷看SVD或特征值分解看到乘法的真正作用。但是，我们不是在做乘法，而是在做减法。我们真的可以把两个看起来很随意的矩阵互相相减，然后得到一些不是完全垃圾的东西吗？它们是通过极坐标分解联系在一起的，但这真的意味着我们可以开始从对方身上减去这些条目吗？

令人惊讶的是，答案是肯定的。事实上，这个能量函数是如此不可或缺，以至于它在几何处理中被赋予了一个特殊的名字：As-Rigid-As-Possible（ARAP）能量（Sorkine和Alexa（2007））。我们稍后会对这个能量做更多的分析，看看它实际上是如何比最初看起来更聪明的。

这种能量不存在StVK的 "过度非线性 "问题。如果我们应用特性B.16，我们得到"
$$ 
\begin{aligned}
\Psi_{ARAP} &= \parallel F-R \parallel ^2_F \\
&= \parallel F \parallel ^2_F + \parallel R \parallel ^2_F - 2tr(F^TR) \\
&= \parallel F \parallel ^2_F + 3 - 2 tr(S)
\end{aligned} \tag{2.16}
$$

在$F$中最多是二次的能量，我们将在后面看到，$R$和$S$的引入带来了一些计算上的挑战，但就目前而言，不可否认的是$\Psi_{ARAP}$ 的非线性比$\Psi_{StVK,stretch}$要小得多，很有吸引力。

在物理模拟中，ARAP能量实际上是流行的co-rotational能量中的第一个项。这种能量至少从20世纪80年代起就为机械工程所熟知（Rankin and Brogan (1986)），ARAP能量对应的是泊松比被设置为零的特定机械情况。这种能量在21世纪初被几个图形学研究者重新发现（Müller等人（2002）；Etzmuss等人（2003）；Irving等人（2004）），这造成了一些混乱，因为他们各自给它起了不同的名字，如 "刚度扭曲 "或 "旋转线性"。然而，这些模型都是等效的，现在大家似乎都把 "co-rotational "作为这种能量的名称了。

#### 2.3.4.5 能量（评分）功能摘要
我们现在已经看到了一堆不同的能量函数。让我们总结一下它们的优点和缺点。

能量 | 优点 | 弊端
---------|----------|---------
 Dirichlet | 不多。一个好的黑人洞口检测器？ | 不能有效地测量变形。
 Neo-Hookean | 当F=I时返回0 | 对于其他配置也会返回0，也可以返回负数的分数。
 C&SL | 当F=I时返回0，并且不返回否定的分数。 | 在纯旋转下不返回零。
StVK | 当F是一个纯旋转时，返回0，否则返回合理的分数| 过度的非线性（四次能量）
ARAP  |当F是一个旋转时返回0，否则返回合理的分数，并且是二次的 | 这个R术语会带来麻烦

正如我在顶部所说的，从分数中去除旋转并不容易，而且有不止一种策略。现在我们有
了一堆选择，我们应该怎么做呢？

## 2.4 力的计算。我应该向后推多少？
现在我们可以测量我们的变形程度了，我们可以回到第2.2节的第二个问题：我应该向后推多少？

变形措施直接产生了力。令人惊讶的是!当我们糊里糊涂地用不同的方法来测量变形时，我们实际上是在尝试不同的弹性能量势。就我们的目的而言，这两者是完全等同的。变形测量直接决定了力的反应。如果你想设计一个新的材料模型，就想出一个不同的方法来测量变形。

为了得到二维空间三角形节点上的力$f$，我们把三角形的顶点（x0 , x1 和 x2 ）堆积成一个大的矢量。

$$ 
x = 
\left[
\begin{matrix}
\bold x_0 \\
\bold x_1 \\
\bold x_2
\end{matrix}
\right]   = \left[
\begin{matrix}
x_0 \\
x_1 \\
y_0  \\
y_1 \\
z_0 \\
z_1 \\
\end{matrix}
\right] \in \Re^6 \tag{2.17}
$$
然后再取梯度。
$$ f = -a \frac{\partial \Psi }{\partial \bold{x}} \tag{2.18}$$

这里，a是原三角形在静止时的面积。一个真正微小的、只有蚂蚁大小的变形三角形是不会施加与帝国大厦大小的变形三角形相同的力的，所以我们应该把它放大。

这一切都很好，公式2.18看起来也很简单，除了我们在§2.3.4.5中看的所有能量都是用$F$而不是$x$写的。事实上，我们对$F$没有$x$的这些垃圾测量污染属性做了大文章，比如平移，而且它显然是更好的主变量。在走了这条路之后，我们现在应该如何取得与$x$有关的梯度呢？

### 2.4.1  计算 $\frac {\partial \Psi} {\partial \bold x} $

我们把$F$的计算推导放到了附录D，但我们要把这个讨论的最终结果拉回到这里。对于一个三角形，我们把它原来的$\overline{\bold x}_i$ 和变形的$\bold x_i$ 顶点打包成两个矩阵，$D_m$ 和 $D_s$ ，这样:
$$
\begin{aligned}
D_m &= \left[ \overline x_i - \overline x_0 | \overline x_2 - \overline x_0 \right] \\
D_s &= \left[  x_i -  x_0 |  x_2 -  x_0 \right]
\end{aligned} \tag{2.19}
$$
然后，变形梯度是这两者的组合。
$$F = D_sD^{-1}_m \tag{2.20}$$

首先是好消息：我们需要关于$x$的梯度，而不是$\overline x$。 如果我们反而需要$\frac {\partial \Psi}{\partial \overline x}$事情会变得非常难看，因为$\overline {x}_i $ 出现在$D_m$ ，这是在一个逆的内部,$D_m^{-1}$，所以我们需要弄清楚矩阵逆的导数。

现在，坏消息是：它仍然很难看。让我们以一个非常简单的能量，$\Psi_{Dirichlet}$为例:
$$ \Psi_{Dirichlet} =  \parallel F \parallel ^2_F  \tag {2.21} $$

下面是我们要做的事情。
• 将$F = D_s D^{−1}_m $插入$ \Psi_{Dirichlet}$中。
• 将所有的东西相乘，得到一个巨大的$\parallel F \parallel ^2_F$的单线方程。
• 对这个庞大的方程进行六次导数，对$\bold x$中的每个条目进行一次导数，并将结果堆积成一个力矢量。
• 希望你在推导的某个地方没有犯错。将表达式移植到C++中，希望你在将这些方程从纸上转移到代码中时不会犯错。

### 2.4.2 不要这样做

让我们再凝视一下这种丑陋的东西，就一分钟。虽然只有一分钟。之后，我必须把目光移开，并告诉你有一个更好的方法。首先，我们可以在$D^{-1}_m$中对每个条目进行标记

$$D^{-1}_m= \left[ \begin{matrix}  m_0 & m_2 \\ m_1 & m_3 \end{matrix} \right]  \tag{2.22}$$

然后我们将尝试乘出矩阵$F^TF$，以期待把所有东西都塞进$\parallel F \parallel ^2_F$, 那么有：

$$
\begin{aligned}
F^TF &= \left[ \begin{matrix} m_0 & m_1 \\ m_2 & m_3 \end{matrix}  \right] \left[ \begin{matrix} x_1 - x_0& x_2 - x_0 \end{matrix}  \right]^T  \left[ \begin{matrix} x_1 - x_0& x_2 - x_0 \end{matrix}  \right] \left[ \begin{matrix} m_0 & m_1 \\ m_2 & m_3 \end{matrix}  \right]  \\
&=\left[ \begin{matrix} m_0 & m_1 \\ m_2 & m_3 \end{matrix}  \right]  \left[ \begin{matrix} (x_1-x_0)^T(x_1-x_0)& (x_2-x_0)^T(x_1-x_0) \\ (x_2-x_0)^T(x_1-x_0) & (x_2-x_0)^T(x_2-x_0) \end{matrix}  \right]   \left[ \begin{matrix} m_0 & m_1 \\ m_2 & m_3 \end{matrix}  \right] 
\end{aligned} \tag{2.23}
$$

为了把事情弄清楚，让我们引入更多的临时变量:

$$
 \left[ \begin{matrix} (x_1-x_0)^T(x_1-x_0)& (x_2-x_0)^T(x_1-x_0) \\ (x_2-x_0)^T(x_1-x_0) & (x_2-x_0)^T(x_2-x_0) \end{matrix}  \right] = \left[ \begin{matrix} d_0 & d_1 \\ d_1 & d_2 \end{matrix} \right] (注意它是对称的)
$$

现在让我们再进一步：

$$
\begin{aligned}
F^TF &= \left[ \begin{matrix} m_0 & m_1 \\ m_2 & m_3 \end{matrix}  \right] \left[ \begin{matrix} d_0& d_1 \\ d_1 & d_2 \end{matrix}  \right]^T\left[ \begin{matrix} m_0 & m_1 \\ m_2 & m_3 \end{matrix}  \right]  \\
&=\left[ \begin{matrix} m^2_0d_0 + 2m_0m_1d_1 + m^2_1d_2 & m_0m_2d_0 + (m_0m_3+m_1m_2)d_1 + m_1m_3d_2 \\ m_0m_2d_0 + (m_0m_3 + m_1m_2)d_1 + m_1m_3d_2 & m^2_2d_0 + 2 m_2m_3d_1 + m^2_3d_2 \end{matrix}  \right]    
\end{aligned} 
$$

然后用Frobenius范数把一切都堆在一起：

$$
\parallel F \parallel^2_F = (m_0^2d_0 + 2M_0m_1d_1 + m^2_1d_2)^2+ (m_0m_2d_0 + (m_0m_3+ m_1m_2)d_1 + m_1m_3d_2)^2 + (m_0m_2d_0 + (m_0m_3+m_1m_2)d_1 + m_1m_3d_2)^2 + (m^2_2d_0 + 2m_2m_3d_1 + m^2_3d_2)^2
$$

好吧，我现在必须停止。天啊，看这个都觉得疼。这个表达方式很足够大，几乎不适合在两行上写。 我可能在某处犯了一个错误；祝你好运找到它。下面，我们必然将$d_0=(x_1-x_0)^T(x_1-x_0)$代回去，这里，$\bold x_0=\left[ \begin{matrix} x_0 & y_0 \end{matrix} \right]^T$, 同样的，$\bold d_1$、$\bold d_2$、$\bold x_1$和$\bold x_2$，然后对其进行多次求导。而且，事情在三维中变得更糟，而且狄利克雷能量甚至不是那么复杂的一个能量! 对于更复杂的材料模型来说，事情甚至会沉到更深的可怕程度。

你可以使用一个计算机代数系统，如Mathematica或Matlab的符号工具箱来处理这一切。这只是把产生丑陋的工作交给了计算机，而最后的结果仍然是丑陋的。当然，有一个更好的方法。

### 2.4.3 现在怎么办
有一个更好的方法，我们将在下一章介绍。当我们这样做时，你会看到,你会发现，现在的力的计算更简单了，但力的梯度（又称能量Hessians）仍然很难看。

你可能很想在这里采取懦夫的方式。所有这些丑陋的东西开始于因为我坚持要用$F$而不是$x$来计算我们的变形量。当坚持用x时， 我们就不会陷入这样的困境，因为 当$\Psi$纯粹用x来写的时候，计算计算$\frac{\partial \Psi}{\partial x}$,就容易多了。看看这个看起来很好的弹簧-质量能量:
$$ \Psi = \frac{\mu}{2} \parallel \bold x - \bold x_0 \parallel ^2_2 \tag{2.24}$$ 
$$ \frac {\partial \Psi}{\partial \bold x} = \mu(\bold x - \bold x0)  \tag{2.25}$$
太简单了。但不要被诱惑。如果我们允许自己被吸进这个Charybdis，我们就会把自己限制在弹簧质量模型的世界里。而更丰富的挤压和压扁的体积反应的世界将无法进入。让我们不要这样做。让我们看看我们能把这个$F$的事情做到什么程度。

# 第三章 用张量的方式计算力
如果你跳过第一章后刚加入我们，下面是我们目前的困境。我们正试图计算力，$f = a \frac {\partial \Psi }{\partial \bold x}$ ，但事实证明，当你的能量$\Psi$基于变形梯度$F$时，直接取x导数是非常难看的。它非常难看，以至于我们开始怀疑使用基于$F$的能量到底是不是一个好主意。现在你们都明白了，那就继续表演吧。

我们将使用链式规则的应用来清理事情，将$\frac {\partial \Psi} {\partial \bold x}$分成两个组成部分:
$$ \frac{\partial \Psi}{\partial \bold x} = \frac{\partial \Psi}{\partial F} \frac {\partial F} {\partial \bold x} \tag{3.1}$$

每一块，,$\frac{\partial \Psi}{\partial F}$和$\frac {\partial F} {\partial \bold x}$孤立地看都是相 ∂x 对干净的。如果你试图咬牙切齿地同时处理这两块，我们在前面一章看到的丑陋的就会出现。然而，如果你以前没有接触过高阶张量，那么当你开始尝试计算这些术语的时候，你就会遇到麻烦。整个事情看起来就像某个教授对你玩的一个肮脏的微积分把戏，他把所有困难的细节都排除在外。现在让我们来看看这些细节。

## 3.1 关于三阶张量的思考
公式3.1中的第一项，即$\frac{\partial \Psi}{\partial F}$项，相对容易。由于$Ψ∈\Re$和$F∈\Re^{3×3}$，所以导数只是一个矩阵。
$$\frac {\partial \Psi}{\partial F} = \left[  \begin{matrix}  \frac{\partial \Psi}{\partial f_{00}} & \frac{\partial \Psi}{\partial f_{01}}  & \frac{\partial \Psi}{\partial f_{02}}  \\
\frac{\partial \Psi}{\partial f_{10}} & \frac{\partial \Psi}{\partial f_{11}}  & \frac{\partial \Psi}{\partial f_{12}}  \\
\frac{\partial \Psi}{\partial f_{20}} & \frac{\partial \Psi}{\partial f_{21}}  & \frac{\partial \Psi}{\partial f_{22}}  
\end{matrix} \right] \in \Re^{3\times 3} \tag{3.2}
$$
$f_{ij}$ 标量是$F$的条目，如附录A中所述。其结果显然是一个3×3矩阵，也被称为二阶张量。

当我们试图计算$\frac{\partial F}{\partial \bold x}$时，困惑开始了。由于$F \in \Re^{3×3}$ 是一个矩阵，$x \in \Re^{12}$ 是一个矢量，导数应该采取什么形式？让我们看看$x_0$ ，$\bold x$的第一个条目。我们可以取$F$对它的导数:
$$\frac {\partial F}{\partial x_0} = \left[  \begin{matrix}  \frac{\partial f_{00}}{\partial x_0} & \frac{\partial f_{01}}{\partial x_0}  & \frac{\partial f_{02}}{\partial x_0}  \\
\ \frac{\partial f_{10}}{\partial x_0} & \frac{\partial f_{11}}{\partial x_0}  & \frac{\partial f_{12}}{\partial x_0}  \\
 \frac{\partial f_{20}}{\partial x_0} & \frac{\partial f_{21}}{\partial x_0}  & \frac{\partial f_{22}}{\partial x_0}  \\
\end{matrix} \right] \in \Re^{3\times 3} \tag{3.2}
$$
已经有了，结果是一个矩阵。我们应该如何处理下一个导数，$\frac{\partial F}{\partial \bold x_1}$，以及之后的一个，再之后的一个？最终我将有一堆12个矩阵，但我该怎么处理它们呢？有没有一些标准的方法把这些矩阵安排在一起？

答案是否定的。问题是，我们现在有一个三阶张量，这通常不会出现在线性代数入门课本中。如果你在关于高阶张量操作的文本中寻找（例如Golub和Van Loan（2013）；Simmonds（2012）；Kolda和Bader（2009）），他们描述了各种安排这堆矩阵的方法，其中一些有重要的听起来的名字，如爱因斯坦符号。当然就是这个了？如果它对爱因斯坦来说足够好，对我们来说也足够好？事实上，不是这样的；我们应该在正确的情况下使用正确的工具，而我们现在不是在看相对论。

![F3.1](https://img-blog.csdnimg.cn/8de26fe975974c159c50378140830e16.png#pic_center)
导数的维度是$\frac{∂ F}{∂x} \in \Re^{3×3×12}$，其中第三个维度是我们在看一个三阶张量的大赠品。许多人认为这是个立方体的矩阵。我们可以把我们这堆12个矩阵，像一副牌一样堆成一个立方体（图3.1）。只是为了播种更多的混乱，如果你看看我的论文，Kim and James（2012），你会发现我在那篇论文中也决定使用这种表述。
![F3.2](https://img-blog.csdnimg.cn/4631f5e5d6534fbba9c01b4f6d9ed84a.png#pic_center)

我们在这里不打算采取这种观点。相反，我们要把三阶张量看作是一个矩阵的向量。把$\frac{\partial F}{\partial \bold x}$张量想象成厨房柜台上的一个大木质砧板。然后，每个矩阵$\frac{\partial F}{\partial \bold x_i}$按顺序，沿着一条线，顺着砧板摆放（图3.2），就像一片片独立包装的美国奶酪。当我们明确写出一个三阶张量时，我们将使用方括号来模仿我们对带奶酪的切菜板的心理想象。比如:
$$\left[ \begin{matrix}  \left[ \begin{matrix}  a & c \\ b& d \end{matrix}  \right]  \\\left[ \begin{matrix}  e & g \\ f& h \end{matrix}  \right]  \\
\left[ \begin{matrix}  i & k \\ j  & l \end{matrix}  \right]  
 \end{matrix}  \right] = \left[  \begin{matrix}  [A] \\ [B] \\ [C] \end{matrix} \right]  \tag{3.4}
$$
在右手边，我们把每个矩阵也用[ ]包裹起来。这是为了区分 三阶张量和更熟悉的二阶块状矩阵。

## 3.2 三阶张量的乘法运算
一旦我们对三阶张量有了心理预期，我们应该如何考虑乘法这样的事情呢？这一点非常重要，因为我们最初的主张之一是，链式$\frac{\partial \Psi}{\partial F} \frac {\partial F} {\partial \bold x}$,会使事情变得更容易，而到目前为止，它确实没有看起来更容易。矩阵$\frac{\partial \Psi}{\partial F}$与砧板$\frac {\partial F} {\partial \bold x}$的乘积是多少？

在这部分，你会因为教授对一个重要的细节视而不见而感到恼火。如果我们要写下坚如磐石的，100%对应于你要做的实际计算，我们真的应该写:
$$ \frac{\partial \Psi}{\partial \bold x} =  \frac {\partial F} {\partial \bold x} ：\frac{\partial \Psi}{\partial F} \tag{3.5}$$
其中':'是一个双收缩。如果你以前没有见过双收缩，我们在附录C.3中回顾了将两个矩阵收缩在一起的含义。那一节的关键内容是:
$$
\bold A: \bold B=\left[ \begin{matrix} a_0 & a_2\\ a_1 & a_3 \end{matrix} \right] \left[ \begin{matrix} b_0 & b_2\\ b_1 & b_3 \end{matrix} \right] = a_0b_0 + a_1b_1 + a_2b_2 + a_3b_3 \tag{3.6}
$$
然后，$\frac{\partial F}{\partial \bold x}$是一个三阶张量， 不是一个矩阵，那三阶跟2阶双收缩是什么意思呢？我们定义：
$$
 A : \bold B = \left[ \begin{matrix} \left[ \begin{matrix} a_0 & a_2 \\ a_1  &a_3& \end{matrix} \right]  \\ \left[ \begin{matrix} a_4 & a_6 \\ a_5  &a_7& \end{matrix} \right]  \\  \left[ \begin{matrix} a_8 & a_{10} \\ a_0  &a_{11}& \end{matrix} \right]  \end{matrix}   \right]  \left[ \begin{matrix} b_0 & b_2 \\ b_1 & b_3 \end{matrix} \right] = \left[ \begin{matrix}  a_0b_0 + a_1b_1 + a_2b_2 + a_3b_3 \\ a_4b_0 + a_5b_1 + a_6b_2 + a_7b_3 \\ a_8b_0 + a_9b_1+ a_{10}b_2 + a_{11}b_3 \end{matrix} \right] \tag{3.7}
$$

我们现在第一次使用黑体字体**A**,来表示高阶张量。我们将使用这些来表示三阶以上的张量。阶数从上下文中显而易见。
为了形成另一个心理画面，$\frac{\partial \Psi}{\partial F}$是一个与我们砧板上的奶酪片   $\frac {\partial F} {\partial \bold x}$大小完全相同的奶酪片.然后我们把这块新的奶酪片对着砧板上所有的奶酪片进行双重收缩(图3.3)，结果是一个好的向量。
![F3.3](https://img-blog.csdnimg.cn/38175fe67a4a4fa9b87e5efb4695f3c2.png#pic_center)


## 3.3 扁平张量的乘法
只是为了让你发疯，还有第二种更传统的方法来写下这一切。任何张量，不管是三阶、四阶还是百阶，都可以被翻转或向量成我们熟悉的二阶矩阵形式。这里我们或多或少将遵循Golub和Van Loan（2013）的惯例。


---
If you like it, don't forget to give me a star.

[![Star This Project](/images/posts/github.svg)](https://github.com/fwzhuang/fwzhuang.github.io)