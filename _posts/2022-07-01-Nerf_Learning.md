---
title: 神经辐射场学习笔记
tags: NeRF
---
神经辐射场(NeRF)，是人工智能与图形学领域的学科交叉。从2020年第一篇论文发表后，该方向受到许多学者的关注，近两年，论文数量已经非常可观了, 下面针对这一方向，记录一些学习笔记。。。 
<!--more-->
[TOC]
###神经辐射场原理简介

##### 隐式场景表示（implicit scene representation）
早几年,定义隐式表征式,通过神经网络根据隐式表达式推断出点云,Mesh和Voxel等是一种神经网络与图形的结合,比较常见例子的有PIFu[https://shunsukesaito.github.io/PIFu/], 通过神经网络来推断出图片中的人体3D Mesh.
#### NeRF
这是神经网络与渲染结合的另一个场景, 实质上构造一个隐式的渲染流程,通过用一个MLP神经网络去推断出渲染所需要的信息,从而得到渲染图像. 最初提出[NeRF](https://www.matthewtancik.com/nerf)概念的论文,其本意是想通过有限视角图片集训练一个神经网络,通过此网络,可以生成任意视角下的渲染图. 后来的研究者逐渐扩展出另一些研究方向,比如有:
  * 生成高精度的Mesh
  * 生成渲染质量更高的渲染效果
  * 加速网络的训练速度
  * 针对性场景NeRF
  * 动态场景NeRF
  * 重打光
  * 等等

#### NeRF整体流程
在介绍NeRF的流程前, 先了解一下神经网络的流程
![DL](/img/assets/Nerf/DL.png)
神经网络分为两个步骤:
(1) 训练部分
1. 第一部分是准备数据
2. 第二部分是定义网络模型
3. 第三部分是定义损失函数Loss.
   
(2) 推理部分
   1. 输入数据
   2. 根据训练的模型,推断出预测值.
   
好了,言归正传, 下图是原文中提到的整个管线.
![pipeline](/img/assets/Nerf/图片1.png)


在进入管线前, 需要提前准备训练数据集.
##### 训练数据集
论文中采用合成数据集跟实拍数据集两种, 主要是获取图片和对应的相机参数（变化坐标、俯仰角、屏幕尺寸、焦距).

[参考来源1](https://zhuanlan.zhihu.com/p/390848839) [参考来源2](https://zhuanlan.zhihu.com/p/495652881)
(1）训练数据：从不同位置拍摄同一场景的图片，目的是获取这些图片的相机位姿、相机内参以及场景的范围。（注：在采集数据时尽量减少背景干扰） 
(2) 获取相机内参：若图像数据集缺少相机参数真值，可以使用经典SFM重建方法解决，比如使用COLMAP或者OpenMVS估计需要的相机参数，让其当做真值使用。 
比如论文中NeRF通过COLMAP可以得到场景的稀疏重建结果，其输出文件包括相机内参，相机外参和3D点的信息.然后进一步利用[Local Light Field Fusion](https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py)中imgs2poses文件将内外参整合到一个文件poses_boudns.npy中，该文件记录了相机的内参,包括图片分辨率（图片高与宽度）、焦距，共3个维度、外参（包括相机坐标到世界坐标转换的平移矩阵T与旋转矩阵R，其中旋转矩阵为R^3x3 的矩阵，共9个维度，平移矩阵为R^3x1 的矩阵，3个维度，因此该文件中的数据维度为Nx17（另有两个维度为光线的始发深度与终止深度，通过COLMAP输出的3D点位置计算得到），其中N为图片样本数(注：在imgs2poses中，计算poses时对旋转矩阵进行[r -u t] ->[-u r -t]的变换，应该是将opencv矩阵向opengl矩阵转换，两者的坐标系不同,前者的坐标轴是[right, down, forwards]或[x, -y, -z]。，后者的坐标轴是[down, right, backwards]或[-y, x, z] 相机面向前方的位置-z。))。

(3) 训练前的输入，[参考来源2]中的解析， 所以从形式上来讲MLP的输入并不是真正的点的三维坐标，而是由像素坐标经过相机外参变换得到的光线始发点与方向向量以及不同的深度值构成的，而方位的输入经过标准化的光线方向。

4）训练过程： 
* 先将这些位置输入MLP以产生volume density和RGB颜色值；
* 取不同的位置，使用体渲染技术将这些真值合成为一张完整的图像； 因为体渲染函数是可微的，所以可以通过最小化上一步渲染合成的、真实图像之间的差来训练优化 NeRF场景表示； 
* 这样，一个NeRF训练完成后，就得到一个以多层感知机的权重表示的模型，一个模型只含有该场景的 的信息，不具有生成别的场景的图片的能力。（这也是NeRF的其中一个较大缺点，即一个模型只能表 达一个场景） 

#### NeRF输入输出
从Pipeline图中可以看到NeRF输入输出，函数是将一个连续的场景表示为一个输入为5D向量的函数，包括一个空间点的3D坐标位置 x=(x, y, z)，以及视角方向d=(theta, fine)。这个神经网络可以写作：F: (x, d)-->(c, cigma)，其中cigma是对应3D 位置（或者说是体素）的密度（volume density），而c=(r, g, b)是视角相关的该3D点颜色。有了NeRF形式存在的 场景表示后，可以对该场景进行渲染，生成新视角的模拟图片。原文使用经典的体渲染（volume rendering）原理，求解穿过场景的任何光线的颜色，从而渲染合成新的图像。

##### 体渲染volume rendering的定义
1）体渲染定义：光射线在位置x出的无穷小粒子处的微分概率。于是，具有近边界tn、远边界tf的 相机光线r(t)=o+td的颜色C(r)是：其中T(t)表示沿光线从tn到t累积透射率，也就是光线从tn传播到t而 没有碰到任何其他粒子（仍存活）的概率。
$$C(r) = \int_{tn}^{tf}T(t) \sigma  (r(t)) c(r(t), d) dt$$
2）渲染过程：从NeRF渲染合成一张完整的图片，就需要为通过虚拟相机的每个像素的光线计算 这个积分C(r)，得到该像素的颜色值。使用计算机求积分，必然是离散采样，作者采用分层采样 （stratified sampling）对这个连续积分进行数值估计， 


3）训练优化的trick： 
* 位置编码（positional encoding）：类似于傅里叶变换，将低维输入到映射到高维空间，提升网络捕捉 高频信息的能力； 
* 体渲染的分层采样（hierarchical volume sampling），通过更高校的采样策略减小估算积分式的计算开 销，加快训练速度。

##### 体渲染 
1. 流程：NeRF在通过一个神经网络获取每个像素的RGB和体积密度后，使用体渲染对这个场景 表示进行体渲染。
2. 体渲染函数解释：o为光源点，d为图像成像位置，od射线经过物体（tn为近点，tf为远点）， 从NeRF渲染合成一张完整的图片，就需要为通过向虚拟相机的每个像素的光线计算积分C(r)。其中 σ(r(t))表示t点处的体积密度（即不透明度），T(t)表示从tn到t的透明度的累积，当tn到t的不透明度不 断增大时，T(t)值会不断减小，即T(t)为一个权重值，代表了tn到t处体积密度对成像点d的贡献率。 

 ![Volume](/img/assets/Nerf/图片5.png)

 3. 体渲染函数分层采样（离散采样）：把积分区间[tn, tf]分成等间距多个小区间，然后再每个小 区间以均匀分布的概率随机采样，以随机采样的点的RGBσ值代表小区间的值，而不是每次都固定用小 
区间的端点或中点。即每个小区间是[tn+(i-1)*(tf-tn)/N, tn+i*(tf-tn)/N]，那么，第i个小区间随机采 样的点ti服从该小区间上的均匀分布。最终体渲染函数的离散形式为：
$$C(r) = \sum_{i=1}^N c_iT_i(1-e^{-\sigma_i\delta_i}), T_i = exp(-\sum_{j=1}^{i-1}\sigma_i\delta_i) $$

##### Positional Encoding的意义
* 这种表示方法，即便两个点在原空间中的距离很近，很难分辨，但通过Positional encoding后，我们还是可以很轻松的分辨两个点！
* Fourier Feature Networks：对原图进行傅里叶变换映射，再输入MLP中，能够达到让MLP从低维特征领域 学习到高频函数。通过如下图的对比，可知通过将图片进行简单的傅里叶变换映射可以使得神经网络快速收敛，并能 获得更高的信噪比。 
![FFN](/img/assets/Nerf/图片2.png)


#### NeRF缺点： 
* 一个模型只能表达一个场景，且优化一个场景耗时久（主要原因：raycast数量较多，导致采样点也过 多，每条raycast有64个粗采样点和128个细采样点，由于每个采样点都需要送入网络计算一遍，因此耗 时很大。具体改进方法有FastNeRF） 
* per-pixel渲染较为低效
* 泛化能力较差，一个场景需要较多的图片才能训练好 
  

###NeRF的网络设计 
1. 位置编码：将空间中的单个点编码成特征向量，其中每个元素由正弦曲线生成，频率呈指数增 加，便于神经网络去理解位置信息。例如将x信息编码成（x, sin(2^k*x), cos(2^k*x), ...)，其中k=10 和4。

2. NeRF网络：如下图，xyz编码后输入到8个fc层（每个都是256）中（在第4个输出中加入skip connection），在第8个fc层后接一个256到1的fc层用于输出体密度sigma，然后再第9个fc层中加入 观测点编码进行拼接后继续接一个128维的fc层，最后128转3的fc层输出rgb值。
 ![Nerf](/img/assets/Nerf/图片6.png)

 3. 细节：一个batch size内有4096条rays 

###Ray Marching算法 
体渲染可以通过投影面Ray Marching方法来实现，Ray Marching具体由4个步骤来实现： 
a. 在投影图像上逐像素产生射线Raycast； 
b. 沿射线对体积的体素采样； 
c. 获取/计算体素特性； 
d. 累积体素特征计算投影图形的颜色灰度值。 

 ![RayMarching](/img/assets/Nerf/图片7.png)

NeRF渲染实现正是按照上述步骤实现，并通过离散采样，投影积分过程转换为累积求和。由于体 素特性由可微的MLP函数表征，整个渲染流程是可微的，从而方便在现代深度学习框架上实现。

###NeRF中的Ray Marching算法

体渲染假定3D场景的所有位置体素都可以由其色彩和不透明度（密度）来表达，沿着指定观察方 向对空间体素积分即可实现渲染。如图，体渲染是一个ray marching过程。具体实现可分为几步： 
1） 根据视角和成像面分辨率确定投影射线位置 
2） 射线上采样体素，逐点输入NeRF网络计算颜色密度。3D场景有效体素的分布是稀疏的。为提 
升采样效率，Nerf采用了二次采样方法。第一次粗采样64点，估计该射线体素的概率密度分布。第二 次根据密度分布，再采样128点。 
3） 射线上所有点积分，得到成像面一个像素的渲染值。 
4） 扫描成像面所有像素，完成整幅图像渲染。 由此可见，NeRF渲染图像需要执行多次的采样和NeRF网络计算。如果产生一幅1024x1024的图像， 
我们可以估算其乘加次数为： 
一个投影像素 192个采样：192x629K = 123,666,432 = 117.94M（其中629K表示一个像素经过MLP 网络的计算次数） 
一幅1024x1024图像：1024x1024x117.94 = 117.94T 
 ![RayMarching2](/img/assets/Nerf/图片8.png)


###不透明度sigma求解顶点和三角面片
在神经辐射场中，只要求解出每个点的不透明度，即可通过MarchingCubes算法提取等值面的方 式求解得到顶点和三角面片 
```
import numpy as np 
import open3d as o3d 
import trimesh 

N = 128 
sigma_threshold = 20 

sigma = rgbsigma[:, -1].cpu().numpy()    # rgbsigma是CNN回归得到的[b, n, 4]数据，其中最后 一维的4列由r,g,b,sigma组成 
sigma = np.maxinum(sigma, 0).reshape(N, N, N)    # 将sigma中小于0的数置0 

vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)   	 # 使用 
mesh = trimesh.Trimesh(vertices/N, triangles) 
mesh.show() 

```

#### nerf-pl实现
原理解释[https://liwen.site/archives/2302]



###NeRF重建三维模型Mesh和加RGB的步骤

1）首先规定一个比较大的立方体，将物体包围在内，然后对该立方体划分成小块； 
2）使用CNN网络，输入XYZ和相机观测角度，输出RGB和不透明度sigma来预测出各个小块是否 有物体以及其不透明度值（若没有物体或者物体相对透明时，sigma值大约在0~10；若物体不透明， 则sigma值可能在50以外）； 
3）预测出sigma值后，使用marching cubes算法即可以获得三维模型的顶点和三角面片，即 mesh模型； 
4）去除噪声：如离群点、面片较少的簇等 5）给mesh增加顶点颜色：对世界坐标系下物体的每个点，将其分别投影到每个相机下的像素坐 
标，获得该相机下的图像的点的RGB值，然后将这些RGB值求和，即作为该点的颜色RGB值。问题： 以上方法，在物体有遮挡的情况下，把物体前面可见部分也投影到背后遮挡面。 

###近期NeRF的研究发展 

1）场景合成的代表作：GIRAFFE（2021年CVPR的best papers），是NeRF和GRAF的融合。 NeRF之后，GRAF（Genarative Radiance Field），其关键点在于引入GAN来实现NeRF，并使用CGAN实现对渲染内容的可控性。在GRAF之后，GIRAFFE实现了composition。在NeRF、GRAF中， 一个Neural Radiance Fields表示一个场景，one model per scene。而在GIRAFFE中，一个Neural Radiance Fields只表示一个物体，one object per scene（背景也算一个物体）。这样做的妙处在于 可以随意组合不同场景的物体，可以改变同一场景中不同物体间的相对位置，渲染生成更多训练数据 中没有的全新图像。如下图所示，GIRAFFE可以平移、旋转场景中的物体，还可以在场景中增添原本 没有的新物体。另外，GIRAFFE还可以改变物体的形状和外观，因为网络中加入了形状编码、外观编 码变量（shape codes, appearance codes ）。

![GIRAFFE](/img/assets/Nerf/图片3.png)

2）IBRNet：学习一个适用于多种场景的通用视图插值函数，从而不用为每个新的场景都新学习 一个模型才能渲染；且网络结构上用了另一个时髦的东西 Transformer。

3） ![GIRAFFE](/img/assets/Nerf/图片4.png)


###NeRF加速专题

1.  PlenOctrees [https://alexyu.net/plenoctrees/]
我们提出了一个框架，可以使用全光八叉树或“PlenOctrees” 实时渲染神经辐射场 (NeRF) 。我们的方法可以在 800x800px 分辨率下以超过 150 fps 的速度渲染，这比传统的 NeRF 快 3000 倍以上，而不会牺牲质量。

![管道](/img/assets/Nerf/pipeline.png)

实时性能是通过将 NeRF 预先制成基于八叉树的辐射场（我们称为 PlenOctrees）来实现的。为了保留与视图相关的效果，例如镜面反射，我们建议通过封闭形式的球面基函数对外观进行编码。具体来说，我们表明可以训练 NeRFs 来预测辐射的球谐表示，将观察方向作为神经网络的输入。此外，我们表明我们的 PlenOctrees 可以直接优化以进一步最小化重建损失，这导致与竞争方法相同或更好的质量。我们进一步表明，这个八叉树优化步骤可以用来加速训练时间，因为我们不再需要等待 NeRF 训练完全收敛。我们的实时神经渲染方法可能会支持新的应用，例如 6 自由度工业和产品可视化，以及下一代 AR/VR 系统。

2. FastNeRF[https://arxiv.org/abs/2103.10380]
最近关于神经辐射场 (NeRF) 的工作展示了如何使用神经网络对复杂的 3D 环境进行编码，这些环境可以从新颖的视角进行逼真的渲染。渲染这些图像对计算的要求非常高，最近的改进距离实现交互速率还有很长的路要走，即使在高端硬件上也是如此。受移动和混合现实设备场景的启发，我们提出了 FastNeRF，这是第一个基于 NeRF 的系统，能够在高端消费 GPU 上以 200Hz 渲染高保真逼真图像。我们方法的核心是受图形启发的分解，它允许 (i) 在空间中的每个位置紧凑地缓存深度辐射图，(ii) 使用光线方向有效地查询该图以估计渲染图像中的像素值。

3. SNeRG [https://phog.github.io/snerg/index.html]
诸如神经辐射场 (NeRF) 之类的神经体积表示已经成为一种引人注目的技术，用于学习从图像中表示 3D 场景，目的是从未观察到的视点渲染场景的逼真图像。然而，NeRF 的计算要求对于实时应用来说是禁止的：从经过训练的 NeRF 渲染视图需要每条射线查询多层感知器 (MLP) 数百次. 我们提出了一种训练 NeRF 的方法，然后将其预计算并存储（即“烘焙”）为一种称为稀疏神经辐射网格 (SNeRG) 的新颖表示，它可以在商品硬件上进行实时渲染。为了实现这一点，我们引入了 1）NeRF 架构的重新表述，以及 2）具有学习特征向量的稀疏体素网格表示。生成的场景表示保留了 NeRF 渲染精细几何细节和视图相关外观的能力，紧凑（每个场景平均小于 90 MB），并且可以实时渲染（笔记本 GPU 上每秒超过 30 帧） . 实际屏幕截图显示在我们的视频中。

4. KiloNeRF[https://arxiv.org/abs/2103.13744]
NeRF 通过将神经辐射场拟合到 RGB 图像，以前所未有的质量合成场景的新视图。然而，NeRF 需要数百万次查询深度多层感知器 (MLP)，导致渲染时间变慢，即使在现代 GPU 上也是如此。在本文中，我们展示了通过使用数千个微型 MLP 而不是一个大型 MLP，实时渲染是可能的。在我们的设置中，每个单独的 MLP 只需要代表场景的一部分，因此可以使用更小、更快评估的 MLP。通过将这种分而治之的策略与进一步的优化相结合，与原始 NeRF 模型相比，渲染速度提高了三个数量级，而不会产生高存储成本。此外，使用师生蒸馏进行培训，

5. NeX[https://nex-mpi.github.io/]
我们提出了 NeX，这是一种基于多平面图像 (MPI) 增强的新型视图合成的新方法，可以实时再现 NeXt 级别的视图相关效果。与使用一组简单 RGBα 平面的传统 MPI 不同，我们的技术通过将每个像素参数化为从神经网络学习的基函数的线性组合来模拟视图相关的效果。此外，我们提出了一种混合隐式-显式建模策略，该策略改进了精细细节并产生了最先进的结果。我们的方法在基准前向数据集以及我们新引入的数据集上进行了评估，该数据集旨在测试与视图相关的建模的极限，具有明显更具挑战性的效果，例如 CD 上的彩虹反射。我们的方法在这些数据集的所有主要指标上都取得了最好的总体得分，渲染时间比现有技术快 1000 倍以上
6. Instant-NGP
通过hash编码对其进行查找加速。
pytorch实现[https://github.com/Hik289/gnp_from-others]



###NeRF的改进方法 
来源：[https://zhuanlan.zhihu.com/p/512538748]	  
1. 针对推理时间慢，有AutoInt、FastNeRF等；

2. 针对训练时间慢，有Depth-supervised NeRF（使用SFM的稀疏输出监督NeRF，能够实现更 少的视角输入和更快的训练速度）

3. 针对NeRF只建模静态场景，有Neural Scene Flow Fields（将动态场景建模为外观、几何体和三维场景运动的时变连续函数，只需要一个已知单目视频作为输入），4D-Facial-Avatars （NerFACE）等
4. 针对泛化性问题，即一个场景需要训练单独一个模型，模型无法扩展到其他场景，有GRF（通 过学习2D图像中每个像素的局部特征，然后将这些特征投影到3D点，从而产生通用和丰富的点表 示），IBRNet，PixelNeRF等；

5. 针对视角数量问题，即NeRF可能需要数百张不同视角的图片才能训练好一个模型，这极大限制其在现实中的应用。有PixelNeRF（使用一个CNN Encoder提取图像特征，从而使得3D点具有泛化性，并且支持少量输入，其能支持单张图像输入）

6. 针对环境问题，即目前大多数NeRF仅在封闭环境内测试，在开放真实环境下效果并不佳，有Urban-NeRF（谷歌街景重建），Block-NeRF等
 ![Urban_NeRF](/img/assets/Nerf/图片9.png)

7. 针对NeRF框架改进，有Mip-NeRF（提出一种基于视锥的采样策略，实现了抗锯齿的效果， 其减少了NeRF中的混叠伪影，并显著提高了NeRF表达精细细节的能力，同时推理速度比NeRF快 7%） 
 ![Mip_NeRF](/img/assets/Nerf/图片10.png)

8. 针对NeRF的应用一：逆渲染，即从真实数据中估计不同模型参数（包括相机、几何体、材 质、灯光参数等），其目的是生成新视图、编辑材质或照明，或创建新动画。 
几何与代理几何：NerfingMVS[9]用SfM估计的稀疏深度来监督单目深度估计网络，调整其尺度，然后再输入 NeRF网络中实现视角一致性； 
照明：NeRV以一组由无约束已知照明照亮的场景图像作为输入，并生成一个可以在任意光照条件下从新视 点渲染的三维表示； 
相机（位姿估计）：Self-Calibrating在没有任何校准对象的情况下，共同学习场景的几何结构和精确的相机 参数，提出了一张还适用于具有任意非线性畸变的普通相机自标定算法。

9. 针对NeRF的应用二：可控编辑，即对场景能进行包括形状、外观、场景组合等的人工编辑， 有EditNeRF-->GRAF-->GIRAFFE，这些方法实现GAN和NeRF的结合，实现可控编辑； 
 ![GIRAFFE](/img/assets/Nerf/图片11.png)

10. 针对NeRF的应用三：数字化人体，包括人脸建模、人体建模以及人手建模等； 
* 人脸建模：4D Facial Avatars，将3DMM和NeRF结合，实现动态神经辐射场，其输入一个单目视频， 能够实现人脸的位姿、表情编辑； 
* 人体建模：Animatable，其引入神经混合权重场来产生变形场，其输入为多视角视频。目前该领域只要 想SMPL靠近，即给定一个规范空间，然后从不同的观测空间估计规范空间； 
 ![Animatable](/img/assets/Nerf/图片12.png)

11. 针对NeRF的应用四：多模态，即针对目前大部分NeRF的工作都是基于图像或者视频输入， 多模态即探索输入其他模态如文字、音频等与图像进行结合。方法有CLIP-NeRF，将CLIP和NeRF结 合，实现通过文字和图像编辑场景；
 ![CLIP_NeRF](/img/assets/Nerf/图片13.png)

12. 针对NeRF的应用五：图像处理，如压缩、去噪、超分、inpainting等 
![Knitworks](/img/assets/Nerf/图片14.png) 


13. 训练数据处理，针对人物，人脸的情况，在准备训练图像集时，那是保持背景干净单一，可以提升效果。

14. Nerf in the wild[https://nerf-w.github.io/]


15. 移动场景重建
    * NSFF[https://www.cs.cornell.edu/~zl548/NSFF/]
    动态部分较模糊。
    * D-NeRF[https://www.albertpumarola.com/research/D-NeRF/index.html]

16. Mip-NeRF[https://jonbarron.info/mipnerf/]

17. 去除反光的
    * 神经渲染场[https://bennyguo.github.io/nerfren/]

18. NeRF训练的场景都比较小,如何处理较大场景
    * nerfplusplus[https://github.com/Kai-46/nerfplusplus]
    * mip-nerf 360[https://jonbarron.info/mipnerf360/]

19. 针对夜晚的场景进行重建
    * [https://bmild.github.io/rawnerf/index.html]

20. 变化过程中NeRF重建
    * hyperNerf[https://hypernerf.github.io/]

21. 神经场景图https://light.princeton.edu/publication/neural-scene-graphs/
    
22. 光的反射[https://dorverbin.github.io/refnerf/]
    效果更好
    【https://aoliao12138.github.io/files/Paper/NeRFAA/NeRFAA_finalproj.pdf】
    球谐高斯[https://github.com/Kai-46/PhySG]
    

23. 都市人[https://city-super.github.io/citynerf/]
24.  像素神经[https://alexyu.net/pixelnerf/]
25.  MVSNeRF[https://apchenstu.github.io/mvsnerf/]
26.  IBR 网络[https://ibrnet.github.io/]
27.  NVSF[https://lingjie0206.github.io/papers/NSVF/]
28.  PlenOctree[https://alexyu.net/plenoctrees/]
29. 千里射频[https://github.com/creiser/kilonerf] 标星
30. 神经衰弱——[https://nerfmm.active.vision/]
31. BARF[https://chenhsuanlin.bitbucket.io/bundle-adjusting-NeRF/]
32. NDF[https://yilundu.github.io/ndf/]

33. 重建Mesh精度
    * [https://github.com/Totoro97/NeuS]
    * [https://github.com/ventusff/neurecon]

    1. 考虑添加深度信息，优化Mesh精度，是否可以接入（https://homes.cs.washington.edu/~holynski/publications/occlusion/index.html）的思想，从视频流中以及slam中的稀疏点云来加密深度边缘。
    2.  目前与深度有关的，Mip-NeRF-RGBD, 其思想只是将真实深度与估算深度做为损失函数
    3.  RGBD 表面重建[https://github.com/dazinovic/neural-rgbd-surface-reconstruction],通过训练一个深度学习网络，TSDF，提交表面精度
    4.  [https://www.cs.cmu.edu/~dsnerf/]添加一个深度监督网络。监督辐射场的损失（https://github.com/dunbar12138/DSNeRF）
    5.  Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction
    6 [https://arxiv.org/abs/2206.00665]


[https://github.com/yenchenlin/awesome-NeRF]

[https://github.com/metaverse3d2022/Nerf-Cuda]
[https://github.com/JulianKnodt/nerf_atlas/blob/6953f58e444d0abc8b8cc62bb06407cae512022c/neural_raytracing.md]
[https://github.com/Xharlie/pointnerf]


#### HDR 
[https://jonathanventura.github.io/PanoSynthVR/assets/abstract.pdf] [https://github.com/jonathanventura/PanoSynthVR]

[https://github.com/LWT3437/LANet?utm_source=catalyzex.com]
[https://github.com/alex04072000/SingleHDRs]
[https://github.com/gabrieleilertsen/hdrcnn]
[https://github.com/mukulkhanna/fhdr]
[https://github.com/timothybrooks/hdr-plus] 谷歌HDR合成的算法 

https://github.com/zju3dv/NeuralRecon-W

《Advances in neural rendering》
神经渲染的目标是以一种可控的方式生成逼真的图像，例如新视点合成、生照明、场景变形、合成等。

神经网络可以看作是一个通用的函数逼近器，采用随机梯度下降法找到最能解释训练集的函数，用训练损失来衡量。而神经渲染类似于经典的函数拟合。

NeRF使用多层感知机来近似3D场景的辐射及密度场。

球谐相关 https://www.irit.fr/STORM/site/recursive-analytic-spherical-harmonics-gradient-for-spherical-lights/



如果您喜欢拼接照片，那么您可能会对PTgui和Hugin感兴趣。这些程序通过将括号中的序列拼接和融合在一起来利用 Exposure Fusion，并获得了一些非常好的结果。


Instant-ngp 小场景Mesh生成精度优化方向 
https://lioryariv.github.io/volsdf/
我正在考虑扩展这个框架以支持 SDF 预测而不是密度。这会有多难？我的意思是像https://lioryariv.github.io/volsdf/
（或 NeuS）
https://github.com/fdarmon/NeuralWarp

这样的东西，你仍然可以从带有体渲染的姿势图像中训练。我天真的猜测是，您需要在渲染之前将几何 MLP 输出从 SDF 转换为密度，并且（可能）更改一些超参数，但我可能会忽略一些东西



---
If you like it, don't forget to give me a star.

[![Star This Project](/img/assets/github.svg)](https://github.com/fwzhuang/fwzhuang.github.io)