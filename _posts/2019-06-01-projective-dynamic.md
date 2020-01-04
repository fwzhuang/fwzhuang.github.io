---
title: Projective Dynamics
tags: Simulation
---

<!--more-->

[矩阵积分的理解参考](https://blog.csdn.net/seamanj/article/details/53300058)

假设X为未知矩阵，那么对于矩阵X的积分

已知

$$ \int x{_{ij}}dx{_{ij}} = \frac {1} {2} x{^2_{ij}} $$

那么有

$$ \int XdX = \sum_i \sum_j \int x{_{ij}}dx{_{ij}} = \frac {1} {2} \sum_i \sum_j x{^2_{ij}} = \frac {1} {2} tr(X^TX) =  \frac {1} {2} || X||{^2_F} $$


隐式欧拉积分方法

$$ \begin{cases} q_{n+1}=q_{n} +hv_{n+1}  \cdots (1)  \\
v_{n+1} =v_{n}+hM^{-1}(f_{int}(q_{n+1})+f_{ext})  \cdots  (2)
\end{cases} $$

其中M为质量矩阵，h为time step, 将（2）代入（1）式，可得

$$　M(q_{n+1}- q_n-hv_n)  = h^2(f_{int}(q_{n+1})+ f_{ext})  \cdots  (3) $$

令
$$ s_n =  q_n + hv_n + h^2 M^{-1}fext$$  

则有

$$ \frac{1}{h^2}M(q_{n+1}- s_n)= f_{int}(q_{n+1}) $$

上式可转化为下面式子的优化问题，

$$ \begin{aligned}
& \int\frac{1}{h^2}M(q_{n+1}- s_n)dq_{n+1} - \int f_{int}(q_{n+1})dq_{n+1} \\
&=\frac{1}{h^2}\int M^{\frac{1}{2}}(q_{n+1}- s_n)d(M^{\frac{1}{2}}(q_{n+1}- s_n)) + \sum_i W_i(q_{n+1})\\
&  =\frac{1}{2h^2}(M^{\frac{1}{2}}(q_{n+1}- s_n):M^{\frac{1}{2}}(q_{n+1}- s_n)) + \sum_i W_i(q_{n+1})\\
&=\frac{1}{2h^2}\|M^{\frac{1}{2}}(q_{n+1}- s_n)\|_F^2 + \sum_i W_i(q_{n+1})
\end{aligned} $$

前一项代表是惯性势能，后面一项代表的弹性势能。



---

If you like TeXt, don't forget to give me a star. :star2:

[![Star This Project](https://img.shields.io/github/stars/kitian616/jekyll-TeXt-theme.svg?label=Stars&style=social)](https://github.com/fwzhuang/fwzhuang.github.io)