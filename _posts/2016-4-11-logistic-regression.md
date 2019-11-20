---
title: 机器学习-逻辑斯特回归!
tags: [Algorithm, ML]
---

逻辑斯特回归是一种广义的线性回归模型，它实际是一种分类方法，主要是用于两分类问题，输出两种分类的概率。
<!--more-->

### 逻辑斯特回归模型

在实现应用过程中，经常会用到逻辑斯特回归模型，比如:

* 判断一个人是否有意愿购买房子
* 判断一个人的性别
* 判断某用户是否会购买某宝的品类等
对于这种二分类问题，通常都会用到逻辑回归模型。

#### 逻辑回归假设

logistic函数之所以特别适应于二分类问题，归功于sigmoid函数， sigmoid函数的定义

$$g(z) = \frac {1}{1 + e^{-z}} $$

其图如下所示：

![sigmoid](/img/assets/ML/sigmoid.png)

logistic函数定义：

$$ h_\theta(x) = g(\theta^T x) $$

其中，g就是sigmoid函数。

#### 损失函数

* 定义损失函数

$$J(\theta) = \frac {1}{m} \sum_{i=1}^m \frac{1}{2} \big(h_\theta(x^{(i)}) - y^{(i)}\big)^2$$

此时，损失函数可能是非凸的，而我们希望损失函数是凸的，这样，可以找到最优点，使得损失函数最小化。

所以，我们定义Cost函数

$$Cost(h_{\theta}(x), y) = \big\{  _{-log(1-h_\theta(x)) \; if \; y= 0}^{-log(\theta(x)) \; if \;y=1 }$$

那种损失函数$$J(\theta)$$为

$$J(\theta) = \frac{1}{m}\sum_{i=1}^{m} Cost(h_{\theta}(x^{(i)}), y^{(i)})  $$

$$ J(\theta) = \frac{1}{m}\sum_{i=1}^{m}\big[-y^{(i)}log( h_\theta(x^{(i)}))-(1-y^{(i)})log(1-h_\theta(x^{(i)}))\big]$$

添加正则化项

$$ J(\theta) = \frac{1}{m}\sum_{i=1}^{m}\big[-y^{(i)}log( h_\theta(x^{(i)}))-(1-y^{(i)})log(1-h_\theta(x^{(i)}))\big] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$


#### 梯度下降法求解

首先向量化的损失函数(矩阵形式)

$$ J(\theta) =-\frac{1}{m}\big((log(g(X\theta))^Ty+(log(1-g(X\theta))^T(1-y)\big)$$

求偏导(梯度)

$$ \frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m}\sum_{i=1}^{m} ( h_\theta (x^{(i)})-y^{(i)})x^{(i)}_{j} $$

向量化的偏导(梯度)

$$ \frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m} X^T(g(X\theta)-y)$$

梯度更新

$$\theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_j)$$


