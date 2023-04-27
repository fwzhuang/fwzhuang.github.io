---
layout: post
title: python常用的库
categories: [Python]
description: Python 常用的第三方库
keywords: python, library
mermaid: false
sequence: false
flow: false
mathjax: false
mindmap: false
mindmap2: false
---


python 常用的第三方库
<!--more-->

python使用环境安装， 可以通过官网下载安装包，然后安装；也可能通过Anaconda进行安装，它集成了python环境，并且装了一些常见的第三方库，推荐使用Anaconda。

**Anaconda**

* 在[官网](https://www.continuum.io/downloads)上下载对应系统版本的Anaconda

* 创建环境和启动环境

```
conda create --name py27 python=2.7
activate py27
```

* 常用命令

```
conda list  # list all the installed packages

conda install lib_name # install library

```

**Jupyter**

Jupyter在conda中已经默认安装了，如果需要本地安装，则通过下面的命令安装

```
pip install jupyter
```

**numpy**

提供常用的数值数组、矩阵等函数

```
pip install numpy 
#or
conda install numpy
```

**scipy**

是一种使用NumPy来做高等数学、信号处理、优化、统计的扩展包 [文档](http://docs.scipy.org/doc/)

```
pip install scipy  
#or
conda install scipy
```

pandas

是一种构建于Numpy的高级数据结构和精巧工具,快速简单的处理数据。

```
pip install pandas 
#or
conda install pandas
```

**matplotlib**

python 绘图库，类型于matlab的plot绘图等功能。

```
pip install matplotlib
#or
conda install matplotlib
```

**ntlk**

自然语言处理工具包(Natural Language Toolkit)

```
pip install -U nltk
```
下载预料库

```
nltk.download()
```

**igraph**

图计算和社交网络分析 http://igraph.org/python/

```
pip install -U python-igraph
#or
conda install -c marufr python-igraph=0.7.1.post6
```

**scikit-learn**
Scikit-learn是建立在Scipy之上的一个用于机器学习的Python模块。

```
pip install -U scikit-learn
# or
conda install scikit-learn
```
