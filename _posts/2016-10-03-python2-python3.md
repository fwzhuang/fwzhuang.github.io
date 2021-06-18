---
layout: post
title: Windows下支持多版本python
categories: Python
tags: [Python]
comments: true
---
在我们的学习过程中，常常在学习开源库中，经常会碰到有的用python2,有的用python3，如果做到版本兼容呢？
<!--more-->

一种方式，可以用anaconda，或者用virtualenv进行虚拟环境的使用，这样可以做到python的不同版本控制，另一种方式，则是安装不同的版本的python，进行用命令进行区别，python2,python3。

** 安装python2 ** 
下载python2的安装包安装，并添加python2到系统环境变量。
具体的方式：通过控制面板-》系统和安全-》系统，选择高级系统设置，环境变量，选择Path，点击编辑，新建，分别添加D:\Python\python27和D:\Python\python27\Scripts到环境变量(假设安装python 2.7)。

** 安装python3 ** 
同样的，下载python3的安装包安装，并添加python3到系统环境变量。
具体的方式：通过控制面板-》系统和安全-》系统，选择高级系统设置，环境变量，选择Path，点击编辑，新建，分别添加D:\Python\python37和D:\Python\pytho37\Scripts到环境变量(假设安装python3.7)。

** 重命名python2 **
找到python2的安装目录， 将python.exe修改为python2.exe， 将pythonw.exe修改为pythonw2.exe， 修改后，可以在命令行中输入python2 -V进行查看是否设置成功

** 重命名python3 ** 
找到python3的安装目录，将python.exe修改为python3.exe， 将pythonw.exe修改为pythonw3.exe， 修改后，可以在命令行中输入python3 -V进行查看是否设置成功



** 安装pip2 **
在DOS命令框输入命令，python2 -m pip install --upgrade pip --force-reinstall，显示重新安装成功。

** 安装pip3 **
在DOS命令框输入命令，python3 -m pip install --upgrade pip --force-reinstall，显示重新安装成功。


如果需要用pip安装库，则分别用pip2 install XXX和pip3 install XXX即可安装各自的python包。





