---
title: VA-CUDA高亮设置
tags: Tool
---


###  步骤 1 
visual studio里面添加cu,cuh文件支持。 通过： 工具-> 选项-> 文本编辑器-> 文件拓展名, 在vc++中添加cu, cuh文件.

###  步骤 2 
VA助手中,添加cuda头文件路径. 通过:  VA配置中,C/C++目录, Custom选项中,添加以下路径, 示例是针对cuda 10.2的路径设置的.
```
C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\common\inc
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include
```

### 步骤 3

VA中, 添加对cu,cuh文件的支持,需要修改注册列表, 方式如下:
使用Win+R组合键打开"运行"窗口，键入入regedit 命令

打开注册表，找到如下位置： HKEY_CURRENT_USER\Software\Whole Tomato\Visual Assist X\VANet10。在右边找到ExtSource 项目，鼠标右键选修改，在原有文字后 添加如下文字：.cu;.cuh; 确定后关闭注册表。

通过以上三步,就能高亮显示cu, cuh文件代码了.

<!--more-->

---

If you like TeXt, don't forget to give me a star. :star2:

[![Star This Project](https://img.shields.io/github/stars/kitian616/jekyll-TeXt-theme.svg?label=Stars&style=social)](https://github.com/fwzhuang/fwzhuang.github.io)