---
layout: post
title: 搭建个人github pages主页!
categories: Web
tags: [github-pages, web]
comments: true
---

### 1. 创建个人仓库
在github上创建一个新的仓库作为你的github pages仓库，名字必须为 username.github,io，其中username是你的用户名。

### 2. 添加静态网页文件
测试样例，在终端上输入

```
git clone https://github.com/username/username.github.io
cd username.github.io
echo "Hello world. My first page!"
git add --all
git commit -m "Initial commit"
git push -u origin master
```

然后github部署完成后，就可以访问https://usermane.github.io了。

### 3. 使用CNAME自定义域名
添加CNAME文件，注意必须是大写， 然后填入你的域名，不需要填入http，只填写域名后面的部分 。

```
touch CNAME
echo youizone.com >> CNAME ##example
git add CNAME
git commit -m "Add CNAME, use my domain"
git push origin master
```

### 4. 修改域名提供商的解析
以阿里云为例，登录阿里云后，在控制台栏找到云解析DNS ，然后选择对应的域名，添加解析，示例如何
![解析](/img/assets/domainDNS.png)
其中 A类型中对应的ip是你ping username.github.io的ip, CNAME的记录值就是你的个人username.github.io
