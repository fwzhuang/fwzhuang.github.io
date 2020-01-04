---
title: 搭建个人github pages主页!
tags: [Web]
---
GitHub Pages is designed to host your personal, organization, or project pages directly from a GitHub repository. To learn more about the different types of GitHub Pages sites, see "User, organization, and project pages."
<!--more-->

# 如何创建Github Page

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


# 如何装饰Github-Page
本示例使用Jekyll来装饰Github page。

### 1. 首先安装Ruby
到[官网](https://rubyinstaller.org/downloads/)下载安装文件, 本人的安装的版本是（rubyinstaller-devkit-2.5.7-1-x64.exe）， 按要求安装即可。安装完后，通过控制台输入
```
ruby -v 
```
看看不否安装成功。

### 2. 安装rubygems
如果的按上面的安装包的话，可以不必额外安装了rubygems，如果是更早的ruby版本，需要额外安装rubygems， 安装过程，自行百度。

### 3. 定制自己的网站主题风格
到[jekyll主题官网](http://jekyllthemes.org/)找到自己喜欢的风格的pages静态网。 下载合适的主题后，编译测试。
```
bundle install
```
通过上面的命令，安装缺失的库。

### 4. 编写自己的Blog文章。
将blog用markdown的方式，编写后放于_post文件夹下，然后上传到你的github-pages

### 5. 本地测试
通过下面命令
```
bundle exec jekyll serve
```
开启本地服务器，如果一切正常的话，通过浏览器访问（http://127.0.0.1:4000） 即可看到你的blog文章。