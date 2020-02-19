###     RNN

>CS231N同济自豪兄讲解版
[TOC]
#### 1.应用场景
* 这里的“多"表示是序列数据（先后发生）
* RNN一大特点：当前输出不仅取决于现在输入，还取决于过去输入

1.一对多
eg：输入一个图像，输出文字描述
![cb5bdbdf8d3c11732c1650730ef9bac0.png](en-resource://database/431:1)
2.多对一
eg：输入一段影评，判断情感是积极或消极（二分类问题）；或者输入一段新闻，判断是军事新闻或政治新闻或娱乐新闻等（多分类问题）
![87649931a01c22499fc6b208aaa83efc.png](en-resource://database/433:1)
3.多对多
eg：谷歌翻译，输入一段中文序列，输出对应英文序列
![c5a04d2757fc8e2c7c1a4f0ac808c4df.png](en-resource://database/435:1)
4.多对多
eg：视频按帧分类：注意这里的按帧分类是和前面帧都相关的，过去帧会对未来帧产生影像而不是单独把帧拿出来进行分类

![c61d32d616c8c94ed403e2d8ce6eca17.png](en-resource://database/437:1)
#### 2.原理探讨

![e562189b021e87517bdee03fb5cfa0ca.png](en-resource://database/439:1)

![a93f8d890c577b6a80b86305a8dcfadb.png](en-resource://database/441:1)
![20063b54786ff04e72a93986c77fb345.png](en-resource://database/443:1)

* 注意这里的权值共享 ，看一个例题
* 一个batch内共用一套权值（我的理解）

![1d4b5fa02720e58dae96791db84111ff.png](en-resource://database/445:1)


![8be629e72164f1e5be7b5e00fa79e4f3.png](en-resource://database/447:1)
*多对一 encoder：把输入序列编码成一个向量
 一对多 decoder：输入一个值，解码成一个序列
####  3.语言模型
语言模型：专业术语，指用上文预测下文
##### 3.1 字符级语言模型

* 注意输入需要把字符变成one hot向量

![142345171d0630bfc7130c7feece68f3.png](en-resource://database/449:1)

* 注意一个箭头代表一个权重

 ![23116589746d5a286def58400ee9c9cc.png](en-resource://database/451:1)

*  测试集时的例子 （输入前一个或几个字符，自动输入后面的字符）

 ![7e49f0af240041f47247db5021a0d599.png](en-resource://database/453:1)
 
#####  3.2 沿时间的反向传播（BPTT）

* 每个权重会被计算多次 

![5d7e9568e88796610aa768b46153c1eb.png](en-resource://database/455:1)
类比之前学的从0开始构造神经网络，按一个批次计算一次损失函数，然后反向传播，用优化器迭代，更新参数w，b；一个批次内用同一组w，b
而这里一个权重会被计算多次

* mini-batch思想

![617c3c874f564302e7f4888b4077a332.png](en-resource://database/457:1)
##### 3.3可视化隐含层
* 不同隐含层中不同的值，负责的是语料库中不同的特征
* 隐含层状态越多 ，越能捕获底层特征
* 如下图是把负责缩进的隐含层可视化（红色表示值高，蓝色表值低）

![c4d126c7ae0f30bcf0b52f2982fd7358.png](en-resource://database/459:1)
##### 3.4 image caption
###### 3.4.1 原理介绍
* 编码器+解码器：可实现不同类别数据转换，比如这里实现从图片-》文字 
* CNN类似于编码器将图片输入为一个向量；RNN类似于解码器，将一个向量变为一段文字

![847ff51a1e6c6cc9026e6bedd4f24737.png](en-resource://database/461:1)

* 4096维的输出作为图像编码，把紫色值（即得到编码向量乘以权重）作为初始隐藏状态

![571af8966effe1363f4cc632c92b389f.png](en-resource://database/463:1)

* 前一个输出作为下一个输入，到结尾输出一个结束符号

![aa7bef5f72784b1f4cbacce9bacb1285.png](en-resource://database/465:1)

* 具体案例
![ae48f87af428696d0f738c8cd43152d9.png](en-resource://database/467:1)
###### 3.4.2 image caption with attention
![8ceae0c84b57ab23855bbade1b8d1257.png](en-resource://database/469:1)
![da56a258f61bb9a63f12d73c5e04c150.png](en-resource://database/471:1)

* soft attention  生成的是概率分布
* hard attention
![0f3910ab3f1421ba1f9936120b6c2f16.png](en-resource://database/473:1)

* 例子

![eb4b24e8d52fb6b12f11a8491546cd0b.png](en-resource://database/475:1)

* 注意力机制也可用于视觉问答，可保留图片中的位置信息

![ba404ab552d0dc13728327a641ca5d6a.png](en-resource://database/477:1)

* 参考论文
![dc9a2c2d9827b7ec5a230d0100e30f8e.png](en-resource://database/479:1)

* RNN 关于维度 理解
![02520c14a00654326dea13d935491103.png](en-resource://database/511:1)
![b7f15edc14f32f81331072bedc3e1391.png](en-resource://database/513:1)

* 训练时是看困惑度，随着epoch增加，困惑度逐渐减小



#### 4.其他RNN
![f8f72a5090960e116d7e754baa18f63b.png](en-resource://database/521:1)

*  RNN存在问题：梯度消失或爆炸

![7b41bd5082c8d7c88daa3c4a4e26dda6.png](en-resource://database/481:1)

* 裁剪梯度能缓解梯度爆炸，但是无法缓解梯度衰减
于是提出

![f16d5335c31485a1e98b8ffd813eb8d7.png](en-resource://database/483:1)
##### 4.1 GRU

* ⻔控循环神经⽹络：捕捉时间序列中时间步距离较⼤的依赖关系
* 重置门 更新门 候选隐藏状态  $\tilde{H}_t$
* **重置⻔**有助于捕捉时间序列⾥**短期**的依赖关系； **更新⻔**有助于捕捉时间序列⾥**⻓期**的依赖关系。
![f927491631734ba9ae92c8af6405e372.png](en-resource://database/519:1)

* 需要初始化参数 ：隐藏层（9个），输出层（2个），以及$H_-1$（一般初始化为0）

##### 4.2 LSTM

 **长短期记忆long short-term memory :**

* 遗忘门:控制上一时间步的记忆细胞 

* 输入门:控制当前时间步的输入

* 输出门:控制从记忆细胞到隐藏状态

* 记忆细胞：⼀种特殊的隐藏状态的信息的流动

![1ac123a892e717ac274d66a171deac76.png](en-resource://database/515:1)

* 需要初始化参数 ：隐藏层（12个），输出层（2个），以及$C_-1$和$H_-1$

* GRU、LSTM都能捕捉时间序列中时间步距离较⼤的依赖关系。

eg： 例题

![300294f7020e7cf5d11f2be957e344d6.png](en-resource://database/525:1)

##### 4.3 深度循环神经网络
![908b52d6c7450dd874781848fb4e2ddc.png](en-resource://database/517:1)

* 红色部分圈的是之前用的循环神经网络结构（RNN GRU LSTM）
* 可以抽取到更高层，更抽象的信息
* 不是越深越好，越深代表模型越复杂，对数据集要求也更高，内容也更抽象,
 层数的加深也·会导致模型的收敛变得困难
##### 4.4 双向循环神经网络
![c812aaf6e83f8a9fc331ee4f91218dce.png](en-resource://database/523:1)

* 是将两个时间方向拼接起来，含有两个方向信息
* 同时考虑前后的影响
* 效果未必好，要做实验
* 在自然语言里比较常见










 


