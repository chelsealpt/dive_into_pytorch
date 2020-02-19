### attention&&Transformer 讲座
>老师：袁源-贪心科技公开课
[TOC]
#####  pre-transformer
![d9d92091344113ff7c7534be6d63ea26.png](en-resource://database/529:1)



* encoder+decoder

![e8ce17374942e86e7b65f8030e463bf4.png](en-resource://database/531:1)

* 缺点 梯度 消失或爆炸
![912eab7db1df71e691ce7f7b47504a7e.png](en-resource://database/533:1)

* lstm和gru只能缓解 ，不能完全解决这个问题
* 因此提出 transformer
#### transformer

>论文 attention is all you need 
* 不用RNN ，只用attention机制，自然语言是怎么运作的
![eda61f2f7650444e718028974176fa0e.png](en-resource://database/537:1)

 ![7df2e7eb0a5cb4832d94e1845d136ba9.png](en-resource://database/539:1)
##### encoder
* encoder核心：把输入单词（用词向量表达）变换成另一个向量
###### 1.self-attention概念介绍
* 在self-attention中，X1和X2 到 Z1和Z2过程时 有信息交换，Z1和Z2到R1和R2时则是独立的；R1和R2都含有从X1，X2得来的信息

![5abeb094459707bcae39a1fd146c3183.png](en-resource://database/545:1)

* 颜色深浅代表强弱，在图中和it最相关的词是 the，animal；在红色笔写的例子中，和it最相关的则是the，apple
* self-attention表示给一个句子，句子不同部分主要关注哪个

![065aa34985cdf9a3bb80a1f74de007e4.png](en-resource://database/549:1)
###### 2.self-attention具体实现

* 图左边的queries表示搜索关键词；key；value
![db5f8820563e68b3b3020dfa149a27ef.png](en-resource://database/611:1)

* 算z1
q和k的相乘为点积相乘，即对应元素位置相乘求和

![b5be6a985522665a1d1be1c3dd6a3be4.png](en-resource://database/551:1)

* 算z2
![389e33837517d4f8533777ea011c9d01.png](en-resource://database/553:1)
###### 3.vectorization（向量化）
上面的例子中，X1和X2两个词作为输入，分别与$W_Q$, $W_K$,$W_V$,相乘，一共有3+3六次乘法；如果当句子长度为100个词时，则会做100x3=300次乘法，计算开销比较大

* 计算开销大，引入vectorization（向量化）,不增加时间复杂度；即将x1和x2拼起来，矢量计算

 ![dccbae9c84e8543a9dc8d059b2f797aa.png](en-resource://database/557:1)
######  4.multi-headed attention

* 让attention有不同的层次，从不同角度看

![600a7b65cda9b82a9271f7446e2ae996.png](en-resource://database/559:1)
![25b1b7153ffaa52728a049f7c3471dc5.png](en-resource://database/561:1)

* 长度不一样z时，按顺序组合起来，乘以w，得到想要形状

![cc70d285d763f9a507b3f277bf477ad2.png](en-resource://database/565:1)
![f5f62b2db76c24b8ee0721efa4ce244c.png](en-resource://database/567:1)

* 实例
橙色从指代什么东西来看，绿色代表指代什么状态，还可以有更丰富层次：比如什么时间，什么地点，正面或者负面

![ada42988ce0c58081c303b5865eef053.png](en-resource://database/569:1)
![ef81f33a0d3a9b165a9bad3458f7eee0.png](en-resource://database/571:1)
###### 5.positional encoding
![5274dc2b125a09c0d86d189b7d42c539.png](en-resource://database/573:1)

* positional encoding:单词互相之间距离，衡量距离

pos指的单词在原来句子中的位置
i表示 产生encoding vector的位置在哪里
![eaec696802b43688aa198c27b6bf8d56.png](en-resource://database/579:1)

![ee1d092a5d4a586c96904fc1301493ea.png](en-resource://database/575:1)
![8810abdd01f9c4892bbc5bac98339625.png](en-resource://database/577:1)
![d9427cc74cd2d36d45c94647fa9362e0.png](en-resource://database/581:1)

* 物理意义：两两互相间做点积，和自己做点积值最大，越远值越小；它在意的是两个位置远不远

![80767b4759a1da760ee5b2cde1e0ff45.png](en-resource://database/583:1)
positional encoding 可以自己训练出来的，只要你用一种方式，使得互相间互相独立且可以表示位置就行（离得近值大，远值小，这就够了，不一定用sin，cos）
###### 6.layer normalization

* layer 不考虑其他数据，只考虑自己，而batch-normalization和batch_size强相关

![4826a2e821b5d41287c4adf94d9a81f4.png](en-resource://database/585:1)

* resnet：用在计算机视觉里，原是用做来图像分类

![f4e7be612d29debe4aeca24e35e44fb2.png](en-resource://database/587:1)

* regularization  techinical

![cf0bbe4800b09b24db1add266b692d3f.png](en-resource://database/589:1)
![58f0250e0b8757e20b0f48d493c820f5.png](en-resource://database/591:1)
![1dea3ac36a9bfd2fdcb1a0450ebd51b3.png](en-resource://database/593:1)

##### decoder
![6cf34218e03c0d782134d1db486d83b7.png](en-resource://database/595:1)

* the loss evaluation

![94e1513abc47f76851344a1a7bea62fe.png](en-resource://database/597:1)

* 训练小技巧

![90367fad8571f31f605cb95f69e83b21.png](en-resource://database/599:1)
#### 总结
![e796c0796a89469117d671f5b6bdc518.png](en-resource://database/601:1)
* mask即让网络知道哪些部分需要忽略，不需要考虑；比如图片红色部分中表示前三个需要考虑，后面的忽略（0表示有用部分）
mask只关注有用的值，padding则是将不存在的字符补全，使得句子长度一样
![df52c60c849efaaca6b07c5ad6b6d847.png](en-resource://database/609:1)

* 优缺点总结
![7397ef82d09aeddb3381817c427cd29c.png](en-resource://database/603:1)
transformer没有时序概念，因此不会类似于RNN存在双向RNN（RNN有是序概念，所以不利于并行）
* 代码实现
第一个是pytorch实现的，第二个keras

![ac7a10915389dd45be40e0612981d4a7.png](en-resource://database/605:1)

































