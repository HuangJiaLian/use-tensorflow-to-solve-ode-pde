# use-tensorflow-to-solve-ode-pde

[Address](git@github.com:HuangJiaLian/use-tensorflow-to-solve-ode-pde.git)

#### 可能的错误

#### ==loss定义出错==:

- ~~reduce_sum 的时候没有添加求和的维度~~。不用写`reduce_sum`
- 求偏导是错的
- 边界条件应该单独考虑, 可以像论文里面描述的那样去做
- ~~-1 和 -ones是不一样的~~

#### 网络结构出错:

- ~~激活函数没有加对~~

程序里面用的是`tanh`: 改了也无济于事

- ~~网络层数不够多~~ 添加8层也无济于事

第一层的时候没有做和论文一样的预处理:

不明白: `H = 2.0*(X-lb)/(ub-lb)-1.0`有什么用? 

==输入层不是不参与运算吗?== 

#### ~~数据的输入有错:~~

解决方法: solve时和fitting时喂数据的方式是大致相同的

- 对照后发现是一样的，说明不是数据的输入有错

#### ~~做图有问题~~

fitting的做图是正确的，对照便知道

对出来是一样的，所以不是图做错了

#### 反推看看哪一项变化比较大

- [x] 获得网络参数
- [x] 恢复网络参数
- [x] 使用TF表示三项误差函数
- [ ] 对三个函数分别喂入训练数据
  - 保证喂入的时候没有出现错误: 用模型直接恢复，和手动恢复(W,b)对比两者的结果
    - sseu两者都一样
    - 但是 sseb1 和 sseb2 ==不一样 ？？？？==
      - 数据保存的精度？
- [ ] 看是什么情况
- [ ] 通过连续和离散的对比看Uxx,  Ut 这些结果是不是对的

#### Todo:

- [x] 使用标准数据进行训练
- [x] 分离训练和测试
- [x] 添加远程恢复模型
- [x] 如何获得/赋初值 weight 和 bias 的值： 
  - 很简单的: weights 和 bias 都是通过`tf.variable`的方式定义的，在训练结束后可以通过`weight.eval()`来获取训练后的值 : ==**Done**==
  - np数组的保存和提取：==**Done**==
  - 保存标准的Weights和biases ==**Done**==
  - 恢复标准的Weights和biases : 放在`solove_c_train.py`中发现loss值非常大 ==？？？？？？==
  - Ctrl + C 事件
- [ ] 实现外网连接BeiHang的GPU电脑
- [ ] 实现图形化登陆GPU电脑
- [ ] 实现4台主机并行计算
- [ ] 找到正确的训练方法解决解方程的问题
  - 先尝试解连续的
  - 阅读论文的源代码

#### 疑问

- 我需要看着loss值的变化，如果发现越变越大我就要减小LR,这样和麻烦，应该怎么去弄呢?

  

