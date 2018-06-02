# use-tensorflow-to-solve-ode-pde

[Address](git@github.com:HuangJiaLian/use-tensorflow-to-solve-ode-pde.git)

#### 可能的错误

#### loss定义出错:

- reduce_sum 的时候没有添加求和的维度。
- 求偏导是错的

数据输入结构出错:

网络结构出错:

- ~~激活函数没有加对~~

#### Todo:

- [x] 使用标准数据进行训练
- [x] 分离训练和测试
- [x] 添加远程恢复模型
- [ ] 如何获得/赋初值 weight 和 bias 的值
- [ ] 实现外网连接BeiHang的GPU电脑
- [ ] 实现图形化登陆GPU电脑
- [ ] 实现4台主机并行计算
- [ ] 找到正确的训练方法解决解方程的问题
  - 先尝试解连续的
  - 阅读论文的源代码

#### 疑问

- 我需要看着loss值的变化，如果发现越变越大我就要减小LR,这样和麻烦，应该怎么去弄呢?
- 

