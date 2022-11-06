# baseline

整理了两种baseline，一种是树模型，一个是nbeats,发现结果提交上都不怎么好，树模型交上去rmse是4.4，nbeats交上去是3.4

## LGB-模型

这个构造了和时间相关的特征，但是可能没有优化，效果不是很好

## Nbeats，rnn模型

这个是优化的重点，因为在本地跑发现nn模型rmse数值是4.多，但是交上去是3.4了，可持续优化

nn模型我用了时序预测包，可直接安装使用，环境是tf2.4，numpy最好降到numpy==1.19.5。

HyperTS安装步骤

```python
conda install -c conda-forge prophet==1.0.1
pip install hyperts
pip install tensorflow==2.4.0
```

