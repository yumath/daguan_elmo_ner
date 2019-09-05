# daguan_elmo_ner
“达观杯”的比赛代码

### 赛事官网

https://biendata.com/competition/datagrand/

### Requirements:

> Tensor flow = 1.12
>
> h5py

### Run

1. 比赛官网下载corpus.txt，放入data目录下
2. 运行create_train_corpus.ipynb创建训练数据
3. 运行setup.py安装bilm
4. 执行bin/train.py，预训练自己的“深度双向语言模型”
5. 执行bin/dump_weights.py将训练好的语言模型参数转换成hdf5格式，并将options.json中n_characters改为262
6. 或者跳过前面5个步骤(还是需要第3步安装一下bilm)，直接下载预训练好的语言模型，放到output目录下。百度云链接：链接:https://pan.baidu.com/s/1PQLThtAqYH0DffgN2GgHAg 密码:2iyr
7. 运行train.py，训练基于Elmo的命名实体识别模型
8. 运行predict.py，完成测试集的结果输出，输出为比赛要求的格式

### Result

线上F-1值91.8

### Reference

1. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

2. [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)