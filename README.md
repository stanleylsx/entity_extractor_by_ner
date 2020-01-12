# Sequence-Labeling
此项目是基于Tensorflow的NER任务项目，使用BiLSTM+Crf模型，支持使用Bert做Embedding，提供可配置文档，配置完直接运行。
## Brief introduction
将Embedding之后的向量输入到BiLSTM层中得到输出层的Softmax，将Softmax值放到条件随机场（CRF层）求解出最大可能的转移模式序列。  
这个项目中默认的Embedding使用词/字表结果放到BiLSTM层，也可以接入Word2vec/Bert之后的值放到BiLSTM层做Embedding，相关配置在system.config/use_pre_trained_embedding，注意和BiLSTM的输入维度对齐即可。  
需要使用Bert做Embedding需要可参考文章[BERT Pre-training...](/papers/2018/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf)。  
BiLSTM层输出的Softmax可以直接做NER识别，因为BiLSTM模型输出的结果是词/字对应各类别的分数，我们可以选择分数最高的类别作为预测结果，但是会造成一些问题，所以加入CRF层，具体比较好的一篇文章请参考[最通俗易懂的BiLSTM-CRF模型中的CRF层介绍](https://zhuanlan.zhihu.com/p/44042528)和[CRF Layer on the Top of BiLSTM - 1](https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/)。  
CRF层需要使用viterbi译码法，知乎上[这个答案](https://www.zhihu.com/question/20136144)比较容易理解。  
![model](img/model.png)  

## Download project and install
```
git clone https://github.com/StanleyLsx/sequence_labeling.git
pip install -r requirements.txt
```

## Update history
Date|Version|Details
:---|:---|---
2020-01-12|v1.0.0|initial project

## How to use
### Train
### Batch test
### Online predict
## Reference