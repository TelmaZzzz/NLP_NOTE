## 任务一：基于TextCNN的文本分类
数据集：[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

参考论文：Convolutional Neural Networks for Sentence Classification，https://arxiv.org/abs/1408.5882

需要了解的知识点：

1. 文本特征表示：词向量
* 对word embedding随机初始化
* 用glove预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/
2. CNN如何提取文本的特征
模型图：
![](https://github.com/TelmaZzzz/NLP_NOTE/tree/master/NLP_prectice/prectice1/TextCnn.png)


说明：

1. 训练集25000句，测试集25000句，需要自己写脚本合在一起；
2. 请将训练集用于训练，测试集用于验证，最后我会再给你一个测试集；
3. 测试结果格式：每行对应一句话的分类结果；
当前的SOTA排名：https://github.com/sebastianruder/NLP-progress/blob/master/english/sentiment_analysis.md