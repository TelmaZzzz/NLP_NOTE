## 基于BiLSTM+CRF的序列标注
用BiLSTM+CRF来训练序列标注模型（Batch_size=1版本）

数据集：[CONLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/)

参考论文：[Neural Architectures for Named Entity Recognition](https://arxiv.org/pdf/1603.01360.pdf)

数据说明：

* CoNLL2003_NER文件夹里是数据部分，已经分好train、validation、test
* 测试结果格式：每行对应一句话的标注结果，词之间用空格相分隔；

当前的SOTA排名：https://github.com/sebastianruder/NLP-progress/blob/master/english/named_entity_recognition.md

程序说明：

* 运行`run.sh`即可训练模型，程序log请查看train.log
* `main.py`主要为数据处理部分
* `train.py`训练、评估代码
* `model.py`BiLSTM+CRF模型实现部分