NLP杂货铺，python实现nlp相关算法、工具，辅助理解相关算法

# 目录

## base_algorithm
基础算法实现，如排序、树等等。    
已实现部分：    
```
sort.py  # 排序算法
```

## base_task
基础任务，如文本分类、文本匹配、序列标注、文本生成等任务。    
已包含部分：    
```
classification.py  # 文本分类任务
```

## framework
框架工具使用，主要会展示tensorflow/pytorch/docker/shell等工具的简单使用。      
已实现部分:    
```
tensorflow2/  # tf2.0的简单教程 
    0_start.py  # 快速入门教程
    1_load_data.py  # dataset,tf_record相关内容
    2_variable_tensor  # 变量、张量、梯度计算、各种operations
    3_customize_layer_model  # 自定义层与模型
    4_train_evaluate  # 内置的模型训练与评估
    5_customize_train  # 自定义训练
    6_customize_other  # 自定义callback、loss、metrics
hadoop.sh  # hadoop的简单使用
```
## papers
论文算法复现，已实现部分：    
```
text_fooler.py  # Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment
word2vec.py # Efﬁcient Estimation of Word Representationsin Vector Space
fasttext.py # Bag of Tricks for Efficient Text Classification和Enriching Word Vectors with Subword Information
textcnn.py # Convolutional Neural Networks for Sentence Classification
```
## pattern_match
模式识别，包含正则的使用，kmp，trie树等算法的实现。    
已实现部分：    
```
regular_expression.py  #正则的使用
kmp.py  #kmp算法
trie.py  #trie树
prefix_suffix_trie.py    #前后缀树
double_array_trie.py  #双数组trie树
acam.py  #ac自动机
```
## read_source
源码阅读        
已实现部分：    
```
bert/  #bert相关源码阅读
    modeling.py  # bert模型部分源码注释
    run_pretraining.py  # pretrain部分源码注释
    optimization.py  #optimization部分源码注释
    tokenization.py  #tokenization部分源码注释
    create_pretraining_data.py  #create_pretraining_data部分源码注释
    extract_feature.py  #extract_feature部分源码注释
```
## statistical_algorithm
统计学习算法实现，包含逻辑回归、支持向量机、条件随机场等等。    
已实现部分:    
```
knn.py  #K近邻
nb.py  #朴素贝叶斯
perceptron.py  #感知机
logistic_regress.py  #逻辑回归
svm.py  #支持向量机
hmm.py  #隐马尔科夫
crf.py  #条件随机场
tree/  #树相关模型
```

# TODO
> 1. Tensorflow2.0学习教程
>   1. GPU与分布式训练
>   1. 性能分析与优化
>   1. 模型保存
>   1. 服务部署
>   1. 其他待定
> 1. BERT源码阅读
>   1. run_classifier.py
>   1. run_squad.py
> 1. 论文阅读与复现
>   1. word2vec（已完成）
>   1. fasttext（已完成）
>   1. TextCNN （已完成）
>   1. 其他待定
> 1. 其他待定