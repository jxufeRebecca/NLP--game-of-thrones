import nltk
#nltk.download('names')
import random
from nltk.corpus import names

#提取名字的最后一个字母，认为它含有性别的特征
def gender_feature(name):
    return {'last_letter': name[-1]}

#获得数据集
male_names = [(name, 'male') for name in names.words('male.txt')]
female_names = [(name, 'female') for name in names.words('female.txt')]
total_names = male_names + female_names
random.shuffle(total_names)  #shuffle函数可以实现对列表元素的随机排序


#生成性别特征集合
feature_set = [(gender_feature(n), g) for (n, g) in total_names]

# 将特征集拆分为训练集和测试集
train_set_size = int(len(feature_set) * 0.6)
train_set = feature_set[:train_set_size]#60%的训练集
test_set = feature_set[train_set_size:]#40%的测试集

#训练朴素贝叶斯分类器
classifier = nltk.NaiveBayesClassifier.train(train_set)

#预测权游中人名的性别
print(classifier.classify(gender_feature('Sansa')))
print(classifier.classify(gender_feature('John')))
print(classifier.classify(gender_feature('ned')))

#算法的精度
print(nltk.classify.accuracy(classifier, test_set))

#显示性别的特征
classifier.show_most_informative_features()