import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('wordnet')

from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import matplotlib as mpl


text = open(r'C:\Users\mac\Desktop\gameofthrones.txt','r').read()

#1、分词:拆成一个一个的单词和符号，并保存为列表（把长句子，拆成有“意义”的小部件）
tokens = nltk.word_tokenize(text)  #问题1里面有很多符号如 ‘’ . , 。。。
                                  #问题2 词形不同、大小写不同 会默认为两个不同的词
#2、单词小写化
words = [w.lower() for w in tokens]


#3、正则表达式去符号，只保留单词
def azfilter(w):
  pattern = re.compile('[^a-z]+')#匹配所有非a-z的字母的字符
  return pattern.sub('',w)
cwords=[azfilter(w) for w in words]

clean_words=list(filter(None, cwords))#把空字符删了

#4、常见词形归一化有两种方式(词干提取与词形归并）
# 词形归并(把各种类型的变形都归为一个形式，比如went归一为go，are归一为be，注意：要先进行词性标注)，词干提取（把不影响词性的inflection的小尾巴砍掉比如walking砍成walk）
#4.1词性标注
tags=nltk.pos_tag(clean_words) #嵌套元组的列表

#4.2 词形归并  注意：tag.startswith(t) 测试是否t开头
def lemma(word,tag):
   wordnet_lemmatizer=WordNetLemmatizer()
   if tag.startswith('NN'):
         return wordnet_lemmatizer.lemmatize(word,pos='n')
   elif tag.startswith('VB'):
         return wordnet_lemmatizer.lemmatize(word, pos='v')
   elif tag.startswith('JJ'):
         return wordnet_lemmatizer.lemmatize(word, pos='a')
   elif tag.startswith('R'):
         return wordnet_lemmatizer.lemmatize(word, pos='r')
   else:
         return word
final_words=[]
for word,tag in tags:
    final_words.append(lemma(word,tag))
print(final_words)


#词频统计
dist = FreqDist(final_words)#字典
#dist.plot(10) 绘制文中前10个高频词的折线图

#权游主要人物的出现频率
mpl.rcParams['font.sans-serif'] = ['SimHei'] #让matplotlib绘图可以显示中文
x=['ned','jon','arya','sansa','bran','tywin','tyrion','cersei','jaime','joffrey','dany']
y=[]
for i in x:
    y.append(dist[i])
plt.barh(x, y, align="center", color="#66c2a5", tick_label=["纳德", "雪诺", "二丫", "三傻", "布兰","泰温","小恶魔",'色曦','詹姆','乔佛里','龙妈'])
plt.xlabel("出现次数")
plt.grid(True, axis="x", ls=":", color="r", alpha=0.3)
plt.show()