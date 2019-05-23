from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.misc import imread

# 读取整个文本
text = open(r'C:\Users\mac\Desktop\gameofthrones.txt','r').read()

#读取背景图片
image=imread('C:/Users/mac/Desktop/3.png')

#去除停用词
stopworddic = set(stopwords.words('english'))
stopworddic.add('said')
stopworddic.add('would')
stopworddic.add('one')
stopworddic.add('hand')
stopworddic.add('back')
# 生成一个词云图像
wordcloud = WordCloud(background_color='white',stopwords=stopworddic,mask=image,contour_width=1,contour_color='gray').generate(text) #generate可以自动分词

# matplotlib的方式展示生成的词云图像
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()