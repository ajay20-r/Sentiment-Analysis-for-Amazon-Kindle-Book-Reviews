#!/usr/bin/env python
# coding: utf-8

# # Amazon Kindle Reviews - Sentiment Analysis
# 
# #### Metadata: 
# - asin - ID of the product, like B000FA64PK
# - helpful - helpfulness rating of the review - example: 2/3.
# - overall - rating of the product.
# - reviewText - text of the review (heading).
# - reviewTime - time of the review (raw).
# - reviewerID - ID of the reviewer, like A3SPTOKDG7WBLN
# - reviewerName - name of the reviewer.
# - summary - summary of the review (description).
# - unixReviewTime - unix timestamp.
# 

# In[67]:


# Importing necessary libraries 
import warnings 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import re
import json 
import nltk
import spacy
import string
import unicodedata
from bs4 import BeautifulSoup
from textblob import TextBlob 
from nltk.stem import WordNetLemmatizer

from IPython import display 
display.set_matplotlib_formats('svg') # Setting the format for matplotlib plots
warnings.filterwarnings('ignore') # Ignoring warnings


# In[68]:


# Loading the dataset
data = pd.read_csv("C:\\Users\\iamaj\\Downloads\\all_kindle_review .csv")
data.head()


# Here, we are going to use only 2 columns Independent(reviewText) and Dependent(rating). Let's ignore all other columns.  

# In[69]:


# Selecting only the relevant columns (reviewText and rating)
data = data[['reviewText', 'rating']]
data.head()


# In[70]:


# Checking the shape of the dataset
data.shape


# In[71]:


# Checking the size of the dataset
data.size


# In[72]:


#  Getting descriptive statistics of the data
data.describe()


# In[73]:


# Checking for null values
data.isnull().sum()


# In[74]:


# Checking unique categories in the 'rating' column
data['rating'].value_counts()


# In[75]:


# Mapping ratings above 3 as 0 (positive) and ratings below 3 as 1 (negative)
data["rating"] = data["rating"].apply(lambda x: 1 if x < 3  else 0) # positive as 0 and negative as 1


# In[76]:


# Downloading necessary NLTK resources
nltk.download('wordnet')


# In[77]:


### from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image  
nltk.download('punkt')


# Function to tokenize the reviews 
def list_tokenizer(rating): 
    ratings = " ".join(rating)
    ratings = nltk.word_tokenize(ratings)
    return str(ratings)


# Function to generate word cloud visualization
def word_cloud(rating, number): 
    wc = WordCloud(background_color = 'black', max_font_size = 50, max_words = 100,stopwords=useless)  
    wc.generate(rating)  
    plt.figure(figsize=(10,8))
    plt.imshow(wc, interpolation = 'bilinear')  
    plt.title(f'WordCloud for {number}')
    plt.axis('off')
    plt.show();


# Now, we have suitable columns for model building, but before that we need to pre-process the text. Let's do that! 
# 
# ### Pre-processing

# Pre-processing the text data before model building

# In[78]:


# 1. Lowercasing the reviews
data['reviewText'] = data['reviewText'].str.lower()  


# In[79]:


# 2. Removing punctuation from the reviews
data['reviewText'] = data['reviewText'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', x))  # it removes the punctuation 


# In[80]:


# 3. Removing stopwords from the reviews

from collections import Counter
from nltk.corpus import stopwords
nltk.download('stopwords')
CustomStopWords= ['me','an','in','a','I','on','and','to',',',
                   'as','at','ok','the','?','of','but','it','.','!','-','from','my','is']
useless = stopwords.words('english') + CustomStopWords
data['reviewText']=data['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (useless)]))


# In[81]:


# 4. Removing HTML tags from the reviews
data['reviewText'] = data['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
                                              


# In[82]:


# 5. Removing emails from the reviews
data['reviewText'] = data['reviewText'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())


# In[83]:


# 6. Remove emails from the reviews
data['reviewText'] = data['reviewText'].apply(lambda x: re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', '', x))  # it will remove te emails 


# In[84]:


# 7. Removing any extra space
data['reviewText'] = data['reviewText'].apply(lambda x: " ".join(x.split()))


# In[85]:


# 8. Lemmatize the text to reduce words to their base or dictionary form

get_ipython().run_line_magic('time', '')
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return "".join([lemmatizer.lemmatize(word) for word in text])


# ### Visualization Using Word Cloud

# In[86]:


# Get all the review text values
values = [x for x in data['reviewText'] ]


# In[87]:


# Combine all the review text into a single string
comments =''
datas = [nltk.word_tokenize(values) for values in data['reviewText']] 
for x in datas:
    value = " ".join(x)
    comments = comments+value


# ### WordCloud for all the Reviews

# Creating word clouds for all reviews, reviews with ratings less than 3, and reviews with ratings greater than 3

# In[88]:


# Create a word cloud for all the reviews
from wordcloud import WordCloud
wc = WordCloud(background_color = 'black', max_font_size = 50, max_words = 100)   
wc.generate(comments)  
plt.figure(figsize=(10,8))
plt.imshow(wc, interpolation = 'bilinear')  
plt.axis('off')
plt.show();


# In[89]:


# Get the text based on ratings
rating_zero = data['reviewText'][data['rating'] == 0]
rating_one = data['reviewText'][data['rating'] == 1]


# ### Wordcloud for reviews with ratings less than 3

# In[90]:


# Create a word cloud for reviews with ratings less than 3
values = [x for x in rating_zero ]
comments =''
datas = [nltk.word_tokenize(value) for value in values] 
for x in datas:
    value = " ".join(x)
    comments = comments+value

wc = WordCloud(background_color = 'black', max_font_size = 50, max_words = 100)   
wc.generate(comments)  
plt.figure(figsize=(10,8))
plt.imshow(wc, interpolation = 'bilinear')  
plt.axis('off')
plt.show();


# ### Wordcloud for reviews with ratings greater than 3

# In[91]:


# Create a word cloud for reviews with ratings greater than 3
values = [x for x in rating_one ]
comments =''
datas = [nltk.word_tokenize(value) for value in values] 
for x in datas:
    value = " ".join(x)
    comments = comments+value
wc = WordCloud(background_color = 'black', max_font_size = 50, max_words = 100)   
wc.generate(comments)  
plt.figure(figsize=(10,8))
plt.imshow(wc, interpolation = 'bilinear')  
plt.axis('off')
plt.show();


# ### Text to Words
# 
# Now, we have pre-processed the file, let's start the model buidling. 
# 
# But before model building, we need to convert the text to numbers. So, let's do this by two methods `BOW` & `TF-IDF`~
# 
# 
# #### 1. Bag of Words 

# In[92]:


# Data split for model building 
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(data['reviewText'], data['rating'], test_size = 0.3)


# In[93]:


# Convert text to numbers using Bag of Words (BOW) 
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

# Convert the training data
xtrain_bow = vectorizer.fit_transform(xtrain).toarray()

# Convert the testing data
xtest_bow = vectorizer.transform(xtest).toarray()


# #### 2. TF-IDF 

# In[94]:


# Convert text to numbers using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer  

tf_vectorizer = TfidfVectorizer()

# Convert the training data 
xtrain_tf = tf_vectorizer.fit_transform(xtrain).toarray()

# Convert the testing data
xtest_tf = tf_vectorizer.transform(xtest).toarray()


# ### Model Building 
# 
# Here, we are going to use the `Gaussina NB` model. 
# 
# First we will see the results for `BOW` 

# In[95]:


from sklearn.naive_bayes import GaussianNB

# Using BOW 
clf_bow = GaussianNB().fit(xtrain_bow, ytrain)  # fitting 
prediction_bow = clf_bow.predict(xtest_bow)  # predictions

# Using TF-IDF 
clf_tf = GaussianNB().fit(xtrain_tf, ytrain)
prediction_tf = clf_tf.predict(xtest_tf)


# In[97]:


# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

def metrics(prediction, actual): 
    print('Confusion_matrix \n', confusion_matrix(actual, prediction))
    print('\nAccuracy:', accuracy_score(actual, prediction))
    print('\nclassification_report\n')
    print(classification_report(actual, prediction))
    
    
metrics(prediction_bow, ytest)


# In[98]:


metrics(prediction_tf, ytest)


# ## Sentiment Polarity

# In[99]:


# Sentiment Polarity
import nltk
nltk.download('wordnet')


# In[100]:


# Calculate polarity using TextBlob
data['reviewText']= data['reviewText'].astype(str) #Make sure about the correct data type
pol = lambda x: TextBlob(x).sentiment.polarity
data['polarity'] = data['reviewText'].apply(pol) # depending on the size of your data, this step may take some time.


# In[101]:


# Visualize polarity distribution
import matplotlib.pyplot as plt
import seaborn as sns
num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(data.polarity, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Number of Reviews')
plt.title('Histogram of Polarity Score')
plt.show();


# In[102]:


# Visualize the relationship between rating and polarity
plt.figure(figsize=(10,6))
sns.boxenplot(x='rating', y='polarity', data=data)
plt.show();


# In[103]:


# Print reviews with extreme values of polarity and subjectivity
data.loc[(data.polarity == 1 & (data.rating == 0))].reviewText.head(10).tolist()


# In[104]:


# Calculate subjectivity using TextBlob
sub = lambda x: TextBlob(x).sentiment.subjectivity
data['subjectivity'] = data['reviewText'].apply(sub)
data.sample(10)


# In[105]:


# Density Plot and Histogram of subjectivity
# Visualize subjectivity distribution
plt.figure(figsize=(10,5))
sns.distplot(data['subjectivity'], hist=True, kde=True,
bins=int(30), color = 'darkblue',
hist_kws={'edgecolor':'black'},
kde_kws={'linewidth': 4})

plt.xlim([-0.001,1.001])
plt.xlabel('Subjectivity', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
plt.title('Distribution of Subjectivity Score', fontsize=15)


# In[106]:


# Visualize the relationship between polarity and subjectivity, colored by rating
plt.figure(figsize=(10,6))
sns.scatterplot(x='polarity', y='subjectivity', hue="rating", data=data)
plt.xlabel('Polarity', fontsize=13)
plt.ylabel('Subjectivity', fontsize=13)
plt.title('Polarity vs Subjectivity', fontsize=15)
plt.show();


# In[107]:


# Print reviews with extreme values of polarity and subjectivity for further examination
data.loc[(data["rating"] == 0) & (data.polarity == 1 ) & (data.subjectivity ==1), "reviewText"].head(10).tolist()


# # Conclusion
# 

# 
# I analyzed the data and created word clouds to visually represent the most frequent words in the reviews. I chose Gaussian Naive Bayes as the model, which had an accuracy of around 58%. Most of the reviews had a neutral sentiment with a slight inclination towards positivity. To address negative feedback, further investigation is needed. This analysis provided insights into customer experiences and areas for improvement.

# In[ ]:




