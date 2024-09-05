#!/usr/bin/env python
# coding: utf-8

# In[141]:


#Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# for text processing
import re
import nltk

# for removing common words
from nltk.corpus import stopwords

# for word stemming
from nltk.stem.porter import PorterStemmer

# for text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer


# In[142]:


# Ignore all warnings

import warnings

warnings.filterwarnings('ignore')


# In[225]:


# Load the dataset

df = pd.read_csv('fakejob.csv')


# In[144]:


#show top 5 entries of the dataset

df.head()


# In[145]:


#show bottom 5 entries of the dataset


df.tail()


# In[146]:


print("Dataset Shape:", df.shape)
print("Dataset Info:",df.info())
print("Fraudulent Value Counts:")
print(df['fraudulent'].value_counts())


# In[147]:


print("Null Values:\n",df.isnull().sum())


# In[148]:


# Drop the 'job_id' column

df = df.drop(columns=['job_id'])


# In[149]:


#Dataset without job_id column

df.head()


# In[150]:


df.describe()


# In[151]:


df.duplicated().sum()


# In[152]:


df = df.drop_duplicates()


# In[153]:


df.duplicated().sum()


# In[154]:


# Calling text Columns

p= df.select_dtypes(include='object').columns


# In[155]:


object_columns_df = df.select_dtypes(include='object')


# In[156]:


object_columns_df.head()


# In[157]:


# Calling Numerical Columns

df.select_dtypes(include='int').columns


# In[158]:


int_columns_df = df.select_dtypes(include='int')


# In[159]:


int_columns_df.head()


# In[160]:


fake = df[df['fraudulent'] == 1]


# In[161]:


fake.head(2)


# In[162]:


real = df[df['fraudulent'] == 0]


# In[163]:


real.head(2)


# In[164]:


# Count the occurrences of unique values in the 'fraudulent' column

df['fraudulent'].value_counts()


# In[165]:


fake.shape


# In[166]:


real.shape


# In[167]:


df.shape


# In[168]:


# Count the occurrences of unique values in the 'fraudulent' column

fraudulent_counts = df['fraudulent'].value_counts()

# Plot the counts using a rainbow color palette

plt.figure(figsize=(8, 6))
sns.barplot(x=fraudulent_counts.index, y=fraudulent_counts.values, palette='rainbow')
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.title('Counts of Fraudulent vs Non-Fraudulent')
plt.show()


# In[169]:


# Count the occurrences of unique values in the 'telecommuting' column

telecommuting_counts = df['telecommuting'].value_counts()

# Plot the counts using a magma color palette

plt.figure(figsize=(8, 6))
sns.barplot(x=telecommuting_counts.index, y=telecommuting_counts.values, palette='magma')
plt.xlabel('telecommuting')
plt.ylabel('Count')
plt.title('Counts of telecommuting vs Non-telecommuting')
plt.show()


# In[170]:


# Count the occurrences of unique values in the 'has_company_logo' column

has_company_logo_counts = df['has_company_logo'].value_counts()

# Plot the counts

plt.figure(figsize=(8, 6))
sns.barplot(x=has_company_logo_counts.index, y=has_company_logo_counts.values)
plt.xlabel('has_company_logo')
plt.ylabel('Count')
plt.title('Counts of With_Logo vs Without_Logo')
plt.show()


# In[171]:


# Count the occurrences of unique values in the 'has_questions' column

has_questions_counts = df['has_questions'].value_counts()

# Plot the counts

plt.figure(figsize=(8, 6))
sns.barplot(x=has_questions_counts.index, y=has_questions_counts.values, palette='cividis')
plt.xlabel('has_questions')
plt.ylabel('Count')
plt.title('Counts Questions vs NO_Questions')
plt.show()


# In[172]:


# visualization of Distribution of employment_type

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

#we use countplot for Distribution of employment_type

sns.countplot(x='employment_type', data=df, palette="pastel", order=df['employment_type'].value_counts().index)

plt.title('Distribution of employment_type')

plt.show()


# In[173]:


# Count the occurrences of unique values in the 'has_questions' column

required_experience_counts = df['required_experience'].value_counts()

# Plot the counts

plt.figure(figsize=(12, 6))
sns.barplot(x=required_experience_counts.index, y=required_experience_counts.values, palette='viridis')
plt.xlabel('required_experience')
plt.ylabel('Count')
plt.title('Distribution of required_experience')
plt.xticks(rotation=45)
plt.show()


# In[174]:


# Count the occurrences of unique values in the 'required_education' column

required_education_counts = df['required_education'].value_counts()

# Plot the counts

plt.figure(figsize=(40, 5))
sns.barplot(x=required_education_counts.index, y=required_education_counts.values, palette='rainbow')
plt.xlabel('required_education')
plt.ylabel('Count')
plt.title('Distribution of required_education')
plt.show()


# In[175]:


# Count the occurrences of unique values in the 'department' column

department_counts = df['department'].value_counts()

# Select the top 20 most frequent occurrences

top_20_departments = department_counts.head(20)

# Plot the top 50 most frequent occurrences as horizontal bar plot

plt.figure(figsize=(14, 10)) 
sns.barplot(y=top_20_departments.index, x=top_20_departments.values, palette='magma')
plt.ylabel('Department')
plt.xlabel('Count')
plt.title('Top 20 Most Frequent Departments')
plt.show()


# In[176]:


# Count the occurrences of unique values in the 'industry' column

industry_counts = df['industry'].value_counts()

# Select the top 20 most frequent occurrences

top_20_industry = industry_counts.head(20)

# Plot the top 20 most frequent occurrences as horizontal bar plot with rotated labels

plt.figure(figsize=(14, 10)) 
sns.barplot(y=top_20_industry.index, x=top_20_industry.values, palette='rainbow')
plt.ylabel('Industry')
plt.xlabel('Count')
plt.title('Top 20 Most Frequent Industries')
plt.show()


# In[177]:


# Count the occurrences of unique values in the 'function' column

function_counts = df['function'].value_counts()

# Select the top 10 most frequent occurrences

top_10_function = function_counts.head(10)

# Plot the top 10 most frequent occurrences as horizontal bar plot with rotated labels

plt.figure(figsize=(14, 10)) 
sns.barplot(y=top_10_function.index, x=top_10_function.values)
plt.ylabel('Industry')
plt.xlabel('Count')
plt.title('Top 10 Most Frequent function')
plt.show()


# In[227]:


# Split the 'location' column into separate columns for country, state, and city

location_split = df['location'].str.split(', ', expand=True)
df['Country'] = location_split[0]
df['City'] = location_split[1]


# In[228]:


df.head(2)


# In[229]:


# Count the occurrences of unique values in the 'City' column

City_counts = df['City'].value_counts()

# Select the top 10 most frequent occurrences

top_10_City = City_counts.head(10)

# Plot the top 10 most frequent occurrences as horizontal bar plot with rotated labels

plt.figure(figsize=(14, 10)) 
sns.barplot(y=top_10_City.index, x=top_10_City.values, palette='rainbow')
plt.ylabel('City')
plt.xlabel('Count')
plt.title('Top 10 Most Frequent City')
plt.show()


# In[230]:


# Count the occurrences of unique values in the 'Country' column
Country_counts = df['Country'].value_counts()

# Select the top 10 most frequent occurrences
top_10_Country = Country_counts.head(10)

# Plot the top 10 most frequent occurrences as horizontal bar plot with rotated labels
plt.figure(figsize=(14, 10))
sns.barplot(y=top_10_Country.index, x=top_10_Country.values, palette='viridis')
plt.ylabel('Country')
plt.xlabel('Count')
plt.title('Top 10 Most Frequent Country')
plt.show()


# In[182]:


df


# In[183]:


df.isnull().sum()


# In[184]:


df.head()


# In[185]:


#fill empty space

df.fillna(" ", inplace=True)


# In[186]:


df.isnull().sum()


# In[187]:


df.head()


# In[188]:


# List of columns to concatenate

columns_to_concat = ['title', 'location', 'department', 'salary_range', 'company_profile',
                     'description', 'requirements', 'benefits', 'employment_type',
                     'required_experience', 'required_education', 'industry', 'function']

# Concatenate the values of specified columns into a new column 'job_posting'

df['job_posting'] = df[columns_to_concat].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Create a new DataFrame with columns 'job_posting' and 'fraudulent'

new_df = df[['job_posting', 'fraudulent']].copy()


# In[189]:


new_df.head(10)


# In[190]:


# text preprocessing(cleaning)

def preprocess_text(text):
    
    # Convert to lowercase
    
    text = text.lower()
    
    # Remove URLs
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove special characters
    
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Remove punctuation
    
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove digits
    
    text = re.sub(r'\d', '', text)
    
    # Remove stop words
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    text = ' '.join(words)
    
    return text

# Apply the combined function to the 'job_posting' column

new_df['job_posting'] = new_df['job_posting'].apply(preprocess_text)


# In[191]:


new_df.head(10)


# In[192]:


Fake = new_df[new_df['fraudulent'] == 1]


# In[193]:


Fake.head()


# In[194]:


Real = new_df[new_df['fraudulent'] == 0]


# In[195]:


Real.head()


# In[196]:


new_df.columns


# In[197]:


from nltk.tokenize import word_tokenize, sent_tokenize

# Tokenize each job posting into words

new_df['job_posting_tokens'] = new_df['job_posting'].apply(word_tokenize)

# Tokenize each job posting into sentences

new_df['job_posting_sentences'] = new_df['job_posting'].apply(sent_tokenize)


# In[198]:


# Display the DataFrame after tokenization

new_df.head()


# In[199]:


from nltk import pos_tag

# perform POS tagging

def pos_tagging(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

# Apply POS tagging to each sentence in 'job_posting_sentences' column

new_df['job_posting_pos_tagged'] = new_df['job_posting_sentences'].apply(lambda x: [pos_tagging(sentence) for sentence in x])


# In[200]:


# diplay DataFrame after applying POS

new_df.head()


# In[201]:


#Function to generate n-grams from a list of tokens

from nltk.util import ngrams

def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))


# Define the value of n for the n-grams

n = 2 

# Apply n-gram generation to the 'job_posting_tokens' column

new_df['job_posting_ngrams'] = new_df['job_posting_tokens'].apply(lambda x: generate_ngrams(x, n))


# In[202]:


# diplay DataFrame after applying N-Grams

new_df.head()


# In[203]:


new_df.columns


# In[204]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(new_df['job_posting_tokens'], new_df['fraudulent'], test_size=0.2, random_state=42)


# In[205]:


# the purpose that we need one squence the has key feature from each text in column 'job_posting_tokens' for task classification  

from sklearn.feature_extraction.text import CountVectorizer

# Flatten the list of lists into a single list of strings

X_train_flattened = [' '.join(sublist) for sublist in X_train]

# Vectorize the text data

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_flattened)

# Flatten the list of lists into a single list of strings for test data

X_test_flattened = [' '.join(sublist) for sublist in X_test]
X_test_vec = vectorizer.transform(X_test_flattened)


# In[206]:


X_train_flattened[0]


# In[207]:


X_train


# In[208]:


# Ensure the shape of X_train_vec matches the number of samples in X_train

print("Shape of X_train_vec:", X_train_vec.shape)
print("Number of samples in X_train:", X_train.shape[0])

# Ensure the shape of X_test_vec matches the number of samples in X_test

print("Shape of X_test_vec:", X_test_vec.shape)
print("Number of samples in X_test:", X_test.shape[0])


# In[209]:


# Split the new data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_train_vec, y_train, test_size=0.2, random_state=42)


# In[210]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn. neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score, roc_curve

def confusionmatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    color = 'white'
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    disp.plot()
    plt.title("Data Model")
    plt.xlabel('Predicted value', color=color)
    plt.ylabel('True Value', color=color)
    plt.show()
    print(classification_report(y_test, y_pred))

def naive_bayes(X_train, X_test, y_train, y_test):
    cl = MultinomialNB()
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    y_pred_prob = cl.predict_proba(X_test)[:, 1]
    confusionmatrix(y_test, y_pred)

def logistic_regresssion(X_train, X_test, y_train, y_test):
    cl = LogisticRegression()
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    confusionmatrix(y_test, y_pred)


def svm(X_train, X_test, y_train, y_test):
    cl = SVC()
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    confusionmatrix(y_test, y_pred)


def rfc(X_train, X_test, y_train, y_test):
    cl = RandomForestClassifier()
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    confusionmatrix(y_test, y_pred)


def knn(X_train, X_test, y_train, y_test):
    cl = KNeighborsClassifier()
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    confusionmatrix(y_test, y_pred)


def mlp(X_train, X_test, y_train, y_test):
    cl = MLPClassifier()
    cl.fit(X_train, y_train)
    y_pred = cl.predict(X_test)
    confusionmatrix(y_test, y_pred)


# In[211]:


naive_bayes(X_train, X_test, y_train, y_test)


# In[212]:


logistic_regresssion(X_train, X_test, y_train, y_test)


# In[213]:


svm(X_train, X_test, y_train, y_test)


# In[214]:


rfc(X_train, X_test, y_train, y_test)


# In[215]:


knn(X_train, X_test, y_train, y_test)


# In[216]:


mlp(X_train, X_test, y_train, y_test)


# In[217]:


from scipy.stats import mannwhitneyu, chi2_contingency

plotvar=df.drop('fraudulent',axis=1)
de = df['fraudulent']

new_data1 = data[data['fraudulent']==0]
new_data2 = data[data['fraudulent']==1]

mw_goodpred = []
chi_goodpred = []
sim = {}
sim1 = {}
for i in plotvar.columns:
    stat, p= mannwhitneyu(x=new_data1[i], y=new_data2[i], alternative = 'two-sided')
    if p<0.05:
        sim[p] = i
        mw_goodpred.append((i,p))
for i in plotvar.columns:
    contigency= pd.crosstab(plotvar[i], de)
    c, p, dof, expected = chi2_contingency(contigency)
    if p<0.05:
        sim1[p] = i
        chi_goodpred.append((i,p))
myKeys = list(sim.keys())
myKeys.sort()
sorted_dict = {i: sim[i] for i in myKeys}
myKeys1 = list(sim1.keys())
myKeys1.sort()
sorted_dict1 = {i: sim1[i] for i in myKeys1}
print("Man Whitney Test result :\n",mw_goodpred)
print("Man Whitney After Sorting :\n",sorted_dict.values())
print("Chi Square Test result :\n",chi_goodpred)
print("Chi Square After sorting :\n",sorted_dict1.values())


# In[218]:


#feature importance graph
from sklearn.datasets import make_classification
import seaborn as sb

cl = MLPClassifier()
cl.fit(X_train_vec, y_train)
importances = cl.feature_importances_
sb.barplot(x=importances, y= plotvar)
plt.show()


# In[ ]:


import pandas as pd

# Create a DataFrame

df = pd.DataFrame(data)

# Calculate Pearson correlation
correlation = df['X_test'].corr(df['y_test'])

