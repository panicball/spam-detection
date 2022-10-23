import string
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns


# Dataset from https://www.kaggle.com/datasets/team-ai/spam-text-message-classification

def data_load():
    dataset = pd.read_csv("Data.csv", na_values=['nan', 'NaN', 'NULL', 'null'], keep_default_na=False)
    # print(dataset.head())
    return dataset


def dataset_cleanup(dataset):
    #print(dataset.describe())

    # print('Sum of Null values in each column: ')
    # print(dataset.isnull().sum(), '\n')
    dataset = dataset.dropna()
        
    duplicatedRow = dataset[dataset.duplicated()]
    # print(duplicatedRow[:15])

    if not duplicatedRow.empty:
        dataset.drop_duplicates(subset=None, inplace=True)

    # print(dataset.describe())

    #print(dataset.groupby('Category').describe().T)

    return dataset.reset_index(drop=True)


def downsampling(dataset):     
    spam_messages = dataset[dataset["Category"] == "spam"]
    ham_messages = dataset[dataset["Category"] == "ham"]

    ham_msg_df = ham_messages.sample(n = len(spam_messages), random_state = 44)
    spam_msg_df = spam_messages

    #print(ham_msg_df.shape, spam_msg_df.shape)

    msg_df = pd.concat([ham_msg_df, spam_msg_df]).reset_index(drop=True)
    return msg_df


def data_statistics(dataset):

    spam_messages = dataset[dataset["Category"] == "spam"]["Message"]
    ham_messages = dataset[dataset["Category"] == "ham"]["Message"]


    #print("Dataset composition:")
    #print(f"Number of spam messages: {len(spam_messages)}")
    #print(f"Number of ham messages: {len(ham_messages)}")

    #print((len(spam_messages)/len(ham_messages))*100)

    # Next, we check the top ten words that repeat the most in both ham and spam messages. 
    # Download stopwords
    # nltk.download('stopwords')

    # Words in spam messages
    spam_words = []
    for each_message in spam_messages:
        spam_words += text_cleanup(each_message)
        
    print(f"Top 10 spam words are:\n {pd.Series(spam_words).value_counts().head(10)}")

    # Words in ham messages
    ham_words = []
    for each_message in ham_messages:
        ham_words += text_cleanup(each_message)
      
    print(f"Top 10 ham words are:\n {pd.Series(ham_words).value_counts().head(10)}")


def text_cleanup(text_string):
    text = "".join([ch for ch in text_string.lower() if ch not in string.punctuation])
    text = [i for i in word_tokenize(text) if i not in stopwords.words("english") and i.isalpha()]
    return text


