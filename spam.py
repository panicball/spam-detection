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


def token_join(text_string):
    text = text_cleanup(text_string)
    text = " ".join(map(str, text))
    return text


def bag_of_words_vectorization(dataset):
    vectorization = CountVectorizer()
    bow_transformer = vectorization.fit(dataset)

    # # Fetch the vocabulary set
    # print(f"20 BOW Features: {vectorization.get_feature_names_out()[20:40]}")
    # print(f"Total number of vocab words: {len(vectorization.vocabulary_)}")

    # Convert strings to vectors using BoW
    messages_bow = bow_transformer.transform(dataset)

    # # Print the shape of the sparse matrix and count the number of non-zero occurrences
    # print(f"Shape of sparse matrix: {messages_bow.shape}")
    # print(f"Amount of non-zero occurrences: {messages_bow.nnz}")

    x = vectorization.fit_transform(dataset)
    x = x.toarray()

    df = pd.DataFrame(data=x, columns = vectorization.get_feature_names_out())
    # print(df)

    return messages_bow, bow_transformer


def tf_idef(messages_bow):
    # TF-IDF
    tfidf_transformer = TfidfTransformer().fit(messages_bow)

    # Transform entire BoW into tf-idf corpus
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    # print(messages_tfidf.shape)

    return tfidf_transformer, messages_tfidf  


def split_train_and_test_datasets(dataset, messages_tfidf):
    
    # Map ham label as 0 and spam as 1
    dataset['Category']= dataset['Category'].map({'ham': 0, 'spam': 1})
    # print(dataset.head())

    # Split the dataset to train and test sets
    msg_train, msg_test, label_train, label_test = train_test_split(
        messages_tfidf, dataset["Category"], test_size=0.2)

    # print(f"train dataset features size: {msg_train.shape}")
    # print(f"train dataset label size: {label_train.shape}")

    # print(f"test dataset features size: {msg_test.shape}")
    # print(f"test dataset label size: {label_test.shape}")

    return msg_train, msg_test, label_train, label_test


def WordCloud_plot(category_messages):

    category_words = " ".join(category_messages.Message.to_numpy().tolist())

    # wordcloud of spam messages
    category_messages_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS ,max_font_size=50, background_color ="black", colormap='Blues').generate(category_words)
    plt.figure(figsize=(16,10))
    plt.imshow(category_messages_cloud, interpolation='bilinear')
    plt.axis('off') # turn off axis
    plt.show()

def Words_Visualisation(dataset):
    spam_messages = dataset[dataset["Category"] == "spam"]
    ham_messages = dataset[dataset["Category"] == "ham"]

    # Most commonly appeared words in spam messages
    WordCloud_plot(spam_messages)
    # Most commonly appeared words in ham messages
    WordCloud_plot(ham_messages)


def messages_distribution_plot(dataset):

    plt.figure(figsize=(8,6))
    sns.countplot(data=dataset,  x=dataset.Category)


def confusion_matrix_plot(label_train, predict_train):

    cf_matrix = confusion_matrix(label_train, predict_train)

    ax= plt.subplot()
    #annot=True to annotate cells
    sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt='')

    # labels, title and ticks
    ax.set_xlabel('Predicted categories')
    ax.set_ylabel('True categories')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Not Spam', 'Spam']); ax.yaxis.set_ticklabels(['Not Spam', 'Spam'])



def classification(input_message):

    dataset = data_load()

    print(dataset.describe())
    dataset = dataset_cleanup(dataset)

    #print(dataset.describe())

    # messages_distribution_plot(dataset)

    dataset = downsampling(dataset)

    # messages_distribution_plot(dataset)
    # Words_Visualisation(dataset)

    # data_statistics(dataset)

    dataset["Message"] = dataset["Message"].apply(token_join)
    messages_bow, bow_transformer = bag_of_words_vectorization(dataset["Message"])
    tfidf_transformer, messages_tfidf = tf_idef(messages_bow)
    msg_train, msg_test, label_train, label_test = split_train_and_test_datasets(dataset, messages_tfidf)

            
    # Instantiate our model
    clf = XGBClassifier()

    # Fit the model to the training data
    clf.fit(msg_train, label_train)

    # Make predictions
    predict_train = clf.predict(msg_train)
    print(f"Accuracy of Train dataset: {metrics.accuracy_score(label_train, predict_train):0.3f}")

    confusion_matrix_plot(label_train, predict_train)

    # an example prediction
    # print("predicted:", clf.predict( tfidf_transformer.transform(bow_transformer.transform([input_message])))[0],)
    # print("expected:", dataset["Category"][9])

    # # print the overall accuracy of the model
    label_predictions = clf.predict(msg_test)
    print(f"Accuracy of the model: {metrics.accuracy_score(label_test, label_predictions):0.3f}")

    print("\n* Result:")
    if clf.predict( tfidf_transformer.transform(bow_transformer.transform([input_message])))[0] == 1:
        print('Spam detected.')
    else:
        print('Provided message is not spam.')

    



print("\n")
print("******************************************************")
print("****                                              ****")
print("****               SPAM DETECTION                 ****")
print("****                                              ****")
print("******************************************************")

print("\n* Please enter your message:")

input_text = ''

input_text = input("* Message:")

if input_text:
    prediction_text = input_text
else:
    prediction_text = "Our records indicate your Pension is under performing to see higher growth and up to 25% cash release reply PENSION for a free review. To opt out reply STOP"
    print(f'\n{prediction_text}')

input_message = token_join(prediction_text)

# print(input_message)

classification(input_message)


