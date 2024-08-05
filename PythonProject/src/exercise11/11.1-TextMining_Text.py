import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared.data_processing import etl_text_file_

# Download necessary data for NLTK (stopwords etc)
nltk.download('punkt')
nltk.download('stopwords')

# Set display options
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_colwidth', None)  # Display full content of each cell


def tokenize_text(text):
    """
    Tokenize a given text into words using NLTK's word_tokenize.

    Args:
        text (str): Input text to tokenize.

    Returns:
        list: List of tokens (words) extracted from the input text.
    """
    return word_tokenize(text)


def stem_words(words):
    """
    Stem a list of words using NLTK's PorterStemmer.

    Args:
        words (list): List of words to stem.

    Returns:
        list: List of stemmed words.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


def lowercase_words(words):
    """
    Convert a list of words to lowercase.

    Args:
        words (list): List of words.

    Returns:
        list: List of words converted to lowercase.
    """
    return [word.lower() for word in words]


def remove_stopwords(words):
    """
    Remove stopwords from a list of words using NLTK's stopwords corpus.

    Args:
        words (list): List of words.

    Returns:
        list: List of words with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]


def remove_special_characters(text):
    """
    Remove special characters and punctuation from a given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with special characters and punctuation removed.
    """
    return ''.join([char for char in text if char not in string.punctuation])


def create_bag_of_words(texts):
    """
    Create a bag-of-words representation from a list of texts using CountVectorizer.

    Args:
        texts (list): List of texts.

    Returns:
        numpy.ndarray: Bag-of-words representation of the texts.
    """
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(texts).toarray()


def preprocess_text(text):
    """
    Preprocess a given text by tokenizing, stemming, converting to lowercase,
    removing stopwords, and removing special characters.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    tokens = tokenize_text(text)
    tokens = stem_words(tokens)
    tokens = lowercase_words(tokens)
    tokens = remove_stopwords(tokens)
    cleaned_text = ' '.join(tokens)
    cleaned_text = remove_special_characters(cleaned_text)
    return cleaned_text


textData1 = etl_text_file_('resources/data/text1.txt', '\t', 'text')
textData2 = etl_text_file_('resources/data/text2.txt', '\t', 'text')

# Apply preprocessing to each DataFrame
textData1['cleaned_text'] = textData1['text'].apply(preprocess_text)
textData2['cleaned_text'] = textData2['text'].apply(preprocess_text)

# Create bag of words representation for each DataFrame
bag_of_words_df1 = create_bag_of_words(textData1['cleaned_text'])
bag_of_words_df2 = create_bag_of_words(textData2['cleaned_text'])

# Display bag of words matrices
print("Bag of Words Matrix for DataFrame 1:")
print(pd.DataFrame(bag_of_words_df1))
print("\nBag of Words Matrix for DataFrame 2:")
print(pd.DataFrame(bag_of_words_df2))

# Combine preprocessed text data into a list
texts = textData1['cleaned_text'].tolist() + textData2['cleaned_text'].tolist()

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the text data
bag_of_words = vectorizer.fit_transform(texts)

# Convert bag-of-words to DataFrame (optional)
bag_of_words_df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())

# Display bag-of-words DataFrame
print("Bag-of-Words Representation:")
print(bag_of_words_df)

# Split bag_of_words back into separate parts for df1 and df2
bag_of_words_df1 = bag_of_words[:len(textData1)]
bag_of_words_df2 = bag_of_words[len(textData2):]

# Calculate cosine similarity between documents
cosine_similarities = cosine_similarity(bag_of_words_df1, bag_of_words_df2)

# Print cosine similarity
print("Cosine Similarity between Document 1 and Document 2:")
print(cosine_similarities)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Split tfidf_matrix back into separate parts for df1 and df2
tfidf_matrix_df1 = tfidf_matrix[:len(textData1)]
tfidf_matrix_df2 = tfidf_matrix[len(textData2):]

# Optionally, convert tfidf_matrix to a DataFrame (for visualization purposes)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display TF-IDF DataFrame (optional)
print("TF-IDF Representation:")
print(tfidf_df)

# Calculate cosine similarity between the first two documents
cosine_similarities = cosine_similarity(tfidf_matrix_df1, tfidf_matrix_df2)

# Print cosine similarity
print("Cosine Similarity between Document 1 and Document 2:")
print(cosine_similarities)
