'''
install the below packages through command prompt
pip install nltk
'''

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Example: Text preprocessing
text = "NLTK provides tools for processing text, performing tokenization, stemming, lemmatization, and removing stop words, crucial for text data preprocessing in NLP tasks."
stop_words = set(stopwords.words('english'))
filtered_text = [word for word in text.split() if word.lower() not in stop_words]
print("Filtered Text:", filtered_text)

# NLTK provides tools for processing text, performing tokenization, stemming, lemmatization, and removing stop words, crucial for text data preprocessing in NLP tasks.