import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Downloading NLTK data
nltk.download('stopwords')  # Downloading stopwords data

# Read data
df = pd.read_csv('C:/Users/aymen/Desktop/projet semestriel/spam.csv', encoding='latin1')
df2 = pd.read_csv('C:/Users/aymen/Desktop/projet semestriel/spam_or_not_spam.csv', encoding='latin1')
df3 = pd.read_csv('C:/Users/aymen/Desktop/projet semestriel/spam_ham_dataset.csv', encoding='latin1')

# Display basic info about the datasets
df.info()
df2.info()
df3.info()

# Assuming 1 represents 'spam' and 0 represents 'ham'
df2['label'] = df2['label'].apply(lambda x: 'spam' if x == 1 else 'ham')

# Handle missing values
df = df.dropna()
df2 = df2.dropna()
df3 = df3.dropna()

# Select relevant columns
df = df[['Message', 'Category']]
df = df.rename(columns={'Message': 'email', 'Category': 'label'})
df2 = df2[['email', 'label']]
df3 = df3[['text', 'label']]
df3 = df3.rename(columns={'text': 'email'})

# Concatenate datasets
final_dataset = pd.concat([df[['email', 'label']], df2[['email', 'label']], df3[['email', 'label']]], axis=0,
                          ignore_index=True)

# Display information about the final dataset
final_dataset.info()

# Encode labels
encoder = LabelEncoder()
final_dataset['label'] = encoder.fit_transform(final_dataset['label'])

# Drop duplicates
final_dataset = final_dataset.drop_duplicates(keep='first')

# Display class distribution
values = final_dataset['label'].value_counts()
total = values.sum()
percentage_0 = (values[0] / total) * 100
percentage_1 = (values[1] / total) * 100
print('percentage of ham:', percentage_0)
print('percentage of spam:', percentage_1)

# Pie chart
colors = ['#FF5733', '#33FF57']
explode = (0, 0.1)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('white')

wedges, texts, autotexts = ax.pie(
    values, labels=['ham', 'spam'],
    autopct='%0.2f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    explode=explode,
    shadow=True
)

for text, autotext in zip(texts, autotexts):
    text.set(size=14, weight='bold')
    autotext.set(size=14, weight='bold')

ax.set_title('Email Classification', fontsize=16, fontweight='bold')
ax.axis('equal')

plt.show()

# Create a CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the text data
word_matrix = vectorizer.fit_transform(final_dataset['email'])

# Convert to a DataFrame for better understanding
word_df = pd.DataFrame(word_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Concatenate with the original dataset
final_dataset = pd.concat([final_dataset, word_df], axis=1)

# Display the dataset with word features
final_dataset.head()
final_dataset.info()

# Features and target variable
X = final_dataset.drop(['email', 'label'], axis=1)
y = final_dataset['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Drop rows with NaN values
X_train = X_train.dropna().reset_index(drop=True)
y_train = y_train.reset_index(drop=True)  # Update y_train accordingly
X_test = X_test.dropna().reset_index(drop=True)  # Drop NaN values from the test set as well
y_test = y_test.reset_index(drop=True)  # Update y_test accordingly
# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
