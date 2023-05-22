import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def spam_filter(csv_file_path):
    # Access and read csv data
    data = pd.read_csv(csv_file_path)
    data.drop(['Unnamed: 0', 'label_num'], axis=1, inplace=True)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    print(data)

    X = data["text"]
    Y = data["label"]

    # Facing an issue here regards stopwords
    # (raise ValueError(ValueError: empty vocabulary; perhaps the documents only contain stop words)
    # Initialize CountVectorizer with stopwords
    # vect = CountVectorizer() not work if text contains stopwords
    vect = CountVectorizer(stop_words='english')

    # Fit and transform the text data
    X = vect.fit_transform(X)

    # Split the dataset into training and testing sets:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train the Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, Y_train)

    # Predict the labels for test data
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)

    # Take user input for new email
    new_email = input("Enter a new email: ")
    X_new = vect.transform([new_email])
    new_prediction = model.predict(X_new)
    print("Prediction:", new_prediction)
    if new_prediction==1:
        print("This is a Spam Mail")
    else:
        print("This is a Ham Mail")

# Example usage:
csv_file_path = "spam.csv"
spam_filter(csv_file_path)
