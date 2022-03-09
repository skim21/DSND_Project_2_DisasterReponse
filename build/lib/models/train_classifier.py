import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import sqlite3
from sklearn import multioutput
from sklearn.metrics import fbeta_score
from scipy.stats import gmean

from pprint import pprint
import os.path

import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    # connect to the database
    conn = sqlite3.connect(database_filepath)

    # run a query
    df = pd.read_sql('SELECT * FROM clean_data', conn)

    # define features and label arrays
    X = df.message.values
    y = df.drop(['message', 'original'], axis=1).values
    category_names = df.drop(['message', 'original'], axis=1).columns

    return X, y, category_names


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf',  multioutput.MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def multioutput_f1score(y_true, y_pred, category_names, verbose=True):
    """
    MultiOutput Fscore

    """
    
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta=0.5,average='weighted')
        score_list.append(score)
    if verbose:
        print("    F1 Score for each category:\n")
        pprint(pd.Series(np.array(score_list), index=category_names))
    f1score = np.asarray(score_list)
    f1score = f1score[f1score<1]
    f1score_mean = gmean(f1score)
    return  f1score_mean


def evaluate_model(model, X_test, Y_test, category_names):
    # output model test results
    Y_pred = model.predict(X_test)
    f1_score = multioutput_f1score(Y_test, Y_pred, category_names)
    print(f"    F1 geometric mean Score: {f1_score}")


def save_model(model, model_filepath):
    # Export model as a pickle file
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        if os.path.isfile(model_filepath):
            model  = joblib.load(model_filepath)
        else:
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, Y_train)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')
                
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)


    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
