import streamlit as st
import time
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, MaxAbsScaler
from utilities import create_pipeline_object, create_df
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import streamlit as st
import streamlit.components.v1 as components
from sklearn.utils import estimator_html_repr
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from pprint import pprint
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

retraining_flag = True

# Title and subtitle
st.write("""
# Twitter Sentiment Analysis
#### Building a Classification Model for Twitter Sentiment Analysis using Sklearn.
"""
)
try:
    df = pd.read_csv('data/train_data.csv')
    print('df loading successful')
except:
    print('Something went wrong in loading data.')

# add introduction about the data 
st.write("""
### Introduction
- Many of us scroll through twitter first thing in the morning. 
- Tweets in Twitter can convey different sentiments to people and it is interesting to build a model that can identify the sentiment of a tweet beforehand.
- This project is about building a classification model on tweets to identify whether they are Positive or Negative.
- The dataset is from Analytics Vidhya website open competitions.
""")

# ADD SEPARATOR
st.write("""
-----
""")


# Feature explanation tab
int_tab_1, int_tab_2 = st.tabs(
    [
        'Text Features', 'Numerical Features',
    ]
)
with int_tab_1:
    st.write("""
    #### There are three types of textual features
    - i will be using straighforward methods like tfidfvectorizer and countvectorizer to extract features from preprocessed text.
    - i used the below mentioned preprocessing steps:
        - removing urls mentioned in the tweets
        - removing usernames mentioned as @ and replacing them with just username
        - removing and replacing all multiple whitespaces with single white space
        - remove topics mentioned as #some_topic. we extract them and use them as a separate feature.
    
    ##### tweet 
    - the tweet is itself a feature, i will be using a tfidf vectorizer to extract features from the original tweet.
    
    ##### topics
    - so i used the hashtags mentioned in a tweet as topics. i just extracted them and again use a tfidfvectorizer to extract features from them.
    
    ##### emojis
    - the emojis present in the tweets are also good features. after extracting i convert them into features using countvectorizer.
    
    """)

with int_tab_2:
    st.write("""
    #### There are five types of numerical features. These features are extracted from text only.
    
    ##### Length of Tweet
    - Basically this is the number of words in a Tweet.

    ##### Number of Topics
    - Total number of hashtags mentioned in the tweet.     
    
    ##### Number of Emojis
    - Total number of emojis present in the tweet.

    ##### Number of Slurrs 
    - Total number of slurrs or abusive words mentioned in the tweet.

    ##### Emoji Score
    - Using scoring dict each emoji is given a score.
    - This score is aggregated sum for all the emojis in the tweet. 
    
    """)

# ADD SEPARATOR
st.write("""
-----
""")

# show label distribution
st.write("""
### Label Distribution And Preparing Train and Test Data
- We wont be using Validation set as we will be using K-fold cross validation for parameter tuning.
- Below is the distribution of the labels in the entire dataset.
- We can see that the dataset is not balanced.
- Next step to split dataframe into train and test. 
""")
chart_1 = alt.Chart(df).mark_bar(size=10).encode(
    x='label:N',
    y='count()',
).properties(
    width=1000,
    height=500
)
st.altair_chart(chart_1, use_container_width=True)

with st.sidebar:
    # ask for test and valid percentage
    train_test_split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 70, 90, 80, 5)
    retraining_flag = True

# split data into train, test and validation
train_df, test_df = train_test_split(
    df, test_size=(100 - train_test_split_size)/100.0, shuffle=True,
    random_state=12, stratify=df['label']
)
st.write(
    'Train Data has : {} tweets and Test Data has : {} tweets. Lets get to modelling.'.format(train_df.shape[0], test_df.shape[0])
)

# ask which text and numerical features to use to users
st.write("""
#### Please select the Numerical Features you are interested in using for model fitting.
""")
all_numeric_features = [
    'num_topics', 'num_emojis', 'length_of_tweet',
    'num_of_slurrs', 'emoji_score'
]
all_textual_features = ['tweet', 'label', 'topics', 'extracted_emojis'] 
numerical_features_to_use = st.multiselect(
    'Numerical Features to use for modelling :',
    all_numeric_features, all_numeric_features
)

all_columns = all_textual_features + numerical_features_to_use 
train_df = train_df[all_columns]
test_df = test_df[all_columns]

# give options for scaling the numerical columns (standard, minmax , max abs)
st.write("""
#### Please select which scaling to use to preprocess numerical features:
""")
scaling_type = st.radio(
    'Select the type of Scaling :',
    ('StandardScaler', 'MinMaxScaler', 'MaxAbsSclaer')
)
if scaling_type == "StandardScaler":
    scaler = StandardScaler()
elif scaling_type == "MinMaxScaler":
    scaler = MinMaxScaler()
else:
    scaler = MaxAbsScaler()



# ask which classifiers to Fit for Voting Classifier
st.write("""
#### Please select the classifiers to be used inside the Voting Classifer.
- A Voting Classifier takes the output of the all the classifiers that are selected and gives that label as output which has the majority vote among the classifiers.
""")
all_classifiers = [
    'logistic', 'svm', 'random_forest', 'knn', 'ada_boost'
]
classifiers_to_use = st.multiselect(
    'Numerical Classifiers to use for modelling :',
    all_classifiers, all_classifiers
)

# import pipeline object
classifier_pipeline = create_pipeline_object(
    scaler, numerical_features_to_use, classifiers_to_use 
)
with open("pipeline_image.html", "w") as f:
    f.write(estimator_html_repr(classifier_pipeline))

# giving time for the html to save before loading
time.sleep(5)
# showing classifier_pipeline
# by rendering the html of the pipeline in an iframe.
st.write("""
#### Currently our Sklearn Pipeline looks like below:
""")
HtmlFile = open("pipeline_image.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code, height=400)


# ask for kfolds to use 
st.write("""
#### Please select K-Fold value to tune hyper parameters using GridSearchCv :
""")
kfold_value = st.radio(
    '',
    (3, 5, 8, 9, 10)
)


# ask for scoring metric to use 
st.write("""
#### Please select Scoring Methods for Evaluation:
""")
scoring_metric_option = st.radio(
    '',
    ('Accuracy', 'F1', 'ROC_AUC')
)



# show the heyperparameter and ask user to fill
# tfidf_ranges = ['(1, 1)', '(1, 2)', '(1, 3)']
# general_tfidf__tfidf__ngram_range = st.multiselect(
#     'Select the Text TFIDF Ngram Ranges you want to experiment with :',
#     tfidf_ranges, tfidf_ranges
# )

# min_df_ranges = ['0.0', '0.05', '0.1']
# general_tfidf__tfidf__min_df = st.multiselect(
#     'Select the Min_DF TFIDF Vocab Ranges you want to experiment with :',
#     min_df_ranges, min_df_ranges
# )

# max_df_ranges = ['0.85', '0.9', '0.95']
# general_tfidf__tfidf__ngram_range = st.multiselect(
#     'Select the Max_DF Text TFIDF Vocan Ranges you want to experiment with :',
#     max_df_ranges, max_df_ranges
# )

# split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 50, 90, 80, 5)


# classifier params 

parameters = {
    # tweet text features
    # "preprocessor__text__tweet_tfidf__tfidf__ngram_range": [(1, 1)],
    # "preprocessor__text__tweet_tfidf__tfidf__min_df": [5],
    # "preprocessor__text__tweet_tfidf__tfidf__max_df": [0.90],
    
    # topic text features
    # "preprocessor__text__topic_tfidf__tfidf__ngram_range": [(1, 1)],
    # "preprocessor__text__topic_tfidf__tfidf__min_df": [2],
    # "preprocessor__text__topic_tfidf__tfidf__max_df": [0.90],
    
    # emoji text features
    # "preprocessor__text__emoji_tfidf__tfidf__ngram_range": [(1, 1)],
    # "preprocessor__text__emoji_tfidf__tfidf__min_df": [1],
    # "preprocessor__text__emoji_tfidf__tfidf__max_df": [0.90],
}

with st.sidebar:
    st.sidebar.write('---')
    st.sidebar.subheader('Learning Parameters')
    
    # for logistic regression
    if 'logistic' in classifiers_to_use:
        logistic_C_values = [0.1, 1, 10]
        logistic_c = st.multiselect(
            'Select the values of C for logistic regression you want to experiment with :',
            logistic_C_values, [1]
        )
        logistic_penalty_options = ['l2', 'elasticnet']
        logistic_penalty = st.multiselect(
            'Select the values of penalty for logistic regression you want to experiment with :',
            logistic_penalty_options, ['l2']
        )
        logistic_params_dict = {
            "classifier__logistic__C": logistic_c,
            'classifier__logistic__penalty': logistic_penalty
        }
        parameters.update(logistic_params_dict)

    # for svm
    if 'svm' in classifiers_to_use:
        svm_kernel = st.sidebar.selectbox(
            'Select SVM Kernel ', ('linear', 'poly', 'rbf')
        )
        svm_C_options = [0.1, 1, 10]
        svm_c = st.multiselect(
            'Select the values of C for SVM you want to experiment with :',
            svm_C_options, [1]
        )
        svm_params_dict = {
            'classifier__svm__C': svm_c,
            'classifier__svm__kernel': [svm_kernel]
        }
        parameters.update(svm_params_dict)
    
    # for random forest
    if 'random_forest' in classifiers_to_use:
        rf_n_estimators = st.sidebar.slider(
            'Number of estimators for Random Forest (n_estimators)',
            0, 500, (10,50), 50
        )
        rf_max_depth = st.sidebar.slider(
            'Random Forest Maximum depth', 5, 15, (5,8), 2
        )
        rf_criterion = st.sidebar.selectbox(
            'Random Forest criterion',('gini', 'entropy')
        )
        rf_params_dict = {
            'classifier__random_forest__n_estimators': [
                i for i in range(rf_n_estimators[0], rf_n_estimators[1], 50)
            ],
            'classifier__random_forest__max_depth': [
                i for i in range(rf_max_depth[0], rf_max_depth[1], 3)
            ],
            'classifier__random_forest__criterion': [rf_criterion],
        }
        parameters.update(rf_params_dict)
        
    # for knn
    if 'knn' in classifiers_to_use:
        knn_parameter_neighbors = st.sidebar.number_input(
            'KNN Select the Number of Neighbors:', 2, 10
        )
        knn_params_dict = {
            'classifier__knn__n_neighbors': [knn_parameter_neighbors]
        }
        parameters.update(knn_params_dict)
            
    # for ada boost
    if 'ada_boost' in classifiers_to_use:
        ab_n_estimators = st.sidebar.slider(
            'Number of estimators for  AdaBoost',
            0, 500, (10,50), 50
        )
        ab_learning_rate = st.sidebar.slider(
            'Ada Boost learning Rate', 0, 10, 1, 1)
        
        adaboost_params_dict = {
            'classifier__ada_boost__n_estimators': [
                i for i in range(ab_n_estimators[0], ab_n_estimators[1], 50)
            ],
            'classifier__ada_boost__learning_rate': [ab_learning_rate]
        }
        parameters.update(adaboost_params_dict)
    retraining_flag = True

st.write("""
    #### The parameters dict is printed below :
""")
st.write("{}".format(parameters))


# train model ; show animation
if retraining_flag == True:
    clf = GridSearchCV(
        classifier_pipeline,
        parameters, 
        cv=kfold_value,
        scoring=scoring_metric_option.lower()
    )
    with st.spinner('Please Wait while the model is training.....'):
        if retraining_flag == True:
            clf.fit(train_df, train_df["label"])
    retraining_flag = False




st.success('Done!')

# show best model and parameters
st.write("""
### Trained Model Best Score and parameters are mentioned below :
""")
st.write("Best Score achieved is : {}".format(clf.best_score_))
st.write("Best Parameters according to GridSearchCV are : ")
st.write("{}".format(clf.best_params_))

# get results on test set 
with st.spinner('Please Wait while the Inference is going on the Test data....'):
    test_df['predictions'] = list(clf.predict(test_df))
st.success('Done!')
st.write('------')

st.write("""
#### The Results on the Test Set are: 
""")
model_accuracy = accuracy_score(test_df['label'], test_df['predictions']).round(2)
model_precision = precision_score(
    test_df['label'], test_df['predictions'],
    labels=clf.classes_).round(2)

model_recall = recall_score(
    test_df['label'], test_df['predictions'],
    labels=clf.classes_).round(2)

model_f1_score = f1_score(
    test_df['label'], test_df['predictions'],
    labels=clf.classes_).round(2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "{}".format(model_accuracy))
col2.metric("Precision", "{}".format(model_precision))
col3.metric("Recall", "{}".format(model_recall))
col4.metric("F1 Score", "{}".format(model_f1_score))

st.subheader("Confusion Matrix") 
cm = confusion_matrix(test_df['label'], test_df['predictions'], labels=clf.classes_)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=clf.classes_
)
disp.plot()
st.pyplot()

# optional plot learning curve
# give option to input sentence and get output.
st.write("""
### Please go ahead and try your own Tweet below : 
""")

with st.form(key='my_form'):
    input_text = st.text_input('Tweet', 'We love this! Would you go? #talk #makememories #unplug #relax #iphone #smartphone #wifi #connect')
    st.write('The input tweet is : ', input_text)
    submit_button = st.form_submit_button(label='Submit')
    temp_df = create_df(input_text)
    prediction = list(clf.predict(temp_df))[0]
    if prediction == '0':
        output = 'Negative'
    else:
        output = 'Positive'

    st.write("The prediction for the provided tweet is : {}".format(output))
