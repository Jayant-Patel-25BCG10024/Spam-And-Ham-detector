# Spam Classifier Project



This project implements a machine learning pipeline for classifying text messages as either Ham (legitimate) or Spam (unsolicited). It uses TF-IDF Vectorization for text featurization and compares the performance of two popular classifiers: Logistic Regression and Multinomial Naive Bayes (MNB).



### Setup and Installation



Follow these steps to set up the project environment and run the spam classifier.



#### 1\. Prerequisites



You need Python 3.6 or higher installed on your system.



#### 2\. Required Libraries



All necessary libraries can be installed using pip.



pip install pandas scikit-learn



The key libraries used are:



pandas: Essential for data loading and manipulation.



scikit-learn: Provides the machine learning models, the TF-IDF feature extraction tool, and evaluation metrics.



#### 3\. Data File and Path Configuration



The script requires a dataset, typically the SMS Spam Collection, in CSV format.



**File Name:** The script is configured to look for a specific file path defined in the main execution block.



**Format:** The CSV file must have no header row and contain at least two columns: the Label (e.g., ham or spam) and the Text (the message content).



###### Action Required:

You must update the data\_file variable in the if \_\_name\_\_ == "\_\_main\_\_": block with the correct, absolute path to your dataset.



\# Main Execution Block Snippet

if \_\_name\_\_ == "\_\_main\_\_":



    #**ACTION:** Replace this placeholder path with the actual location of your 'spam\_data.csv' file.

    data\_file = 'C:/Users/Jayant Patel/Desktop/VScode/Project/spam\_data.csv'



    # ... rest of the code

#### 

### Code Details and Architecture



The spam classification pipeline is logically divided into four main steps.



**Step 0:** Data Loading and Preparation (load\_data)



This function handles the input data cleanup and transformation.



**-Input:** File path of the raw CSV data.



**-Pre-processing:**



1.Loads data using header=None.



2.Renames columns 0 $\\rightarrow$ label and 1 $\\rightarrow$ text.



3.Label Encoding: Converts categorical labels to numerical targets: 'ham' ---> 0 (Negative Class) and 'spam' --->1 (Positive Class).



4.Performs data splitting (80% train, 20% test) using train\_test\_split with stratify=y to ensure an even distribution of spam/ham in both sets.



##### Step 1: Feature Engineering (featurize\_text)



Raw text is converted into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.



**Tool:** sklearn.feature\_extraction.text.TfidfVectorizer



**Parameters:**



max\_features=5000: Limits the vocabulary size to the top 5000 most informative words.



stop\_words='english': Removes common English words that do not contribute to classification.



**Process:**



The vectorizer is fitted on the training data (X\_train) to learn the vocabulary and Inverse Document Frequency (IDF) weights.



The vectorizer is then used to transform both the training and testing sets into sparse matrices of TF-IDF scores.



##### Step 2: Model Training (train\_lr\_model, train\_nb\_model)



Two distinct classifiers are trained on the vectorized training data (X\_train\_vec, y\_train).



**Logistic Regression:** This is a Linear model for binary classification, which is good for interpretability. It uses the solver='liblinear' parameter, which is efficient for small to medium datasets.



**Multinomial Naive Bayes:** This is a Probabilistic model well-suited for text counts and frequencies. It is known to be fast and robust, and works effectively with TF-IDF features using its default parameters.

##### 

##### Step 3: Model Evaluation (evaluate\_model)



Performance is assessed on the held-out test data (X\_test\_vec, y\_test).



###### Metrics Calculated:



**Accuracy:** Overall prediction correctness.



**Precision (Spam):** Ratio of correctly predicted spam to all messages predicted as spam. (Minimizes False Positives / Ham going to Spam folder).



**Recall (Spam):** Ratio of correctly predicted spam to all actual spam messages. (Minimizes False Negatives / Spam leaking into Inbox).



**F1 Score:** Harmonic mean of precision and recall, providing a balanced measure of performance.



##### Step 4: Prediction (predict\_new\_email)



This function demonstrates using the trained model (defaults to Naive Bayes in the main script) to classify new, unseen messages.



-New messages are passed through the same fitted vectorizer before being fed to the classifier for prediction.



Running the Pipeline



The if \_\_name\_\_ == "\_\_main\_\_": block executes the entire pipeline:



1.Loads and splits the data.



2.Runs TF-IDF feature engineering.



3.Trains both LR and NB models.



4.Evaluates and prints metrics for both models.



5.Predicts labels for a small list of custom example emails.



**To run the script:**



python your\_script\_name.py

