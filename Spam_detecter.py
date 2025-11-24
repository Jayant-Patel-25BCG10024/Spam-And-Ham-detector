import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- STEP 0: Load and Prepare Data ---
def load_data(filepath):
    """
    Loads, renames, and cleans a standard spam/ham dataset.
    Assumes structure: Col 1 = Label (ham/spam), Col 2 = Text.
    """
    try:
        # Load the data, using header=None to ensure pandas reads the first row as data
        # The popular SMS Spam Collection often lacks a true header, or uses v1/v2
        df = pd.read_csv(filepath, encoding='latin-1', header=None) 
        
        # Rename columns to standard names for processing
        df = df.rename(columns={0: 'label', 1: 'text'})
        
        # Drop extra columns that are often empty in these datasets
        df = df[['label', 'text']]

        # Convert labels to numerical format: 0 for 'ham', 1 for 'spam'
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Drop any rows where the conversion failed (i.e., invalid labels became NaN)
        df.dropna(subset=['label'], inplace=True) 
        df['label'] = df['label'].astype(int)
        
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please update the path.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- STEP 1: Feature Engineering (TF-IDF Vectorization) ---
def featurize_text(X_train, X_test):
    """Converts text into numerical feature vectors using TF-IDF."""
    # Use max_features and stop_words for efficient feature selection
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, vectorizer

# --- STEP 2: Model Training Functions ---

def train_lr_model(X_train_vec, y_train):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_vec, y_train)
    return model

def train_nb_model(X_train_vec, y_train):
    """Trains a Multinomial Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

# --- STEP 3: Model Evaluation ---
def evaluate_model(model, X_test_vec, y_test, model_name):
    """Evaluates the model and prints key metrics."""
    y_pred = model.predict(X_test_vec)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    print(f"\n--- {model_name} Performance Metrics ---")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision (Spam): {metrics['Precision']:.4f}")
    print(f"Recall (Spam): {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    
    return metrics

# --- STEP 4: Prediction Function ---
def predict_new_email(text_list, vectorizer, model):
    """Predicts the label for new, unseen email text."""
    new_text_vec = vectorizer.transform(text_list)
    prediction = model.predict(new_text_vec)
    
    print("\n--- New Email Predictions ---")
    for text, pred in zip(text_list, prediction):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"Text: '{text[:50]}...' -> Prediction: {label}")

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # 🚨 ACTION: Replace this placeholder path with the actual location of your 'spam_data.csv' file.
    data_file = 'C:/Users/Jayant Patel/Desktop/VScode/Project/spam_data.csv' 
    
    data = load_data(data_file)
    
    if data is not None and not data.empty:
        # Split data into features (X) and target (y)
        X = data['text']
        y = data['label']
        
        # Split data into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 1. Feature Engineering (TF-IDF)
        X_train_vec, X_test_vec, vectorizer = featurize_text(X_train, X_test)
        
        # 2a. Train Logistic Regression
        classifier_lr = train_lr_model(X_train_vec, y_train) 
        
        # 2b. Train Multinomial Naive Bayes
        classifier_nb = train_nb_model(X_train_vec, y_train)
        
        # 3. Model Evaluation and Comparison
        results_lr = evaluate_model(classifier_lr, X_test_vec, y_test, "Logistic Regression")
        results_nb = evaluate_model(classifier_nb, X_test_vec, y_test, "Naive Bayes")
        
        # 4. Test with custom examples
        new_emails = [
            "Please confirm the meeting time for the project proposal tomorrow morning.", # Ham
            "URGENT! You have won a FREE gift card prize. Click link to claim now!!!" # Spam
        ]
        # Using Naive Bayes for the final prediction, as it's often slightly better for this task
        predict_new_email(new_emails, vectorizer, classifier_nb)