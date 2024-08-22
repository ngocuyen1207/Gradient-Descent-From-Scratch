import pandas as pd
from cuml.model_selection import train_test_split
import cupy as cp
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')
sys.path.append('/home/uyen/smoke_detection_training/')

def load_and_preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path, index_col=0).dropna(axis=0)
    categorical_cols = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']
    numerical_cols = ['YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']
    vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    transformer = ColumnTransformer([
        ('vectorizer',vectorizer,'HaveWorkedWith'),
        ('encoder', one_hot_encoder,categorical_cols),
        ('scaler', scaler, numerical_cols)
        ], remainder="passthrough")
    X = df.drop([target_column],axis=1)
    X = transformer.fit_transform(X).toarray()
    y = df[target_column]
    
    # Train-validation-test split
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

    return cp.array(X_train), cp.array(y_train.values), cp.array(X_val), cp.array(y_val.values), cp.array(X_test), cp.array(y_test.values)
