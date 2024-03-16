
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


class analysis_toolbox:
    def __init__(self):
        import pandas as pd
        self.file_path = "/Users/ISASC_ST/Documents/Data Project/filtered_dataset.csv"
        self.df = pd.read_csv(self.file_path)

    def ML_predict(self, ethnicity):
        # Assuming 'African' is a binary target variable in your dataframe
        vectorizer = TfidfVectorizer(max_features=1000)
        x = vectorizer.fit_transform(self.df['name']).toarray()
        y = self.df[ethnicity]

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(x_train, y_train)

        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_resampled, y_resampled)

        # Evaluate the model
        y_pred = model.predict(x_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred, zero_division=0))


# Initialize and use the analysis toolbox
toolbox = analysis_toolbox()
toolbox.ML_predict("Asian")

