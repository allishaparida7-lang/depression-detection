import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
csv_file = r"C:\Users\91934\Desktop\project minor\classified.csv"
df = pd.read_csv(csv_file)

# Encode class labels if not numeric
if df['class_label'].dtype == 'object':
    le = LabelEncoder()
    df['class_label'] = le.fit_transform(df['class_label'])


# Select numeric feature columns only (exclude target and metadata)
X = df[['ch0_psd_mean_asym','ch0_theta_power_asym','ch0_alpha_power_asym','ch0_beta_power_asym','ch0_alpha_theta_ratio_asym','ch0_alpha_beta_ratio_asym','ch0_entropy_asym','ch0_fractal_dim_asym','ch1_psd_mean_asym','ch1_theta_power_asym','ch1_alpha_power_asym','ch1_beta_power_asym','ch1_alpha_theta_ratio_asym','ch1_alpha_beta_ratio_asym']]
X = X.select_dtypes(include=['float64', 'int64'])  # keep only numeric
y = df['class_label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline: imputation -> scaling -> SVM
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# Define hyperparameter grid for GridSearch
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.1, 1],
    'svm__kernel': ['rbf', 'linear', 'poly']
}

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
