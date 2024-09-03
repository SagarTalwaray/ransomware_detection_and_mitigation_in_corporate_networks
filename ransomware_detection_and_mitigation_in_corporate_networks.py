# **Machine Learning Techniques for Ransomware Detection and Mitigation in Corporate Networks**
"""

# Commented out IPython magic to ensure Python compatibility.
#Importing necessary libraries
import pandas as pd
import numpy as np
from google.colab import drive
import io
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import RocCurveDisplay, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import warnings
warnings.filterwarnings('ignore')

# For data visualization
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from wordcloud import WordCloud

# Checking the shape of the dataframe
print(df.shape)

# Displaying the first few rows of the dataframe
df.head()

# Displaying the first few rows of the dataframe
df.tail()

# Using the column attribute to see what type of information is stored in the data.
df.columns

# By the info method we can check the Nan values and the datatype of all the columns in our dataframe
df.info(verbose= True)

# The describe() function applies basic statistical computations on the dataset like extreme values,count of data points etc.
df.describe(include='all').T

"""### **Duplicates:**"""

#Checking for the duplicated entries in the dataset.
MissV = len(df[df.duplicated()])
print("There are",MissV, "duplicate values.")

"""### **Missing values:**"""

#Sum of all the null values present in each column.
for i in df.columns.tolist():
  print("Total missing values in",i,":",df[i].isna().sum())

"""## **EDA:**"""

# Plot the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Benign', data=df)
plt.title('Distribution of Target Variable (Benign)')
plt.show()

# Plot the distributions of numerical features
num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_features.remove('Benign')

plt.figure(figsize=(15, 10))
for i, feature in enumerate(num_features, 1):
    plt.subplot(4, 5, i)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(feature)
plt.tight_layout()
plt.show()

"""## **Correlation Matrix:**"""

# Selecting only numeric columns for correlation matrix
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Compute the correlation matrix for numeric columns only
correlation_matrix = df[numeric_columns].corr()

# Plot the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

"""## **Outlier Detection:**"""

# Function to detect outliers using IQR method
def detect_outliers(df):
    outliers = pd.DataFrame(columns=df.columns)

    for col in df.describe().columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outliers = pd.concat([outliers, df.loc[outlier_indices]])

    return outliers

# Plotting boxplots for each numerical feature
for col in df.describe().columns:
    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    df.boxplot(column=col, ax=ax)
    ax.set_title('Boxplot of ' + col)
    plt.show()

# Detecting outliers in the dataset
outliers = detect_outliers(df)

# Display the detected outliers
print("Detected outliers:")
outliers

"""## **logarithmic transformation**"""

# df contains the numerical features along with target column containig 0 and 1.
numerical_features = [
    'DebugSize', 'DebugRVA', 'MajorImageVersion', 'MajorOSVersion', 'ExportRVA',
    'ExportSize', 'IatVRA', 'MajorLinkerVersion', 'MinorLinkerVersion',
    'NumberOfSections', 'SizeOfStackReserve', 'DllCharacteristics',
    'ResourceSize', 'BitcoinAddresses'
]

# Apply logarithmic transformation to handle wide range of values
log_transformed_df = df.copy()
for feature in numerical_features:
    log_transformed_df[feature] = np.log1p(df[feature])  # log1p is used to handle zeros in the data

# Setting the figure size
plt.rcParams['figure.figsize'] = (19, 6)

# Plotting the combined boxplot for log-transformed numerical features
ax = log_transformed_df[numerical_features].plot(kind='box', title='Combined Boxplot (Log Transformed)', showmeans=True)

# Display the plot
plt.show()

"""## **Non-Numeric Column Treatment:**"""

# 'FileName' and 'md5Hash' are the non-numeric columns
non_numeric_columns = ['FileName', 'md5Hash']

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), non_numeric_columns)
    ],
    remainder='passthrough'  # Leave the rest of the columns unchanged
)

"""## **CHecking for imbalance:**"""

# Plotting the pie chart to check the balance in the dataset.
plt.figure(figsize=(7, 5), dpi=100)
proportion = df['Benign'].value_counts()
labels = ['Malicious', 'Benign']  # Adjust labels if needed
plt.title('Proportion of Benign and Malicious for Target Feature')
plt.pie(proportion, explode=(0, 0.2), labels=labels, shadow=True, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
plt.legend()
plt.show()

# Checking the count of the classes in the target variable.
print(df['Benign'].value_counts())

"""## **Normalization:**

### **Using Standardscaler:**
"""

# Applying normalization operation for numeric stability
standardizer = StandardScaler()

# List of numerical columns to scale
columns_to_scale = [
    'DebugSize', 'DebugRVA', 'MajorImageVersion', 'MajorOSVersion', 'ExportRVA',
    'ExportSize', 'IatVRA', 'MajorLinkerVersion', 'MinorLinkerVersion',
    'NumberOfSections', 'SizeOfStackReserve', 'DllCharacteristics',
    'ResourceSize', 'BitcoinAddresses'
]

# Scaling the numerical columns
df[columns_to_scale] = standardizer.fit_transform(df[columns_to_scale])

"""## **Dropping unnecessary variable and splitting the target variable:**"""

# Remove non-numeric columns
df = df.drop(columns=['FileName', 'md5Hash'])
# Splitting the data into set of independent variables and a dependent variable.
X = df.drop('Benign', axis=1).values
y = df['Benign'].values

"""## **Train and Test Split**"""

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# Checking the shape of our train and test data.
print(X_train.shape)
print(X_test.shape)

"""# **Logistic Regression Model:**"""

# Define and train the logistic regression model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Predict on training data
y_pred_train = logistic_model.predict(X_train)

# Predict on test data
y_pred_test = logistic_model.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

"""### **Confusion Matrix:**"""

# Generate confusion matrix
r_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:\n", r_matrix)

# Print classification report for additional metrics
print("Classification Report:\n", classification_report(y_test, y_pred_test))

#Plotting the cofusion matrix.
labels = ['8299','5210','543','1570']

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(r_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix for Logistic Regression');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# Display the visualization of the Confusion Matrix.
plt.show()

"""# **XGBoost Model:**"""

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

# Define and train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and retrain the model
best_params = grid_search.best_params_
best_xgb_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
best_xgb_model.fit(X_train, y_train)

# Perform cross-validation on the entire dataset
cv_scores = cross_val_score(best_xgb_model, X, y, cv=5, scoring='accuracy')
# Print the cross-validation scores
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# Predict on training data
y_pred_train = best_xgb_model.predict(X_train)

# Predict on test data
y_pred_test = best_xgb_model.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

"""### **Confusion Matrix:**"""

# Generate confusion matrix and classification report for test data
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

# Print classification report for additional metrics
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

#Plotting the cofusion matrix.
labels = ['8819','6740','23','40']

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(r_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix for Logistic Regression');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# Display the visualization of the Confusion Matrix.
plt.show()

"""# **Random Forest Classifier:**"""

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the model
rf_model = RandomForestClassifier()

# Perform grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and retrain the model
best_params = grid_search.best_params_
best_rf_model = RandomForestClassifier(**best_params)
best_rf_model.fit(X_train, y_train)

# Predict on training data
y_pred_train = best_rf_model.predict(X_train)

# Predict on test data
y_pred_test = best_rf_model.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

"""### **Confusion Matrix**"""

# Generate confusion matrix and classification report for test data
conf_matrix = confusion_matrix(y_test, y_pred_test)
class_report = classification_report(y_test, y_pred_test)

print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

#Plotting the cofusion matrix.
labels = ['8820','6744','22','36']

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(r_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix for Logistic Regression');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# Display the visualization of the Confusion Matrix.
plt.show()

"""# **Gradient Boosting Model:**"""

# Define and train the GradientBoosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predict on training and test data
y_pred_train = gb_model.predict(X_train)
y_pred_test = gb_model.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

"""### **Confusion Matrix:**"""

# Generate confusion matrix
r_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:\n", r_matrix)

# Print classification report for additional metrics
print("Classification Report:\n", classification_report(y_test, y_pred_test))

#Plotting the cofusion matrix.
labels = ['8792','6688','50','92']

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(r_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix for Logistic Regression');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# Display the visualization of the Confusion Matrix.
plt.show()

"""#**Support Vector Machine:**"""

# Define and train the SVC model
svc_model = SVC(random_state=42)
svc_model.fit(X_train, y_train)

# Predict on training and test data
y_pred_train = svc_model.predict(X_train)
y_pred_test = svc_model.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

"""### **Confusion Matrix:**"""

# Generate confusion matrix
r_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:\n", r_matrix)

# Print classification report for additional metrics
print("Classification Report:\n", classification_report(y_test, y_pred_test))

#Plotting the cofusion matrix.
labels = ['8818','6743','24','37']

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(r_matrix, annot=labels, fmt='', cmap='Blues')

ax.set_title('Confusion Matrix for Logistic Regression');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');

# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

# Display the visualization of the Confusion Matrix.
plt.show()

# Plotting the table to compare the accuracy scores of all the models.

print('**** Comparison of  Models ****')
table = PrettyTable(['Model', 'Test Accuracy', 'Precision',  'F1_score', 'Recall'])
table.add_row(['Logistic regression', 0.86,	0.84,	0.94,	0.89])
table.add_row(['Random Forest Classifier', 0.99,	1.00,	1.00,	1.00])
table.add_row(['XGBoost Classifier', 0.99,	1.00,	1.00,	1.00])
table.add_row(['Support Vector Machine', 0.75,	0.700,	1.0,	0.82])
table.add_row(['Gradient Boosting Classifier', 0.99,	0.99,	0.99,	0.99])


print(table)