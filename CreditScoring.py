import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score, roc_curve, classification_report


def load_and_validate_data(file_path):
    """Load and validate the dataset."""
    df = pd.read_csv(file_path)
    print("Dataset Overview:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    return df

def feature_engineering(df):
    """Advanced feature engineering for credit risk assessment."""
    
    df['Debt_Income_Ratio'] = df['Loan Amount'] / (df['Age'] * 10000)
    df['Credit_Experience_Ratio'] = df['Number of Credit Accounts'] / df['Age']
    df['Interest_Load'] = df['Loan Amount'] * df['Interest Rate'] / 100
    
    df['Long_Term_High_Rate'] = ((df['Loan Term'] > 36) & 
                                 (df['Interest Rate'] > 8)).astype(int)
    df['Young_Borrower'] = (df['Age'] < 25).astype(int)
    df['Senior_Borrower'] = (df['Age'] > 60).astype(int)
    df['High_Utilization'] = (df['Credit Utilization Ratio'] > 0.7).astype(int)
    
    df['Normalized_Payment_History'] = df['Payment History'] / df['Credit Utilization Ratio']
    
    df['Loan_per_CreditAccount'] = df['Loan Amount'] / (df['Number of Credit Accounts'] + 1)

    employment_risk = df['Employment Status'].isin(['Unemployed'])
    utilization_risk = df['High_Utilization'] == 1
    education_risk = df['Education Level'].isin(['High School'])
    loan_type_risk = df['Type of Loan'].isin(['Personal Loan'])
    young_risk = df['Young_Borrower'] == 1

    df['Risk_Score'] = (
        employment_risk.astype(int) * 0.3 +
        utilization_risk.astype(int) * 0.25 +
        education_risk.astype(int) * 0.15 +
        loan_type_risk.astype(int) * 0.15 +
        young_risk.astype(int) * 0.15
    )

    df['Credit_Risk'] = np.where(
        df['Risk_Score'] >= df['Risk_Score'].quantile(0.4), 
        1, 
        0
    )

    return df



def create_preprocessing_pipeline():
    """Create preprocessing pipeline with ColumnTransformer."""
    num_features = ['Age', 'Credit Utilization Ratio', 'Number of Credit Accounts',
                   'Loan Amount', 'Interest Rate', 'Debt_Income_Ratio',
                   'Credit_Experience_Ratio']
    cat_features = ['Type of Loan']
    ordinal_features = ['Education Level']
  
    edu_order=['High School','Bachelor','Master','PhD']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             cat_features),
            ('ord', OrdinalEncoder(categories=[edu_order],
                                 handle_unknown='use_encoded_value',
                                 unknown_value=-1), 
             ordinal_features),
        ]
    )
    
    return preprocessor

def create_classification_pipeline(preprocessor):
    """Create complete classification pipeline."""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',RandomForestClassifier(
                                                        n_estimators=150,   
                                                        max_depth=10,        
                                                        min_samples_split=5,    
                                                        min_samples_leaf=2,      
                                                        max_features='sqrt',    
                                                        class_weight='balanced',
                                                        random_state=42,             
                                                        n_jobs=-1                
                                                )
        )
    ])
    return pipeline

def evaluate_model(pipeline, x_train, y_train, x_test, y_test):
    """Evaluate model performance."""
    # Train and predict
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_pred_proba=pipeline.predict_proba(x_test)[:,1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    report=classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    scores = cross_val_score(pipeline, x_train, y_train, cv=5)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Classification Report:\n {report}")
    print(f"AUC-ROC: {auc:.2f}")
    print(f"Cross-validated Accuracy: {scores.mean():.2f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Feature importance analysis
    feature_importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    
    #  Sort by importance
    sorted_idx = np.argsort(feature_importances)

    plt.figure(figsize=(10, 8)) 
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        feature_importances[sorted_idx],
        color='skyblue'
    )
    plt.xlabel('Importance')
    plt.title('Feature Importance (Sorted)')
    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Load and prepare data
    print("\nLoading and Preparing Data...")
    df = load_and_validate_data("credit_scoring.csv")
    print("\nData Preprocessing Completed.\n")
    print("\nData Cleaning...")
    df.drop_duplicates(inplace=True)
    print("\nData Cleaning Completed.\n")
    
    # Feature engineering
    print("\nFeature Engineering...")
    df = feature_engineering(df)
    print("\nFeature Engineering Completed.\n")
    
    # Split data
    print("\nSplitting Data...")
    x = df.drop(['Credit_Risk', 'Risk_Score'], axis=1)
    y = df['Credit_Risk']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    print("\nData Split Completed.\n")
    
    # Create and fit pipeline
    print("\nCreating and Fitting Pipeline...")
    preprocessor = create_preprocessing_pipeline()
    print("\nPipeline Creation Completed.\n")
    print("\nFitting Pipeline...")
    pipeline = create_classification_pipeline(preprocessor)
    print("\nPipeline Fitting Completed.\n")
    
    # Evaluate model
    print("\nEvaluating Model...")
    evaluate_model(pipeline, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    print("\nCredit Risk Assessment Using Machine Learning\n")
    main()