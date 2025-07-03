import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from scipy.stats import chi2_contingency

# 1. DATA LOADING
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

# 2. DATA CLEANING
def clean_data(df):
    df = df.copy()
    # Standardize column names for consistency
    df.rename(columns={
        'customerID': 'CustomerID',
        'gender': 'Gender',
        'tenure': 'Tenure',
        'MonthlyCharges': 'MonthlyCharges',
        'TotalCharges': 'TotalCharges',
        'Contract': 'Contract',
        'PaymentMethod': 'PaymentMethod',
        'InternetService': 'InternetService',
        'TechSupport': 'TechSupport',
        'Churn': 'Churn',
    }, inplace=True)
    # Convert TotalCharges to numeric (handle spaces and missing values)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # Fill missing TechSupport if any
    if df['TechSupport'].isnull().any():
        df['TechSupport'] = df['TechSupport'].fillna(df['TechSupport'].mode()[0])
    # Convert 'Churn' to binary
    df['ChurnBinary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    # Categorical columns as per Kaggle dataset
    cat_cols = ['Gender', 'Contract', 'PaymentMethod', 'InternetService', 'TechSupport']
    # Ensure all categorical columns exist
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df, df_encoded

# 3. EXPLORATORY DATA ANALYSIS (EDA)
def plot_churn_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    ax.set_title('Churn Rate Distribution')
    st.pyplot(fig)

def plot_correlation_heatmap(df_encoded):
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

def plot_churn_by_category(df, col):
    if col not in df.columns or df[col].dropna().empty:
        st.warning(f'Column {col} is missing or empty.')
        return
    fig, ax = plt.subplots()
    try:
        churn_rate = df.groupby(col)['ChurnBinary'].mean().sort_values(ascending=False)
        sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax)
        ax.set_ylabel('Churn Rate')
        ax.set_title(f'Churn Rate by {col}')
        st.pyplot(fig)
    except Exception as e:
        st.error(f'Error plotting churn by {col}: {e}')

def plot_churn_by_numeric(df, col):
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        st.warning(f'Column {col} is missing or not numeric.')
        return
    fig, ax = plt.subplots()
    try:
        sns.boxplot(x='Churn', y=col, data=df, ax=ax)
        ax.set_title(f'{col} by Churn')
        st.pyplot(fig)
    except Exception as e:
        st.error(f'Error plotting {col} by Churn: {e}')

# 4. STATISTICAL INSIGHTS
def chi_square_test(df, col):
    if col not in df.columns:
        return np.nan, np.nan
    table = pd.crosstab(df[col], df['Churn'])
    chi2, p, _, _ = chi2_contingency(table)
    return chi2, p

def correlation_with_churn(df_encoded):
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    if 'ChurnBinary' not in numeric_df.columns:
        raise ValueError('ChurnBinary column missing from numeric data.')
    corr = numeric_df.corr()['ChurnBinary'].sort_values(ascending=False)
    return corr

# 5. PREDICTIVE MODELING
def train_model(df_encoded):
    feature_cols = [col for col in df_encoded.columns if col not in ['CustomerID', 'Churn', 'ChurnBinary'] and pd.api.types.is_numeric_dtype(df_encoded[col])]
    X = df_encoded[feature_cols]
    y = df_encoded['ChurnBinary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, roc_auc, cm, X_test, y_test, y_proba, feature_cols

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC Curve')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)

# 6. STREAMLIT DASHBOARD
def main():
    st.title('Churn Rate Analysis in SaaS Companies')
    st.markdown('---')
    df = load_data()
    df, df_encoded = clean_data(df)

    st.header('Data Overview')
    st.write(df.head())

    st.header('Churn Rate Distribution')
    plot_churn_distribution(df)

    st.header('Correlation Heatmap')
    plot_correlation_heatmap(df_encoded)

    st.header('Churn Rate by Category')
    col1, col2 = st.columns(2)
    with col1:
        plot_churn_by_category(df, 'Contract')
        plot_churn_by_category(df, 'PaymentMethod')
    with col2:
        plot_churn_by_numeric(df, 'MonthlyCharges')
        plot_churn_by_numeric(df, 'Tenure')

    st.header('Statistical Insights')
    st.markdown('**Correlation with Churn:**')
    try:
        corr = correlation_with_churn(df_encoded)
        st.write(corr.head(10))
    except Exception as e:
        st.error(f'Error computing correlation with churn: {e}')
    st.markdown('**Chi-square Test Results:**')
    for col in ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']:
        try:
            chi2, p = chi_square_test(df, col)
            st.write(f'{col}: chi2={chi2:.2f}, p-value={p:.4f}')
        except Exception as e:
            st.error(f'Error in chi-square test for {col}: {e}')

    st.header('Predictive Modeling')
    try:
        model, acc, roc_auc, cm, X_test, y_test, y_proba, feature_cols = train_model(df_encoded)
        st.write(f'**Accuracy:** {acc:.2f}')
        st.write(f'**ROC-AUC:** {roc_auc:.2f}')
        plot_confusion_matrix(cm)
        plot_roc_curve(y_test, y_proba)
    except Exception as e:
        st.error(f'Error in predictive modeling: {e}')
        return

    st.header('Customer Table')
    st.dataframe(df)

    st.header('Churn Prediction')
    with st.expander('Predict Churn for New Customer'):
        gender = st.selectbox('Gender', ['Male', 'Female'])
        tenure = st.slider('Tenure (months)', 0, 72, 12)
        monthly = st.slider('Monthly Charges', float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()), float(df['MonthlyCharges'].median()))
        contract = st.selectbox('Contract', sorted(df['Contract'].dropna().unique()))
        payment = st.selectbox('Payment Method', sorted(df['PaymentMethod'].dropna().unique()))
        internet = st.selectbox('Internet Service', sorted(df['InternetService'].dropna().unique()))
        tech = st.selectbox('Tech Support', sorted(df['TechSupport'].dropna().unique()))
        total = monthly * tenure
        # Prepare input for model
        input_dict = {
            'Tenure': tenure,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
        }
        # Dynamically set one-hot encoded columns
        for col in feature_cols:
            if col.startswith('Gender_'):
                input_dict[col] = 1 if col == f'Gender_{gender}' else 0
            elif col.startswith('Contract_'):
                input_dict[col] = 1 if col == f'Contract_{contract}' else 0
            elif col.startswith('PaymentMethod_'):
                input_dict[col] = 1 if col == f'PaymentMethod_{payment}' else 0
            elif col.startswith('InternetService_'):
                input_dict[col] = 1 if col == f'InternetService_{internet}' else 0
            elif col.startswith('TechSupport_'):
                input_dict[col] = 1 if col == f'TechSupport_{tech}' else 0
            elif col not in input_dict:
                input_dict[col] = 0
        input_df = pd.DataFrame([input_dict])[feature_cols]
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0, 1]
            st.write(f'**Predicted Churn:** {"Yes" if pred == 1 else "No"} (Probability: {proba:.2f})')
        except Exception as e:
            st.error(f'Error in churn prediction: {e}')

    st.header('Business Insights')
    st.markdown('''
    - Customers with month-to-month contracts have a churn rate nearly 3x higher than those with longer contracts.
    - Higher monthly charges and lower tenure are strongly associated with increased churn.
    - Customers without tech support are more likely to churn.
    - Electronic check users show higher churn rates compared to other payment methods.
    ''')

if __name__ == '__main__':
    main() 
