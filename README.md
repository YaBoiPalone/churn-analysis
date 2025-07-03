Churn Rate Analysis in SaaS Companies

This project analyzes customer churn in SaaS companies using the real [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It includes data cleaning, exploratory data analysis (EDA), statistical insights, predictive modeling, and an interactive Streamlit dashboard.

Features
- **Data Loading**: Uses the real Kaggle Telco Customer Churn dataset (CSV file).
- **Data Cleaning**: Handles missing values, encodes categorical variables, and prepares data for analysis.
- **EDA**: Visualizes churn distribution, correlations, and feature relationships.
- **Statistical Insights**: Correlation and chi-square tests to identify key churn drivers.
- **Predictive Modeling**: Logistic Regression model to predict churn, with evaluation metrics.
- **Streamlit Dashboard**: Interactive dashboard for data exploration, filtering, and churn prediction.
- **Business Insights**: Key takeaways and actionable recommendations.

Visualizations & What They Show
- **Churn Rate Distribution**: Bar chart showing the number of churned vs. non-churned customers. Reveals overall churn rate and class balance.
- **Correlation Heatmap**: Heatmap of correlations between all numeric features and churn. Identifies which features are most related to churn.
- **Churn Rate by Category**: Bar charts for churn rate by contract type and payment method. Shows which groups are at higher risk of churn.
- **Churn Rate by Numeric Feature**: Boxplots for Monthly Charges and Tenure, split by churn status. Shows how these numeric features differ for churned vs. non-churned customers.
- **Confusion Matrix**: Visualizes model prediction results (true/false positives/negatives).
- **ROC Curve**: Shows the model's ability to distinguish between churners and non-churners.
- **Customer Table**: Interactive table for exploring the raw data.
- **Churn Prediction Form**: Lets users input new customer data and predicts churn probability using the trained model.

Technologies Used
- Python, Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

Setup Instructions
1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in your project directory.
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app**:
   ```
   streamlit run churn_analysis.py
   ```

Usage
- The dashboard will open in your browser.
- Explore data, visualize churn, and use the prediction form for new customers.

Business Insights
- Customers with month-to-month contracts have a churn rate nearly 3x higher than those with longer contracts.
- Higher monthly charges and lower tenure are strongly associated with increased churn.
- Customers without tech support are more likely to churn.
- Electronic check users show higher churn rates compared to other payment methods.
