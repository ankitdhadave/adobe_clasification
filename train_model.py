
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ==============================
# Load data
# ==============================
df = pd.read_excel("Adobe Dataset Final.xlsx")

# ==============================
# Drop ID + leakage columns
# ==============================
drop_cols = [
    'Customer_ID',
    'Churn_Status',
    'Lifetime_Value',
    'Plan_Upgrade_Flag',
    'Plan_Downgrade_Flag'
]

df = df.drop(columns=drop_cols, errors='ignore')

# ==============================
# Target & features
# ==============================
X = df.drop(columns=['Conversion_Status'])
y = df['Conversion_Status']

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Column groups (FROM NOTEBOOK)
# ==============================
ordinal_cols = [
    'Company_Size',
    'Subscription_Plan'
]

categorical_cols = [
    'Region',
    'Industry',
    'Device_Type',
    'Marketing_Channel'
]

numeric_cols = [
    'Avg_Session_Time_mins',
    'Cloud_Storage_Usage_GB',
    'Files_Created',
    'Website_Visits',
    'Satisfaction_Score',
    'Last_Login_Days_Ago',
    'Login_Frequency_per_Week',
    'Email_Click_Through_Rate',
    'Days_Active_in_Trial',
    'Discount_Offered_USD',
    'Subscription_Fee_USD',
    'Support_Tickets_Raised',
    'Competitor_Subscription_Price',
    'Engagement_Score'
]

# ==============================
# Preprocessing Pipeline
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(), ordinal_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='drop'
)

# ==============================
# Model
# ==============================
scale_pos_weight = 1499 / 501  # imbalance ratio

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

# ==============================
# Full Pipeline
# ==============================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

# ==============================
# Save model
# ==============================
joblib.dump(pipeline, "conversion_model.pkl")

print("✅ Model trained & saved as conversion_model.pkl")