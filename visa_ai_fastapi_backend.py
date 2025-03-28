from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # For encoding categorical features
from aif360.metrics import BinaryLabelDatasetMetric
from lime.lime_tabular import LimeTabularExplainer

app = FastAPI()

# Manually load the COMPAS dataset from the CSV file
compas_df = pd.read_csv("compas-scores-two-years.csv")

# Preprocess the data
# Select the columns you need (adjust based on your dataset's column names)
compas_df = compas_df[['age', 'priors_count', 'race', 'two_year_recid']]  # Adjust column names accordingly

# Handle non-numeric 'race' column by encoding it
label_encoder = LabelEncoder()
compas_df['race'] = label_encoder.fit_transform(compas_df['race'])  # Encoding 'race' to numeric values

# Handle any other non-numeric columns similarly, if applicable.

# Define features and labels
X = compas_df[['age', 'priors_count', 'race']].values
y = compas_df['two_year_recid'].values  # Adjust the target column name accordingly

# Train a simple logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Setup LIME
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=['age', 'priors_count', 'race'],
    class_names=['No Recidivism', 'Recidivism'],
    mode='classification'
)

# Bias metrics setup using AIF360
from aif360.datasets import BinaryLabelDataset

# Recreate AIF360 dataset from the DataFrame
reconstructed_dataset = BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=compas_df,
    label_names=['two_year_recid'],  # Adjust column names accordingly
    protected_attribute_names=['race']
)

privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]
metric = BinaryLabelDatasetMetric(reconstructed_dataset, unprivileged_groups, privileged_groups)

class VisaInput(BaseModel):
    age: float
    priors_count: float
    race: int

@app.post("/predict")
def predict_visa(data: VisaInput):
    input_array = np.array([[data.age, data.priors_count, data.race]])
    prediction = model.predict(input_array)[0]
    decision = "Accepted" if prediction == 0 else "Rejected"
    return {"visa_decision": decision}

@app.post("/explain")
def explain_decision(data: VisaInput):
    input_array = np.array([data.age, data.priors_count, data.race])
    explanation = explainer.explain_instance(input_array, model.predict_proba)
    exp_list = explanation.as_list()
    return {"explanation": exp_list}

@app.get("/audit")
def audit_bias():
    di = metric.disparate_impact()
    md = metric.mean_difference()
    return {
        "disparate_impact": di,
        "mean_difference": md
    }
