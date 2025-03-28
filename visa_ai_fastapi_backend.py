# visa_ai_fastapi_backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from lime.lime_tabular import LimeTabularExplainer

app = FastAPI()

# Load and filter COMPAS dataset from local path
compas = CompasDataset(
    data_dir="data/",
    filename="compas-scores-two-years.csv"
)

# Align lengths manually (in case of mismatches)
features_df = pd.DataFrame(compas.features, columns=compas.feature_names)
labels = compas.labels.ravel()[:len(features_df)]
protected = compas.protected_attributes.ravel()[:len(features_df)]

features_df['label'] = labels
features_df['race'] = protected

# Keep only selected features for the prototype
filtered_df = features_df[['age', 'priors_count', 'race', 'label']]
X = filtered_df[['age', 'priors_count', 'race']].values
y = filtered_df['label'].values

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

# Bias metrics setup using AIF360 (recreate AIF360 dataset from filtered data)
from aif360.datasets import BinaryLabelDataset

reconstructed_dataset = BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=['age', 'priors_count', 'race', 'label']),
    label_names=['label'],
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
