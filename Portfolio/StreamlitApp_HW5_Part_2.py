import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json 

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.serializers import JSONSerializer 
from sagemaker.deserializers import JSONDeserializer 
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def convert_input_pca_regression(raw_json_input, content_type='application/json'):
    import json
    import pandas as pd

    if content_type == 'application/json':
        data = json.loads(raw_json_input)

        # if single prediction input comes in as a dict, wrap it in a list
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        return df

    raise ValueError(f"Unsupported content type: {content_type}")

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer_pca.shap",
    "pipeline": "finalized_pca_model.tar.gz",
    "keys": [
        "EMA_5", "ROC_5", "MOM_5", "RSI_5", "MA_5",
        "EMA_10", "ROC_10", "MOM_10", "RSI_10", "MA_10",
        "EMA_15", "ROC_15", "MOM_15", "RSI_15", "MA_15",
        "EMA_20", "ROC_20", "MOM_20", "RSI_20", "MA_20",
        "EMA_30", "ROC_30", "MOM_30", "RSI_30", "MA_30"
    ],
    "inputs": [
        {"name": k, "type": "number", "min": -100.0, "max": 100.0, "default": 0.0, "step": 0.1}
        for k in [
            "EMA_5", "ROC_5", "MOM_5", "RSI_5", "MA_5",
            "EMA_10", "ROC_10", "MOM_10", "RSI_10", "MA_10",
            "EMA_15", "ROC_15", "MOM_15", "RSI_15", "MA_15",
            "EMA_20", "ROC_20", "MOM_20", "RSI_20", "MA_20",
            "EMA_30", "ROC_30", "MOM_30", "RSI_30", "MA_30"
        ]
    ]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename=MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename, 
        Bucket=bucket, 
        Key= f"{key}/{os.path.basename(filename)}")
        # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    local_path = local_path

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# Prediction Logic
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        return str(pred_val), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    raw_json_input = json.dumps(input_df)
    input_df = convert_input_pca_regression(raw_json_input, 'application/json')

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])

    input_df_transformed = preprocessing_pipeline.transform(input_df)
    component_names = [f"KPCA_{i+1}" for i in range(input_df_transformed.shape[1])]
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=component_names)

    shap_values = explainer(input_df_transformed)

    pred_class = best_pipeline.predict(input_df)[0]
    class_index = list(best_pipeline.named_steps['model'].classes_).index(pred_class)

    st.subheader("🔎 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, class_index], max_display=10)
    st.pyplot(fig)

    top_feature = pd.Series(
        shap_values[0, :, class_index].values,
        index=shap_values[0, :, class_index].feature_names
    ).abs().idxmax()

    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    
    res, status = call_model_api(user_inputs)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(user_inputs,session, aws_bucket)
    else:
        st.error(res)



