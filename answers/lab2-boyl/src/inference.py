import joblib
import os
from sagemaker_inference import encoder, decoder


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, 'model.joblib'))


def input_fn(request_body, content_type):
    return decoder.decode(request_body, content_type)


def predict_fn(input_data, model):
    return model.predict_proba(input_data)[:, 1]


def output_fn(prediction, accept):
    return encoder.encode(prediction, accept)
