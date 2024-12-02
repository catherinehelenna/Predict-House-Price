import sklearn
print(sklearn.__version__)
import joblib
print(joblib.__version__)

# Load the model with a context manager
with open("deployment/capped_lasso_pipeline.joblib", "rb") as file:
    model = joblib.load(file)