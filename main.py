from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

app = FastAPI()

data = None
model = None

# request body models
class TrainRequest(BaseModel):
    target_column: str

class PredictRequest(BaseModel):
    Temperature: float
    Run_Time: float

# endpoint to upload the dataset
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global data
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    data = pd.read_csv(file.file)
    return {"message": "Dataset uploaded successfully", "columns": list(data.columns)}

# endpoint to train the model
@app.post("/train")
def train(request: TrainRequest):
    global model, data
    if data is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    # check if target column exists
    if request.target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")
    
    # prepare data
    X = data.drop(columns=[request.target_column, "Machine_ID"])
    y = data[request.target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted")
    }
    
    return {"message": "Model trained successfully", "metrics": metrics}

# endpoint to predict
@app.post("/predict")
def predict(request: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="No model trained")
    
    input_data = pd.DataFrame([request.model_dump()])
    
    # get prediction and confidence score
    # took from here https://stackoverflow.com/questions/31129592/how-to-get-a-classifiers-confidence-score-for-a-prediction-in-sklearn
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]  # returns probabilities for each class
    confidence = max(probabilities)  # confidence for the predicted class
    
    return {
        "prediction": int(prediction),
        "confidence": float(confidence)
    }