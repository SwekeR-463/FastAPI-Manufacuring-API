# FastAPI Manufacturing API

This is a FastAPI-based application to train and predict machine downtime using a Logistic Regression model and a synthetically generated dataset. It supports dataset upload, model training, and prediction endpoints.

---

## Setup Instructions

### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/SwekeR-463/FastAPI-Manufacuring-API.git
cd FastAPI-Manufacturing-API
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Server
Run the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

---

## Endpoints

### 1. **Upload Dataset**
Uploads a CSV dataset to the server.

- **URL**: `/upload`
- **Method**: `POST`
- **Headers**:
  - `Content-Type: multipart/form-data`
- **Body**:
  - `file`: The CSV file containing the dataset.

#### Example Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@synthetic_manufacturing_data.csv"
```

#### Example Response:
```json
{
  "message": "Dataset uploaded successfully",
  "columns": ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]
}
```

---

### 2. **Train the Model**
Trains the Decision Tree model on the uploaded dataset.

- **URL**: `/train`
- **Method**: `POST`
- **Headers**:
  - `Content-Type: application/json`
- **Body**:
  ```json
  {
    "target_column": "Downtime_Flag"
  }
  ```

#### Example Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"target_column": "Downtime_Flag"}'
```

#### Example Response:
```json
{
    "message": "Model trained successfully",
    "metrics": {
        "accuracy": 0.956,
        "f1_score": 0.9559214200203747
    }
}
```

---

### 3. **Make a Prediction**
Predicts whether there will be downtime based on input features.

- **URL**: `/predict`
- **Method**: `POST`
- **Headers**:
  - `Content-Type: application/json`
- **Body**:
  ```json
  {
    "Temperature": 85.69,
    "Run_Time": 80.11
  }
  ```

#### Example Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Temperature": 85.69, "Run_Time": 80.11}'
```

#### Example Response:
```json
{
    "prediction": 0,
    "confidence": 0.9999636998087638
}
```

---
