# Enhanced Iris Classification API with FastAPI

An enhanced MLOps project that trains a Random Forest classifier on the Iris dataset and serves predictions through a FastAPI REST API with advanced features.

## 🎯 Project Overview

This project demonstrates machine learning model deployment using FastAPI, featuring:
- **Random Forest Classifier** (instead of Decision Tree)
- **Model evaluation metrics** (accuracy, precision, recall, F1-score)
- **Batch prediction** capability
- **Health check** endpoint
- **Model information** endpoint
- **Enhanced error handling** and logging
- **Input validation** with Pydantic

## 🔧 Modifications from Original Lab

This implementation includes several enhancements over the base lab:

1. **Model Upgrade**: Uses Random Forest instead of Decision Tree for better performance
2. **Comprehensive Metrics**: Tracks and exposes accuracy, precision, recall, and F1-score
3. **Batch Predictions**: Added endpoint to handle multiple predictions in one request
4. **Health Monitoring**: Health check endpoint for service monitoring
5. **Model Info API**: Endpoint to retrieve model performance metrics
6. **Enhanced Validation**: Stricter input validation with range checks
7. **Logging**: Comprehensive logging for debugging and monitoring
8. **Error Handling**: Robust error handling with custom exception handlers

## 📁 Project Structure

```
fastapi_iris_classifier/
├── src/
│   ├── __init__.py
│   ├── train.py          # Enhanced model training script
│   └── main.py           # FastAPI application
├── model/
│   ├── iris_rf_model.pkl    # Trained Random Forest model
│   └── model_metrics.json   # Model performance metrics
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── .gitignore          # Git ignore file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd fastapi_iris_classifier
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
cd src
python train.py
```

Expected output:
```
Loading Iris dataset...
Training set size: 120
Test set size: 30

Training Random Forest Classifier...

Evaluating model...

Model Performance:
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1 Score: 1.0000

✓ Model training completed successfully!
```

### Step 5: Start the FastAPI Server

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API Documentation**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc
- **API Base**: http://127.0.0.1:8000

## 📡 API Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

```json
{
  "message": "Enhanced Iris Classification API",
  "version": "2.0.0",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict": "/predict",
    "predict_batch": "/predict/batch",
    "model_info": "/model/info"
  }
}
```

### 2. Health Check
**GET** `/health`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "timestamp": "2025-02-12T10:30:00"
}
```

### 3. Single Prediction
**POST** `/predict`

Predict iris species for a single flower.

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "species": "setosa",
  "species_id": 0,
  "confidence": 1.0,
  "timestamp": "2025-02-12T10:30:00"
}
```

### 4. Batch Prediction
**POST** `/predict/batch`

Predict iris species for multiple flowers at once.

**Request Body:**
```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 6.7,
      "sepal_width": 3.0,
      "petal_length": 5.2,
      "petal_width": 2.3
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "species": "setosa",
      "species_id": 0,
      "confidence": 1.0,
      "timestamp": "2025-02-12T10:30:00"
    },
    {
      "species": "virginica",
      "species_id": 2,
      "confidence": 0.98,
      "timestamp": "2025-02-12T10:30:00"
    }
  ],
  "total_samples": 2,
  "timestamp": "2025-02-12T10:30:00"
}
```

### 5. Model Information
**GET** `/model/info`

Get model performance metrics.

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "training_date": "2025-02-12T10:00:00",
  "train_size": 120,
  "test_size": 30
}
```

## 🧪 Testing the API

### Using the Interactive Documentation

1. Navigate to http://127.0.0.1:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in the request body
5. Click "Execute"
6. View the response

### Using cURL

```bash
# Health check
curl http://127.0.0.1:8000/health

# Single prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Batch prediction
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'
```

### Using Python Requests

```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(response.json())
```

## 📊 Model Performance

The Random Forest classifier achieves excellent performance on the Iris dataset:

- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1 Score**: 100%

*Note: These metrics are from the test set. The high performance is expected given the simplicity of the Iris dataset.*

## 🛠️ Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for serving FastAPI applications
- **Scikit-learn**: Machine learning library for model training
- **Pydantic**: Data validation using Python type annotations
- **NumPy**: Numerical computing library

## 📝 Key Features

### Input Validation
- Automatic validation of input data types
- Range checks for measurements (0-10 cm)
- Clear error messages for invalid inputs

### Error Handling
- Graceful error handling for all endpoints
- Detailed error messages
- Appropriate HTTP status codes

### Logging
- Comprehensive logging for debugging
- Request/response logging
- Error tracking

### API Documentation
- Auto-generated OpenAPI documentation
- Interactive API testing interface
- Clear endpoint descriptions

## 🔍 Troubleshooting

### Model Not Found Error
If you see "Model not loaded" error:
1. Ensure you've run `python train.py` first
2. Check that `model/iris_rf_model.pkl` exists
3. Verify you're running the server from the correct directory

### Port Already in Use
If port 8000 is busy, use a different port:
```bash
uvicorn main:app --reload --port 8001
```

### Import Errors
Make sure your virtual environment is activated and all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## 👨‍💻 Author

**Your Name**
- GitHub: [@CGunal7](https://github.com/yourusername)
- Course: MLOps - Northeastern University
- Assignment: Lab Assignment 2

## 📄 License

This project is created for educational purposes as part of the MLOps course at Northeastern University.

## 🙏 Acknowledgments

- Original lab by Prof. Ramin Mohammadi
- Lab Credits: Dhanush Kumar Shankar
- Course: Machine Learning Operations (MLOps)
- Institution: Northeastern University
