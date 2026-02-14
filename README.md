# 🐧 Enhanced Penguin Species Classification API with FastAPI

An enhanced MLOps project that trains a Random Forest classifier on the Palmer Penguins dataset and serves predictions through a FastAPI REST API with advanced features.

## 🎯 Project Overview

This project demonstrates machine learning model deployment using FastAPI, featuring:
- **Palmer Penguins Dataset** - Real Antarctic penguin research data
- **Random Forest Classifier** for species prediction
- **Model evaluation metrics** (accuracy, precision, recall, F1-score)
- **Batch prediction** capability
- **Health check** endpoint
- **Model information** endpoint
- **Enhanced error handling** and logging
- **Input validation** with Pydantic

## 🐧 About the Dataset

The **Palmer Penguins dataset** is a modern alternative to the classic Iris dataset, containing real data collected from three penguin species in the Palmer Archipelago, Antarctica:

- **Adelie Penguin** - Smallest of the three, found throughout the Antarctic coast
- **Chinstrap Penguin** - Named for the thin black band under their head
- **Gentoo Penguin** - Largest species, with a bright orange-red bill

**Features (4 measurements):**
1. Bill length (mm)
2. Bill depth (mm)
3. Flipper length (mm)
4. Body mass (g)

**Dataset size:** ~340 samples (after cleaning)

## 🔧 Modifications from Original Lab

This implementation includes several enhancements over the base lab:

1. **Different Dataset**: Uses **Palmer Penguins** instead of Iris (less commonly used, more interesting!)
2. **Model Upgrade**: Uses Random Forest instead of Decision Tree for better performance
3. **Comprehensive Metrics**: Tracks and exposes accuracy, precision, recall, and F1-score
4. **Batch Predictions**: Added endpoint to handle multiple predictions in one request
5. **Health Monitoring**: Health check endpoint for service monitoring
6. **Model Info API**: Endpoint to retrieve model performance metrics
7. **Enhanced Validation**: Stricter input validation with range checks
8. **Logging**: Comprehensive logging for debugging and monitoring
9. **Error Handling**: Robust error handling with custom exception handlers
10. **Real Research Data**: Uses actual scientific data from penguin studies

## 📁 Project Structure

```
fastapi_penguin_classifier/
├── src/
│   ├── __init__.py
│   ├── train.py              # Enhanced model training script
│   └── main.py               # FastAPI application
├── model/
│   ├── penguin_rf_model.pkl  # Trained Random Forest model
│   └── model_metrics.json    # Model performance metrics
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/CGunal7/Penguin_FastAPI.git
cd Penguin_FastAPI
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
Loading Palmer Penguins dataset...

Dataset info:
- Original samples: 344
- Species: ['Adelie' 'Chinstrap' 'Gentoo']
- Clean samples (after removing NaN): 333
- Number of features: 4
- Features: ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

Training set size: 266
Test set size: 67

Training Random Forest Classifier...

Model Performance:
Accuracy: 0.9851
Precision: 0.9855
Recall: 0.9851
F1 Score: 0.9851

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
  "message": "🐧 Enhanced Penguin Species Classification API",
  "version": "2.0.0",
  "dataset": "Palmer Penguins Dataset",
  "description": "Classify Antarctic penguins (Adelie, Chinstrap, Gentoo) based on physical measurements",
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

Predict penguin species for a single penguin.

**Request Body:**
```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0
}
```

**Response:**
```json
{
  "species": "Adelie",
  "species_id": 0,
  "confidence": 0.98,
  "timestamp": "2025-02-12T10:30:00"
}
```

### 4. Batch Prediction
**POST** `/predict/batch`

Predict penguin species for multiple penguins at once.

**Request Body:**
```json
{
  "samples": [
    {
      "bill_length_mm": 39.1,
      "bill_depth_mm": 18.7,
      "flipper_length_mm": 181.0,
      "body_mass_g": 3750.0
    },
    {
      "bill_length_mm": 46.5,
      "bill_depth_mm": 17.9,
      "flipper_length_mm": 192.0,
      "body_mass_g": 3500.0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "species": "Adelie",
      "species_id": 0,
      "confidence": 0.98,
      "timestamp": "2025-02-12T10:30:00"
    },
    {
      "species": "Chinstrap",
      "species_id": 1,
      "confidence": 0.95,
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
  "dataset": "Palmer Penguins",
  "accuracy": 0.9851,
  "precision": 0.9855,
  "recall": 0.9851,
  "f1_score": 0.9851,
  "training_date": "2025-02-12T10:00:00",
  "train_size": 266,
  "test_size": 67,
  "n_features": 4,
  "feature_names": ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
  "n_classes": 3
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

### Example Test Data

**Adelie Penguin (Species 0):**
```json
{
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181.0,
  "body_mass_g": 3750.0
}
```

**Chinstrap Penguin (Species 1):**
```json
{
  "bill_length_mm": 46.5,
  "bill_depth_mm": 17.9,
  "flipper_length_mm": 192.0,
  "body_mass_g": 3500.0
}
```

**Gentoo Penguin (Species 2):**
```json
{
  "bill_length_mm": 47.5,
  "bill_depth_mm": 14.5,
  "flipper_length_mm": 215.0,
  "body_mass_g": 5200.0
}
```

### Using cURL

```bash
# Health check
curl http://127.0.0.1:8000/health

# Single prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181.0, "body_mass_g": 3750.0}'

# Batch prediction
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181.0, "body_mass_g": 3750.0}]}'
```

### Using Python Requests

```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0
    }
)
print(response.json())
```

## 📊 Model Performance

The Random Forest classifier achieves excellent performance on the Palmer Penguins dataset:

- **Accuracy**: ~98.5%
- **Precision**: ~98.6%
- **Recall**: ~98.5%
- **F1 Score**: ~98.5%

The high performance demonstrates that penguin species can be accurately classified based on physical measurements.

## 🛠️ Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for serving FastAPI applications
- **Scikit-learn**: Machine learning library for model training
- **Pydantic**: Data validation using Python type annotations
- **NumPy**: Numerical computing library
- **Seaborn**: Data visualization and dataset loading
- **Pandas**: Data manipulation and analysis

## 📝 Key Features

### Input Validation
- Automatic validation of input data types
- Range checks for measurements
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
- Clear endpoint descriptions with penguin species info

## 🔍 Troubleshooting

### Model Not Found Error
If you see "Model not loaded" error:
1. Ensure you've run `python train.py` first
2. Check that `model/penguin_rf_model.pkl` exists
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
- [Palmer Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## 👨‍💻 Author

**Gunal chandra sekar**
- GitHub: [@CGunal7](https://github.com/CGunal7)
- Course: MLOps - Northeastern University
- Assignment: Lab Assignment 2


**Why Penguins? 🐧** The Palmer Penguins dataset is a modern, real-world dataset that's perfect for demonstrating ML classification. It's more interesting than Iris and less commonly used, making it a great choice for standing out in your assignment!
