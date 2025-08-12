
                                    GlucoVision: AI-Powered Diabetes Prediction and Monitoring App



## Project Overview

GlucoVision is an interactive web application built with Streamlit that leverages machine learning to predict diabetes risk based on user-provided health metrics. The app uses a trained model (either Logistic Regression or Random Forest) on the Pima Indians Diabetes Dataset to provide predictions, data visualizations, and model performance insights. It includes features for data exploration, interactive charts, model dashboards, and real-time predictions.

Key Features:
- **Data Explorer**: View dataset overview and statistical summaries.
- **Visualizations**: Interactive histograms, box plots, and correlation heatmaps.
- **Model Dashboard**: Performance metrics, class distribution, and feature importance.
- **Prediction**: Input health metrics to get diabetes risk predictions with probability.

The backend handles data preprocessing, model training/selection, and evaluation. The app is designed for ease of use, making it accessible for users to monitor their health proactively.

## Tech Stack

- **Programming Language**: Python 3.12+
- **Libraries**:
  - Streamlit (for the web app)
  - Pandas & NumPy (data manipulation)
  - Scikit-learn (machine learning pipelines, models, and metrics)
  - Plotly (interactive visualizations)
  - Joblib (model persistence)
- **Dataset**: Pima Indians Diabetes Dataset (`data/dataset.csv`)
- **Models**: Logistic Regression or Random Forest (selected via cross-validation)

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/glucovision.git
   cd glucovision
   ```

2. **Set Up Virtual Environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install manually:
   ```
   pip install streamlit pandas numpy scikit-learn plotly joblib
   ```

4. **Download Dataset**:
   - Place the Pima Indians Diabetes Dataset as `data/dataset.csv` in the project root. You can download it from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Usage

1. **Run the Streamlit App**:
   ```
   streamlit run app.py
   ```
   - Open your browser at `http://localhost:8501`.

2. **Navigation**:
   - Use the sidebar to switch between pages: Data Explorer, Visualizations, Model Dashboard, and Predict.
   - Upload a custom CSV dataset via the sidebar if desired (must match the structure of the default dataset).

3. **Training the Model** (Optional):
   - Run `python model_training.py` to train and save the model (`model.pkl`). This script handles data loading, preprocessing, model comparison, and evaluation.

## Project Structure

```
glucovision/
├── app.py                  # Main Streamlit application
├── notebook
   |_model_training.py       # Script for model training and evaluation
├── data/
│   └── dataset.csv         # Pima Indians Diabetes Dataset
├── model.pkl               # Trained model (generated after training)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Dataset

- **Source**: Pima Indians Diabetes Dataset from UCI Machine Learning Repository (Kaggle).
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- **Target**: Outcome (0: No Diabetes, 1: Diabetes).
- **Size**: 768 samples.

## Model Training

- **Preprocessing**: Zeros in key columns (Glucose, BloodPressure, etc.) treated as missing values and imputed with median.
- **Models Compared**: Logistic Regression and Random Forest via 5-fold stratified cross-validation on ROC-AUC.
- **Selection**: Best model saved as `model.pkl`.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Classification Report.



## Screenshots

1. **Home Page**:
   
2. **Data Explorer**:
   

3. **Visualizations**:
   

4. **Model Dashboard**:
  

5. **Prediction Page**:


## License

MIT License. See [LICENSE](LICENSE) for details.


                                                               Developed by Sandun Indrasiri Wijesingha  
                                                                     Email: ssandu809@gmail.com  
                                                                     © All Rights Reserved
