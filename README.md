#  Titanic Passenger Survival Predictor

##  Project Overview

This project is a simple web application that predicts whether a Titanic passenger would have survived based on selected passenger details. It uses a pre-trained Support Vector Classifier (SVC) machine learning model and an interactive interface built with Streamlit.

The project was completed as part of a learning assignment to understand existing machine learning code on GitHub, download the project files, and deploy the application using Streamlit Community Cloud.

##  Features

- Interactive web interface using Streamlit.
- Predicts passenger survival using a pre-trained SVC model.
- Accepts four passenger features:
  - Passenger class
  - Gender
  - Number of siblings or spouses aboard
  - Port of embarkation
- Converts categorical inputs into numerical values using Label Encoding.
- Displays the predicted outcome and encoded feature values.

##  Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Pickle

##  Project Structure

```text
my-titanic-app/
├── app.py
├── svc_trained_model.pkl
├── requirements.txt
└── README.md
```

### File Descriptions

- **app.py:** Contains the Streamlit application, model-loading logic, label encoders, and prediction functionality.
- **svc_trained_model.pkl:** Contains the pre-trained machine learning model.
- **requirements.txt:** Lists the Python libraries and their versions required to run the application.
- **README.md:** Provides information about the project and instructions for running it.

## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd my-titanic-app
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

##  Live Demo

**Streamlit App:** https://my-titanic-app.streamlit.app/

##  Learning Objectives

- Understand the structure of an existing machine learning project.
- Learn how a pre-trained model is loaded from a Pickle file.
- Understand categorical feature encoding using Scikit-learn's LabelEncoder.
- Learn how user input is converted into model-ready data.
- Deploy a Python machine learning application using Streamlit Community Cloud.

##  Disclaimer

This application is for educational purposes only. It is based on a simplified 100-row Titanic dataset, and its predictions reflect patterns learned from that dataset. The results should not be considered a reliable historical analysis or a guarantee of survival outcomes.
