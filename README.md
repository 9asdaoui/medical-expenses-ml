# 🏥 Medical Expenses Prediction - Machine Learning

## 📋 Project Description

This project uses Machine Learning techniques to predict medical insurance costs based on personal characteristics of insured individuals. The system compares multiple regression algorithms to identify the best performing model and provides an interactive interface via Streamlit for real-time cost estimates.

### Objectives
- Analyze factors influencing health insurance costs
- Build and compare multiple Machine Learning models
- Optimize hyperparameters to improve performance
- Deploy an interactive web application for predictions

### Implemented Models
- **Linear Regression**: Basic linear model
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Optimized gradient boosting
- **SVR (Support Vector Regression)**: Support vector machine for regression

### Key Features
✅ Automatic data preprocessing  
✅ Training of multiple ML models  
✅ Hyperparameter optimization (GridSearchCV)  
✅ Performance evaluation and comparison  
✅ Interactive web application with Streamlit  
✅ Detailed reports and visualizations  

---

## 📁 File Structure

```
medical-expenses-ml/
│
├── README.md                      # Project documentation
│
├── app/
│   └── main.py                    # Streamlit application for predictions
│
├── data/
│   ├── raw/
│   │   └── assurance-maladie.csv  # Original raw data
│   └── processed/
│       └── cleaned_data.csv       # Cleaned and preprocessed data
│
├── models/                        # Saved trained models
│   ├── best_RandomForest.pkl      # Optimized Random Forest
│   ├── best_XGBoost.pkl          # Optimized XGBoost
│   ├── best_SVR.pkl              # Optimized SVR
│   ├── RandomForest_Final.pkl    # Final Random Forest model
│   ├── XGBoost_Final.pkl         # Final XGBoost model
│   ├── SVR_Final.pkl             # Final SVR model
│   └── linear_regression.pkl     # Linear Regression
│
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb    # Data preprocessing
│   └── 03_evaluat_compar.ipynb   # Model evaluation and comparison
│
├── outputs/
│   └── figures/                   # Charts and visualizations
│
├── reports/                       # Results reports
│   ├── model_trainig_results.csv # Initial training results
│   ├── tuning_results.csv        # Results after optimization
│   └── final_train_results.csv   # Final results
│
└── src/                           # Python source code
    ├── __init__.py
    ├── processe.py               # Data loading and splitting
    ├── train.py                  # Basic model training
    ├── tune.py                   # Hyperparameter optimization
    └── retrain_rf.py             # Specific Random Forest retraining
```

---

## 🚀 Execution Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/9asdaoui/medical-expenses-ml.git
cd medical-expenses-ml
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost joblib streamlit matplotlib seaborn jupyter
```

---

### 📊 Running Training Scripts

#### 1. Data Preprocessing
The data is already preprocessed in `data/processed/cleaned_data.csv`. If you want to explore the process:
```bash
jupyter notebook notebooks/02_Preprocessing.ipynb
```

#### 2. Initial Model Training
```bash
python src/train.py
```
This script:
- Loads preprocessed data
- Trains 4 models (Linear Regression, Random Forest, XGBoost, SVR)
- Saves models to `models/`
- Generates a report in `reports/model_trainig_results.csv`

#### 3. Hyperparameter Optimization
```bash
python src/tune.py
```
This script:
- Performs GridSearchCV for Random Forest, XGBoost, and SVR
- Finds the best hyperparameters
- Saves optimized models (`best_*.pkl`)
- Generates a detailed report in `reports/tuning_results.csv`

#### 4. Specific Retraining (optional)
```bash
python src/retrain_rf.py
```

---

### 🌐 Launching the Web Application

To use the interactive prediction interface:

```bash
streamlit run app/main.py
```

The application will automatically open in your default browser (usually at `http://localhost:8501`).

#### Using the Application
1. Enter patient information:
   - **Age**: Person's age
   - **BMI**: Body Mass Index
   - **Children**: Number of dependent children
   - **Smoker**: Smoking status (Yes/No)
   - **Sex**: Gender (male/female)
   - **Region**: Geographic region
2. Click "Estimate Charges"
3. Get the estimated medical expenses

---

### 📈 Exploratory Analysis

To explore the data and view analyses:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

To see model evaluation and comparison:

```bash
jupyter notebook notebooks/03_evaluat_compar.ipynb
```

---

## 📊 Results

Model performance results are available in the `reports/` folder:
- **model_trainig_results.csv**: Initial performance (MSE, R²)
- **tuning_results.csv**: Performance after optimization
- **final_train_results.csv**: Final results

Visualizations are available in `outputs/figures/`.

---

## 🔧 Input Variables

| Variable  | Type        | Description                                    |
|-----------|-------------|------------------------------------------------|
| age       | Numeric     | Age of the insured                            |
| bmi       | Numeric     | Body Mass Index                               |
| children  | Numeric     | Number of children/dependents                 |
| sex       | Categorical | Gender (male/female)                          |
| smoker    | Categorical | Smoking status (Yes/No)                       |
| region    | Categorical | Geographic region (northeast, northwest, etc.) |

**Target Variable**: `charges` - Medical expenses in dollars

---

## 📝 Technical Notes

- Target values are transformed with `log1p` to normalize the distribution
- Preprocessing includes:
  - **MinMaxScaler** for numeric features
  - **OneHotEncoder** for categorical features
- Train/test split: 80/20
- Random state: 474 (for reproducibility)
- The final deployed model is **XGBoost** (best performance)

---

## 👨‍💻 Author

**9asdaoui**
- GitHub: [@9asdaoui](https://github.com/9asdaoui)

---

## 📄 License

This project is for educational and demonstration purposes.

---

## 🤝 Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.

---

**Happy Learning! 🎓**
