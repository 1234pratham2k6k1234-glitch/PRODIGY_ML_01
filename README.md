# PRODIGY_ML_01
##  Overview
This project is a Machine Learning task focused on predicting **house prices** using a **Linear Regression** model.  
The model estimates the selling price of a house based on three important features:

**Square Footage (Living Area)**  
**Number of Bedrooms**  
**Number of Bathrooms**

This project demonstrates the complete workflow of a basic supervised ML regression problem, including:
- Dataset loading
- Feature selection
- Data preprocessing
- Model training
- Model evaluation
- Visualization of results
- Making custom predictions

---

##  Objective
To build a regression model that can **predict the sale price of a house** using the given features and evaluate the model performance using standard regression metrics.

---

##  Dataset Information
The dataset used for this project is taken from Kaggle:

**House Prices: Advanced Regression Techniques**  
 https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

We use the `train.csv` file for training and testing the model.

---

## Features Used
The Kaggle dataset contains many features, but for this task we focus on only these 3:

| Feature Name | Meaning |
|------------|---------|
| **GrLivArea** | Above ground living area square footage |
| **BedroomAbvGr** | Total bedrooms above ground |
| **FullBath** | Total full bathrooms |

**Target Column:**  
| Target | Meaning |
|--------|---------|
| **SalePrice** | Final selling price of the house |

---

## Tools & Technologies Used
- **Python**
- **Pandas** (Data handling)
- **NumPy** (Numerical operations)
- **Matplotlib / Seaborn** (Visualization)
- **scikit-learn** (Machine learning model + evaluation)

---

##  Machine Learning Workflow
### 1️⃣ Data Loading
The dataset (`train.csv`) is loaded using Pandas.

### 2️⃣ Data Selection
Only relevant columns are extracted:
- GrLivArea
- BedroomAbvGr
- FullBath
- SalePrice

### 3️⃣ Train/Test Split
The dataset is split into:
- **80% Training data**
- **20% Testing data**

This helps verify model performance on unseen data.

### 4️⃣ Model Training
A **Linear Regression** model is trained using scikit-learn:
- The model learns the relationship between inputs (features) and output (SalePrice).

### 5️⃣ Prediction & Evaluation
After training, the model predicts house prices for the test set.

Evaluation is done using these metrics:

**MAE (Mean Absolute Error)**  
**MSE (Mean Squared Error)**  
**RMSE (Root Mean Squared Error)**  
**R² Score (Goodness of Fit)**

---

## Results & Visualizations
The notebook/script generates:
**Actual vs Predicted Scatter Plot**  
**Residual Distribution Plot** (to analyze prediction errors)

These plots help understand:
- how close the predictions are to actual values
- the distribution of errors in the model

---

## Example Prediction
The model can also predict house prices for custom inputs.

Example Input:
- **2000 sqft**
- **3 bedrooms**
- **2 bathrooms**

The model outputs the estimated selling price based on training data trends.

---

## Project Structure
Task-01-House-Price-Prediction/
│── train.csv
│── house_price_prediction.ipynb
│── README.md

---

## How to Run This Project

### Step 1: Download Dataset
Download `train.csv` from Kaggle:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

### Step 2: Clone This Repository

``bash

git clone <your-repo-link>
cd house-price-prediction

###Step 3: Install Dependencies

bash

pip install pandas numpy scikit-learn matplotlib seaborn

###step 4: Run Notebook

Open the .ipynb file in:
*Jupyter Notebook / Jupyter Lab
*Google Colab
*VS Code Notebook

##Conclusion

This project demonstrates how Linear Regression can be used to solve a real-world regression problem by predicting house prices using basic property features.

It serves as a strong foundation for further improvements such as:
 Adding more features
 Using advanced regression models (Ridge, Lasso, XGBoost)
 Feature scaling and hyperparameter tuning
 Deployment using Streamlit

##Author

Pratham Agarwal
 Email: pratham.vishal.agarwal@gmail.com
 GitHub: https://github.com/RoughRanger2005
