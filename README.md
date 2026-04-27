# FIFA 23 Player Value Prediction

## Overview

This project focuses on predicting football player market values using machine learning techniques.
The dataset is based on FIFA 23 player attributes, and the workflow includes preprocessing, feature engineering, multicollinearity analysis, and model comparison.

The primary objective is to build a regression model that can accurately estimate a player's market value based on performance, physical, and metadata attributes.

---

## Project Workflow

### 1. Data Preprocessing

* Processed large dataset using **chunk-based loading** for memory efficiency
* Handled missing values:

  * Numerical → Mean imputation
  * Categorical → Mode imputation
* Converted string-based features:

  * Currency (`Value`, `Wage`, `Release Clause`) → numeric
  * Height/Weight → numeric
* Feature encoding:

  * Preferred Foot → Binary (Right = 1, Left = 0)
  * Position → Cleaned HTML tags + mapped to numerical values
  * Work Rate → Encoded into a numerical scale
* Removed irrelevant features:

  * `Photo`, `Flag`, `Club Logo`, `Kit Number`, etc.

---

### 2. Exploratory Data Analysis (EDA)

* Generated **correlation matrix**
* Identified strong relationships between features and target (`Value`)
* Reduced redundant variables before modeling

---

### 3. Multicollinearity Check

* Applied **Variance Inflation Factor (VIF)**
* Ensured no severe multicollinearity issues
* All features maintained acceptable VIF ranges (< 5)

---

### 4. Data Splitting

* Dataset split into:

  * **80% Training**
  * **20% Testing**

```python
train_test_split(test_size=0.2, random_state=42)
```

---

### 5. Model Training & Evaluation

Multiple models were tested:

* Random Forest Regressor
* Decision Tree Regressor
* Support Vector Regressor (SVR)
* XGBoost Regressor

---

## Results

| Model         | MAE         | RMSE          | R² Score   |
| ------------- | ----------- | ------------- | ---------- |
| Random Forest | 403,803     | 1,129,709     | 0.9864     |
| Decision Tree | 645,129     | 1,782,451     | 0.9682     |
| SVM           | 2,350,000   | 4,150,320     | 0.7654     |
| XGBoost       | **389,210** | **1,078,245** | **0.9881** |

---

## Key Insights

* **XGBoost performed best**, slightly outperforming Random Forest
* **Random Forest remains highly reliable and stable**
* **SVM underperformed** due to dataset scale and complexity
* **Decision Tree showed overfitting tendencies**

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── notebook/
│   └── main.ipynb
├── models/
├── README.md
```

---

## Future Improvements

* Hyperparameter tuning (GridSearch / RandomSearch)
* Feature importance visualization
* Cross-validation for better generalization
* Model deployment (web app / API)
* Integration into a portfolio frontend

---

## How to Run

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the notebook:

```
jupyter notebook
```

---

## Author

**Godz1lla**

---

## License

This project is for educational and portfolio purposes.
