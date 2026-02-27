# Marketing Campaign Analysis & Prediction

## Project Overview
This project aims to analyze customer data to understand behavioral patterns and predict campaign responses. By building a robust Machine Learning Pipeline, we can automate marketing decisions and target high-potential customers effectively.

## Dataset
The dataset consists of **2,240 Customers** with **29 unique features**, including Demographics, Spending Habits, and Campaign History. 

## Methodology
1. **Data Preprocessing:** Handled missing values (Median Imputation for Income), scaled numerical features using `StandardScaler`, and applied `One-Hot Encoding` for categorical variables.
2. **Feature Engineering:** Derived new features like `Age` and `Customer_Tenure` to capture life-stage influence and loyalty duration.
3. **Feature Selection:** Utilized `SelectKBest` with ANOVA F-value to identify the top 10 most influential features.
4. **Handling Imbalanced Data:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance, improving the model's sensitivity (Recall & F1-Score) to actual responders.
5. **Modeling:** Trained a **GradientBoostingClassifier** within an automated Imbalanced Pipeline (`ImbPipeline`) to ensure zero data leakage.

## Results
The project was completed successfully with a final model accuracy of **90.18%**, showing robust performance across diverse customer segments.
