# Kidney Disease Classification Project

This project demonstrates a complete machine learning pipeline to classify chronic kidney disease (CKD) using the UCI Chronic Kidney Disease dataset. The repository includes data exploration, cleaning, feature engineering, and the development of multiple predictive models.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Data Exploration & Preprocessing](#data-exploration--preprocessing)
  - [Feature Selection](#feature-selection)
  - [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The aim of this project is to predict chronic kidney disease by leveraging multiple machine learning techniques. A robust data processing pipeline is implemented and three classifiers are compared:
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Artificial Neural Networks (ANN)**

Hyperparameter tuning is performed using GridSearchCV, and the models are evaluated using confusion matrices and classification reports.

## Dataset

The dataset used in this project is the **Chronic Kidney Disease dataset (UCI ID: 336)** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease). It contains 400 instances and 25 attributes (24 features plus the target), including clinical measurements such as:
- Age
- Blood Pressure
- Specific Gravity
- Albumin
- Sugar
- Red Blood Cells
- And more…

The dataset has missing values and includes both numerical and categorical features that are carefully preprocessed.

## Project Structure

├── data/ # (Optional) Folder for dataset files (if not using direct download) ├── kidney_Classification_Code_.ipynb # Main Jupyter Notebook with the full analysis ├── Final_Project_kidney_disease_Classification.pdf # Project report with details and figures ├── requirements.txt # List of required Python packages.

# Install the dependencies:

Ensure you have Python 3.x installed, then run:

bash
Copy
pip install -r requirements.txt
This project relies on libraries such as:

**pandas**

**numpy**

**matplotlib**

**seaborn**

**scikit-learn**

**ucimlrepo**

## Usage

To explore and run the project:

Launch Jupyter Notebook:

bash
Copy
jupyter notebook kidney_Classification_Code_.ipynb
Execute the Notebook:

Run through the notebook cells to perform data preprocessing, exploratory data analysis, model training, and evaluation.

Alternatively, you can open the notebook in Google Colab using the link provided in the repository.

## Methodology

**Data Exploration & Preprocessing**

**Data Import & Cleaning:**

The dataset is imported using the ucimlrepo package. Missing numerical values are imputed with the mean, and categorical values with the mode.

## Visualization:

Heatmaps, histograms, and boxplots are used to understand feature distributions and to identify outliers.

## Encoding:

Categorical features are label-encoded to make them suitable for machine learning models.

## Feature Selection

Correlation Analysis:
A correlation matrix is generated to assess relationships between features and the target variable.

## Random Forest Feature Importance:

Feature importance scores from a Random Forest model are used to select the top predictors.

## Model Training & Evaluation

Data Splitting & Normalization:
The dataset is split into training and testing sets (using stratification) and normalized using MinMaxScaler.

## Model Building:

Three classifiers are built and tuned:

**Random Forest:** Parameters are tuned using GridSearchCV.

**K-Nearest Neighbors (KNN):** Optimal parameters are selected via GridSearchCV.

**Artificial Neural Network (ANN):** A Multi-Layer Perceptron is tuned for optimal performance.

## Evaluation:

Models are evaluated using test set accuracy, confusion matrices, and detailed classification reports.

## Results

The tuned models achieved high accuracy:

**Random Forest:** Best CV Score ~99% with test accuracy ~96%.

**KNN:** Test accuracy ~97%.

**ANN:** Test accuracy ~97%.

Visualizations such as confusion matrices and heatmaps of classification reports provide further insights into model performance.

## Contributing

Contributions are welcome! Please fork the repository, make improvements, and submit a pull request. It is recommended to open an issue first to discuss your proposed changes.

## Acknowledgements

**Dataset:** UCI Machine Learning Repository – Chronic Kidney Disease dataset.

**Inspiration:** Thanks to the data science community for their support and resources.

**Author:** Shahroze-Gondal
