# 🗽 US Gun Violence Hotspot Prediction: A Spark ML Stacking Ensemble Approach

## 📝 Project Summary

This project analyzes the Gun Violence Archive data to identify potential **"Hotspots"**—incidents resulting in a high number of casualties—using Apache Spark and advanced Machine Learning techniques. The primary goal is to build a highly accurate predictive model to flag events most likely to escalate into major incidents based on temporal and geographical features.

The core challenge addressed is the **severe class imbalance** (less than 3% of incidents are classified as Hotspots), which is mitigated using an **Oversampling** technique (SMOTE-like sampling with noise) before training a **Stacking Ensemble Model**.

## 🎯 Problem Definition (The "Hotspot" Target Variable)

The business problem is a **binary classification** task: predicting whether a gun violence incident will be a "Hotspot" or a "Non-Hotspot".

An incident is defined as a **Hotspot (Target = 1)** if:
1. The total number of victims (Killed + Injured) is **greater than or equal to 5**, **OR**
2. The number of people killed is **at least 2**.

All other incidents are classified as Non-Hotspot (Target = 0).

## 🛠️ Methodology & Spark Pipelines

The entire analysis and modeling pipeline is implemented using **PySpark** for scalable processing.

### 1. Data Cleaning and Preprocessing

* **Initial Filteration:** Dropped rows with missing values in the core analytical columns (`date`, `state`, `city_or_county`, `latitude`, `longitude`, `n_killed`, `n_injured`).
* **Type Casting:** Converted core columns to their correct types: `date` (to DateType), `n_killed`, `n_injured` (to IntegerType), and `latitude`, `longitude` (to DoubleType).
* **Anomaly Handling:** Specifically filtered out an invalid URL string entry found in the `latitude` column.

### 2. Feature Engineering

| Feature Category | Features Created | Description |
| :--- | :--- | :--- |
| **Temporal** | `year`, `month`, `day_of_week`, `day_of_month` | Extracted from the `date` column to capture seasonal and weekly patterns. |
| **Spatial** | `lat_grid`, `lon_grid` | Latitude and longitude were **rounded to 3 decimal places** (e.g., `40.3467` -> `40.347`). This creates spatial "bins" or grids for efficient grouping. |
| **Aggregate** | `incident_count`, `total_killed`, `total_injured`, `avg_killed_per_incident` | Calculated by grouping the data by the new `lat_grid` and `lon_grid`. This enriches each record with contextual statistics from its immediate geographic area, which are crucial predictors for Hotspots. |

### 3. Class Imbalance Resolution (Oversampling)

The original dataset exhibits extreme class imbalance:
* **Non-Hotspot (0):** 97.56% (226,090 rows)
* **Hotspot (1):** 2.44% (5,663 rows)
* **Ratio:** $\approx 40:1$

To ensure the model can learn from the minority class, a **SMOTE-like oversampling** technique was applied:
* The minority class (Hotspot) was oversampled with replacement and a small random noise added to the spatial coordinates (`latitude`, `longitude`) to reduce overfitting.
* The final balanced training set achieved a more manageable **New Ratio of $\approx 3.0:1$**.

### 4. Model Training: Stacking Ensemble

The final classification is performed by a **Stacking Ensemble**, which leverages the strengths of multiple models:

1.  **Level 1 (Base Models):**
    * **Logistic Regression (LR)**
    * **Decision Tree Classifier (DT)**
    * **Gradient Boosted Trees (GBT)**
2.  **Level 2 (Meta-Model):**
    * A final **Logistic Regression** model is trained on the prediction probabilities (from LR & DT) and the prediction label (from GBT) of the base models.

The ensemble approach aims for higher robustness and predictive performance than any single model alone.

## 📊 Key Results

The model was evaluated on a held-out test set, demonstrating excellent performance in classifying Gun Violence Hotspots.

| Metric | Logistic Regression (Base) | Decision Tree (Base) | GBT (Base) | **Stacking Ensemble (Final)** |
| :--- | :--- | :--- | :--- | :--- |
| **AUC-ROC** | 0.9557 | 0.9590 | **0.9876** | 0.9794 |
| **Accuracy** | 0.6424 | 0.9637 | 0.9629 | **0.9623** |
| **F1-Score** | 0.6424 | 0.9637 | 0.9629 | **0.9629** |

### Stacking Ensemble Performance Details

The final Stacking Ensemble model achieved high class-specific accuracy, demonstrating its ability to handle the previously imbalanced nature of the data:

* **Overall AUC-ROC:** **0.9794**
* **Classification Accuracy:** 0.9623
* **F1-Score:** 0.9629

| Class | Total Samples | Correct Predictions | **Class Accuracy** |
| :--- | :--- | :--- | :--- |
| **Non-Hotspot (0)** | 33,860 | 32,491 | **0.9596** |
| **Hotspot (1)** | 11,168 | 10,840 | **0.9706** |
