# playground-series-s3e21

## Description
This is the repository for the Playground Series, Season 3, Episode 21 - "Improve a Fixed Model the Data-Centric Way!"

See the full description of the competition here: https://www.kaggle.com/competitions/playground-series-s3e21

### Overview
This is the 2023 edition of Kaggle's Playground Series.
The series aims to offer light-weight challenges for the Kaggle community, focusing on a variety of machine learning and data science aspects.
The Tabular Tuesday will be launched every Tuesday at 00:00 UTC in August, with each challenge spanning 3 weeks.
Competitions utilize synthetic datasets crafted from real-world data to promote quick iterations on model and feature engineering, visualizations, and more.

### Synthetic Datasets
Playground competitions use synthetic data to balance real-world data relevance while keeping test labels private.
The current state of synthetic data generation is advanced, aiming to provide datasets with minimal artifacts.

### Evaluation & Task
Unique competition type where participants submit a dataset.
This dataset is used to train a random forest regressor which predicts against a hidden test set.
The evaluation metric is the Root Mean Square Error (RMSE) between model predictions and the hidden test set's actual values.

The code below shows the training that will be performed on the participant's submission.

```python
from sklearn.ensemble import RandomForestRegressor

y_train = train.pop('target')  # 'train' is the participant's submission
rf = RandomForestRegressor(
       n_estimators=1000,
       max_depth=7,
       n_jobs=-1,
       random_state=42)
rf.fit(train, y_train)
y_hat = rf.predict(test)  # 'test' set remains undisclosed to participants

```
## EDA and Feature Engineering Notebook Summary
### 1. Data Pre-processing:
- **Data Loading:** Loaded data from `sample_submission.csv`.
- **Missing Values:** Checked and handled missing or NaN values.
- **Column Removal:** Dropped the 'id' column for visualization purposes.
- **Visualization:** Utilized boxplots to understand the range and potential outliers.
- **Outlier Removal:** Identified and removed an outlier from `NH4_5` column.

### 2. Data Visualization:
- **Distributions:** Plotted boxplots and histograms for column distributions post-outlier removal.
- **Correlation Analysis:** Generated a heatmap to visualize inter-column correlations.

### 3. Feature Importance:
- **Model Choice:** Employed RandomForestRegressor for feature importance assessment.
- **Key Features:** Highlighted `O2_1`, `O2_2`, and `BOD_5` as particularly influential.

### 4. Model Evaluation and Outlier Treatment:
- **Evaluation Techniques:** Leveraged KFold cross-validation with multiple preprocessing techniques:
  - Direct dataset use post outlier removal.
  - Isolation Forest for outlier treatment.
  - Column zeroing, except specified columns.
  - LocalOutlierFactor for outlier detection.
  - SGDOneClassSVM for outlier detection.
  - Simple clipping on target values.
- **Performance Metrics:** Tracked MSE (Mean Squared Error) and MAE (Mean Absolute Error) for each method.

### 5. Model Training for Submission:
- **Data Treatment:** Combined various outlier treatments:
  - Target column clipping.
  - SGDOneClassSVM, Isolation Forest, and LocalOutlierFactor for outlier handling.
  - Row removal based on known low-quality labels (referenced from another Kaggle notebook).
  - Column zeroing, with exception rules.
- **Training & Evaluation:** Trained a RandomForestRegressor on the processed dataset and evaluated with KFold cross-validation.
- **Submission:** Processed data and predictions saved as `submission.csv`.


## VAE and Auto-Encoder Notebook Summary
### **1. PCA Visualization**:
#### Objective: Visualize original and generated samples in 2D.
  - Apply PCA to reduce data dimensions to 2D.
  - Plot 2D projections using scatter plots.

### **2. Auto-Encoder Architecture**:
#### Objective: Define an auto-encoder for data reconstruction.
  - Utilize layers to encode the input data into a compressed representation.
  - Decode the compressed representation back to original dimensions.

### **3. Auto-Encoder Training**:
#### Objective: Train the auto-encoder for accurate data reconstruction.
  - Train on normalized data over multiple epochs with the Adam optimizer.
  - Track training using a custom loss, save best model based on loss.
  - Visualize training loss progression.

### **4. VAE (Variational AutoEncoder) Architecture**:
#### Objective: Define VAE for data generation.
  - **Encoder**: Outputs both mean and log variance for latent space.
  - **Decoder**: Generates input from the latent space.
  - Utilize reparameterization for latent space sampling.

### **5. VAE Training**:
#### Objective: Train VAE for accurate data reconstruction.
  - Train on normalized data over multiple epochs with the Adam optimizer.
  - Track training using a custom loss, save best model based on loss.
  - Visualize training loss progression.

### **6. Data Generation**:
#### Objective: Create new samples with VAE and Auto-Encoder.
  - Generate samples using the trained VAE, sampling from standard normal distribution.
  - Reconstruct data using the trained auto-encoder.
  - Scatter plot comparisons: original vs. generated vs. reconstructed samples.

### **7. Model Evaluation with RandomForestRegressor**:
#### Objective: Assess performance using RandomForest.
  - Implement K-fold validation.
  - Evaluation metrics: Mean squared error (MSE) and mean absolute error (MAE).
  - Custom function (`kfold_train_model`) for k-fold based training.

### **8. RMSE Contribution Calculation**:
#### Objective: Quantify RMSE contribution for each data point.
  - Use `compute_rmse_contributions` to measure RMSE using cross-validation.
  - Rank data points based on RMSE contributions.
  - Compute contributions for original, generated (VAE & auto-encoder), and reconstructed data.

### **9. Data Augmentation**:
#### Objective: Enrich original dataset.
  - Retain original data, and complement with generated/reconstructed data.
  - Pick worst-performing samples (highest RMSE) from generated/reconstructed data.
  - Augment original dataset with these samples.

### **10. Training on Augmented Data**:
#### Objective: Train and assess RandomForest on augmented datasets.
  - Training datasets: 
    1. Original 
    2. Original + VAE-generated samples 
    3. Original + auto-encoder reconstructed samples.
  - Compare models using MAE and MSE metrics.
  - Conclude on data generation method efficacy.

### **11. Final Dataset Generation**:
#### Objective: Ready dataset for deployment/submission.
  - Inspect augmented dataset's shape.
  - Export final dataset to 'submission.csv'.


