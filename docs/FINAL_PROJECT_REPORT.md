# Gradient Boosting on Tabular Data With Leakage-Resistant Validation

## 1. Introduction

Machine learning models are often evaluated using training and testing data. However, if information related to the target variable accidentally enters the training or validation process, the model can show very high performance even though it may not perform similarly in a real situation. This problem is known as data leakage.

In this project, we developed a Gradient Boosting based machine learning system for tabular machine data. The main focus of the project is not only to build a prediction model, but also to identify possible leakage and make sure that the model is evaluated using reliable validation methods.

The project uses sensor-related information such as temperature, pressure, vibration, rotational speed, operating hours, ambient temperature, and load percentage to predict machine failure.

The workflow includes data preparation, feature analysis, leakage analysis, Gradient Boosting model training, random validation, temporal validation, group validation, negative testing, and final acceptance testing.

## 2. Problem Statement

A machine-failure prediction model can give misleading results when information from the future, related records, or target-related features are accidentally included during model training or validation.

For example, a feature that is created after a machine failure occurs should not be used to predict whether that failure will occur. Similarly, records belonging to the same machine should be handled carefully when performing validation so that information from the same machine does not incorrectly appear in both training and testing data.

Therefore, the project aims to develop a machine-learning workflow that can detect these problems and provide safer validation before the model is used.

## 3. Objectives

The main objectives of this project are:

1. To prepare and analyze a tabular machine-failure dataset.
2. To develop a Gradient Boosting baseline model.
3. To study feature importance and feature-target relationships.
4. To identify suspicious features that may indirectly reveal the target.
5. To test temporal and group-based validation.
6. To compare model performance before and after leakage removal.
7. To perform mandatory negative tests for failure conditions.
8. To verify that important results can be reproduced.
9. To perform independent acceptance testing of the developed workflow.
10. To provide a simple interface for making machine-failure predictions.
## 4. Dataset and Data Preparation

### 4.1 Dataset Description

The dataset used in this project is a machine-failure dataset containing 5,000 records and 10 columns. Each record represents the operating condition of a machine at a particular time.

The main columns in the dataset are:

- `timestamp` – records the time of the observation.
- `machine_id` – identifies the machine.
- `temperature` – machine temperature.
- `pressure` – operating pressure.
- `vibration` – vibration level.
- `rotational_speed` – rotational speed of the machine.
- `operating_hours` – number of operating hours.
- `ambient_temperature` – surrounding temperature.
- `load_percent` – machine load percentage.
- `failure` – target variable indicating whether a machine failure is present.

The target variable used for prediction is `failure`.

### 4.2 Feature Selection

For training the Gradient Boosting model, the following seven sensor and operating-condition features were used:

1. Temperature
2. Pressure
3. Vibration
4. Rotational speed
5. Operating hours
6. Ambient temperature
7. Load percentage

The `timestamp` and `machine_id` columns were retained as validation-related information rather than being directly used as model features.

### 4.3 Data Preparation

The timestamp column was converted into a datetime format and the records were sorted according to time. This was important because the dataset has a time component and later experiments required temporal validation.

The dataset was then separated into input features and the target variable. The same approved dataset was used for the main experiments so that the results could be compared under consistent conditions.

The final machine-learning input consisted of the seven selected features, while `failure` was used as the prediction target.

### 4.4 Dataset Quality and Leakage Consideration

Before training the model, the dataset and its features were examined for relationships with the target. This was done because a feature that contains information about the target, either directly or indirectly, can produce misleading model performance.

The project therefore treats leakage detection as a separate part of the workflow instead of relying only on the final model accuracy.

## 5. Baseline Gradient Boosting Model

### 5.1 Model Development

After preparing the dataset, a Gradient Boosting Classifier was trained using the seven selected features.

The dataset was divided into training and testing data. The baseline model was used as the reference model for the later leakage experiments and validation comparisons.

The model was trained using the same feature set that was defined during data preparation. The trained model was also saved so that it could be used later by the prediction API.

### 5.2 Baseline Results

The final baseline model produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7190 |
| Precision | 0.7236 |
| Recall | 0.7682 |
| F1-Score | 0.7452 |
| ROC-AUC | 0.7857 |

These results provide the reference point for the remaining experiments. The model performance was evaluated using multiple metrics instead of relying only on accuracy.

### 5.3 Feature Importance

The feature importance obtained from the trained Gradient Boosting model was:

| Feature | Importance |
|---|---:|
| Operating hours | 0.416697 |
| Vibration | 0.287242 |
| Temperature | 0.148673 |
| Load percent | 0.090140 |
| Pressure | 0.024495 |
| Ambient temperature | 0.020334 |
| Rotational speed | 0.012418 |

Operating hours and vibration contributed the most to the model according to the calculated feature importance. Rotational speed had the smallest contribution among the selected features.

These values were used as part of the feature analysis before performing the leakage experiments.

## 6. Leakage Detection and Controlled Experiment

### 6.1 Why Leakage Detection Was Required

A model can show unusually high performance when a feature contains information that would not actually be available at prediction time. To study this problem, the project includes a separate leakage analysis and controlled testing process.

The project considered indirect or proxy leakage, where a feature does not directly contain the target but still provides strong information about it. Temporal and group leakage were also investigated through separate validation experiments.

### 6.2 Controlled Indirect Leakage Experiment

For a controlled experiment, a synthetic feature named `maintenance_risk_score` was created from the target variable with added random noise.

This feature was intentionally created to represent a situation where a feature indirectly reveals the target. It is a controlled software/data fixture used for testing the leakage-detection workflow and is not presented as a naturally occurring feature in the original machine dataset.

The experiment compared three cases:

1. Clean baseline model.
2. Model trained with the controlled leakage feature.
3. Model after removing the leakage feature.

### 6.3 Experimental Results

The results were:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Clean Baseline | 0.7160 | 0.7160 | 0.7776 | 0.7455 | 0.7796 |
| Indirect Leakage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| After Leakage Removal | 0.7160 | 0.7160 | 0.7776 | 0.7455 | 0.7796 |

The controlled leakage feature produced a ROC-AUC of 1.0 and dominated the model's feature importance. This provided clear evidence that the feature was suspicious and should not be accepted for the final model.

After removing the feature, the model returned to the same performance as the clean baseline. This shows that the unusually high performance caused by the controlled leakage did not represent genuine model improvement.

### 6.4 Leakage Detection Response

When the suspicious feature was detected, the defined safe response was to reject the feature and retrain the model without it.

This approach prevents a misleading validation result from being treated as a genuine improvement.

The experiment also demonstrates an important limitation of the project: the controlled test proves that the implemented workflow can detect the specific type of target-derived leakage used in this experiment. It should not be interpreted as proof that every possible form of leakage can always be detected.

## 7. Validation Experiments

To check whether the model performance was reliable under different data-splitting conditions, three validation approaches were used: random validation, temporal validation, and group validation.

### 7.1 Random Validation

Random validation was used as the standard reference evaluation. The data was divided into training and testing sets and the Gradient Boosting model was evaluated on the test set.

The results were:

| Metric | Result |
|---|---:|
| Accuracy | 0.715 |
| Precision | 0.7155 |
| Recall | 0.7757 |
| F1-Score | 0.7444 |
| ROC-AUC | 0.7796 |

### 7.2 Temporal Validation

Since the dataset contains timestamps, temporal validation was performed to check whether information from future observations was being used to predict earlier observations.

The training period was:

**2025-01-01 00:00:00 to 2025-01-17 15:00:00**

The testing period was:

**2025-01-17 16:00:00 to 2025-01-21 19:00:00**

The temporal validation results were:

| Metric | Result |
|---|---:|
| Accuracy | 0.827 |
| Precision | 0.8279 |
| Recall | 0.9963 |
| F1-Score | 0.9044 |
| ROC-AUC | 0.6413 |

The important point in this experiment is that the training data ended before the testing data began. This prevents future observations from being included in the training data for the temporal test.

### 7.3 Group Validation

Group validation was performed using `machine_id`. This was done to check whether records from the same machine were incorrectly shared between training and testing data.

The machines were divided as follows:

- **Training machines:** M01 to M08
- **Testing machines:** M09 and M10

The results were:

| Metric | Result |
|---|---:|
| Accuracy | 0.714 |
| Precision | 0.7352 |
| Recall | 0.7717 |
| F1-Score | 0.7530 |
| ROC-AUC | 0.7913 |

This validation method provides a different view of model performance because the testing machines were not included in the training group.

### 7.4 Validation Comparison

The three validation methods produced different results, which shows why model evaluation should not depend on a single train-test split.

| Validation Method | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| Random | 0.715 | 0.7444 | 0.7796 |
| Temporal | 0.827 | 0.9044 | 0.6413 |
| Group | 0.714 | 0.7530 | 0.7913 |

The temporal experiment produced a higher accuracy and F1-score but a lower ROC-AUC. Therefore, the results need to be considered using multiple metrics rather than using accuracy alone.

## 8. Mandatory Negative Testing

Negative testing was performed to check how the system behaves when invalid, unsafe, or failure conditions are introduced.

The project contains five mandatory negative tests. These are controlled software/data test fixtures used to verify that the workflow detects problematic conditions and does not silently accept them.

### 8.1 Negative Test Results

| Test | Condition Tested | Result | Safe Response |
|---|---|---|---|
| NT-1 | Validation performance collapses under deployment-like conditions | PASS | Hold deployment and investigate the validation boundary |
| NT-2 | Hidden leakage is exploited by the model | PASS | Reject the suspicious feature and retrain |
| NT-3 | Invalid temporal/group validation split | PASS | Reject the invalid validation split |
| NT-4 | Simulation performance does not transfer to representative conditions | PASS | Enter degraded/safe mode and require representative validation |
| NT-5 | Nominal optimization improves while worst-case risk increases | PASS | Reject the optimization and retain the safer configuration |

### 8.2 Overall Negative-Test Result

All five mandatory negative tests passed:

**5/5 mandatory negative tests PASS**

For every test, the expected failure condition was detected and a safe response was recorded.

The results were saved in:

- `experiments/mandatory_negative_test_results.csv`
- `experiments/mandatory_negative_test_log.json`

These tests provide evidence that the workflow can identify defined failure conditions instead of silently accepting invalid results.
## 9. Acceptance Testing

Acceptance testing was performed to verify that the complete workflow meets the important project requirements under representative and failure conditions.

Four acceptance tests were completed.

### 9.1 AC-1 — Representative Operation

AC-1 checked whether the model's useful performance remained after the controlled leakage feature was removed.

The controlled leakage experiment showed a large performance increase when the suspicious feature was included. After removing the feature, the model returned to the clean baseline performance.

**Result: PASS**

The leakage effect was confirmed, and the model performance survived leakage removal.

### 9.2 AC-2 — Boundary and Failure Operation

AC-2 checked the mandatory negative-test campaign.

All five mandatory failure conditions were detected:

- Leakage condition detected.
- Invalid temporal/group split detected.
- Representative transfer failure detected.
- Unsafe optimization detected.
- Validation boundary failure detected.

**Result: PASS**

All five mandatory negative tests passed and safe responses were recorded.

### 9.3 AC-3 — Independent Acceptance Evidence

AC-3 independently repeated the important controlled leakage experiment using the approved dataset and configuration.

The independent test produced:

| Measurement | Result |
|---|---:|
| Leakage feature ROC-AUC | 1.0000 |
| Clean model ROC-AUC | 0.7771 |
| Leaky model ROC-AUC | 1.0000 |
| Recovered model ROC-AUC | 0.7771 |

The clean and recovered models produced identical metrics in the independent run. The metric difference for accuracy, precision, recall, F1-score, and ROC-AUC was zero.

**Result: PASS**

This provides independent evidence that the controlled leakage effect and the recovery after leakage removal can be reproduced.


### 9.4 AC-4 — Frozen Resource Envelope

AC-4 checked whether the final evaluation was performed under the declared and frozen project resource envelope.

The acceptance test verified that the reference and final configurations used the same:

- Python version
- pandas version
- NumPy version
- scikit-learn version
- joblib version
- dataset
- dataset size
- feature count
- target variable
- workload
- model

The test also verified that no hardware, workload, dataset, or software-version changes were declared between the reference and final configurations.

**Result: PASS**

The frozen resource-envelope evidence was accepted, and the required resources were recorded as unchanged.

### 9.5 KPI Evaluation

The project KPIs were evaluated using the generated experimental and negative-test evidence.

| KPI | Measured Value | Required Threshold | Status |
|---|---:|---:|---|
| KPI-1 — Leakage Detection Effectiveness | 100.0 | 80.0 | PASS |
| KPI-2 — Performance After Leakage Removal | 100.0 | 80.0 | PASS_WITH_LIMITATION |
| KPI-3 — Split Integrity Test Pass Rate | 100.0 | 80.0 | PASS_WITH_LIMITATION |
| KPI-4 — Safe-Recovery Success Under Injected Faults | 100.0 | 80.0 | PASS_WITH_LIMITATION |
| KPI-5 — Normalized Tracking/Regulation RMSE | N/A | N/A | NOT_APPLICABLE |
| KPI-6 — Safety-Envelope Violation Count | 0 | 0 | PASS |

KPI-1 reached a measured value of 100.0 based on the controlled leakage-detection evidence. The controlled leakage feature was detected and subsequently rejected.

KPI-2 reached 100.0 because the model performance was retained after the controlled leakage feature was removed. However, a numeric O2 reference KPI score was not recorded, so the required comparison against the O2 reference cannot be claimed.

KPI-3 reached 100.0 based on the completed split-integrity evidence. The temporal/group validation failure condition was detected. As with KPI-2, a numeric O2 split-integrity reference was not recorded.

KPI-4 reached 100.0 because all five mandatory negative tests detected their defined failure conditions and recorded safe responses. A numeric O2 recovery reference was not recorded.

KPI-5 was marked not applicable because this project is a binary classification system rather than a tracking or regulation regression task. Therefore, an RMSE value was not fabricated.

KPI-6 recorded zero safety-envelope violations across the defined mandatory adversarial software/data test conditions. This result represents the project's controlled software/data assurance evidence and is not a claim of physical safety certification.

The KPI evidence was saved in:

- `experiments/kpi_results.csv`
- `experiments/kpi_report.json`
### 9.6 Acceptance Summary

| Acceptance Test | Status |
|---|---|
| AC-1 — Representative operation | PASS |
| AC-2 — Boundary/failure operation | PASS |
| AC-3 — Independent acceptance evidence | PASS |
| AC-4 — Frozen resource envelope | PASS |

All four acceptance tests passed.

## 10. System Architecture and Implementation

### 10.1 Overall Workflow

The developed system follows a sequence of steps from data preparation to final prediction and validation.

The main workflow is:

**Dataset → Data Preparation → Feature Analysis → Leakage Detection → Validation → Gradient Boosting Model → Evaluation → Prediction API**

The validation stage includes random, temporal, and group-based validation. The testing stage includes negative tests and acceptance tests to check the behavior of the complete workflow.

### 10.2 Project Structure

The project is organized into separate folders so that the different parts of the system can be maintained independently.

```text
GradientBoostingLeakageProject/
│
├── data/
├── experiments/
├── ml/
├── models/
├── backend/
├── frontend/
├── tests/
├── docs/
├── main.py
└── requirements.txt
```

### 10.3 Machine Learning Components

The machine-learning part of the project is divided into separate scripts so that each major part of the workflow can be tested independently.

The `ml` folder contains scripts for data analysis, baseline model training, the main Gradient Boosting pipeline, leakage detection, leakage-resistant model evaluation, temporal validation, group validation, and the controlled indirect leakage experiment.

The controlled indirect leakage experiment is also implemented separately. This makes it possible to compare the clean model, the model with the controlled leakage feature, and the model after leakage removal.
### 10.4 Prediction API

A FastAPI backend was developed to provide predictions from the trained Gradient Boosting model.

The API accepts the seven input features used by the model and returns the predicted result, failure probability, model name, and leakage-resistant status.

The API was tested using both normal and higher-risk machine conditions.

For the normal test case, the system returned:

**Prediction: NORMAL**

**Failure Probability: 13.8%**

For the higher-risk test case, the system returned:

**Prediction: FAILURE_RISK**

**Failure Probability: 62.0%**

The API also validates the input values before sending them to the model. If a required input is empty or invalid, the request is not sent to the model and an invalid-input message is displayed.
 This helps avoid invalid requests and provides a basic safe response at the user-interface level.
### 10.5 Frontend

A simple web-based frontend was developed to interact with the prediction API.

The frontend provides input fields for the seven sensor and operating features:

- Temperature
- Pressure
- Vibration
- Rotational Speed
- Operating Hours
- Ambient Temperature
- Load Percent

After entering the values, the user can submit them for prediction. The frontend sends the input data to the FastAPI backend and displays the prediction result and failure probability.

The interface also shows the API connection status so that the user can know whether the backend is available.

Input validation was added to prevent empty fields from being submitted. This helps avoid invalid requests and provides a basic safe response at the user-interface level.
## 11. Reproducibility

Reproducibility was treated as an important part of the project so that the main experiments could be repeated using the same software environment and dataset.

The project environment was recorded using `pip freeze` and saved as:

`experiments/reproducibility_environment.txt`

The main software versions used during the final testing were:

| Software | Version |
|---|---:|
| Python | 3.14.2 |
| Pandas | 3.0.5 |
| NumPy | 2.5.2 |
| Scikit-learn | 1.9.0 |
| Joblib | 1.5.3 |

The complete list of installed packages and versions is available in the reproducibility environment file.

The acceptance experiments and their results were also saved as separate CSV and JSON files. This provides a record of the conditions and results used during the final validation.

The independent acceptance test was run separately from the original acceptance test. It reproduced the controlled leakage effect and produced the same clean and recovered ROC-AUC value of 0.7771.

The source code and experiment evidence were committed to Git and pushed to the project's GitHub repository. This allows the project files and recorded evidence to be tracked as part of the development history.
## 12. Limitations

The project demonstrates leakage-resistant validation using the available machine dataset and controlled experiments. However, there are some limitations.

The indirect leakage experiment uses a synthetically created target-derived feature called `maintenance_risk_score`. It was created specifically to test whether the leakage detection process can identify a feature that contains information from the target variable. Therefore, this experiment should not be considered evidence of naturally occurring leakage in a real industrial dataset.

The dataset contains 5000 records and a limited number of machine and sensor features. The results may therefore change when the system is applied to larger datasets, different machines, or different operating environments.

The project evaluates temporal and group validation using the available timestamp and machine identifiers. These validation methods demonstrate the importance of respecting time and machine boundaries, but they do not cover every possible validation problem that can occur in real-world deployments.

The leakage detector is also not intended to identify every possible form of data leakage. The project mainly demonstrates detection of the controlled target-derived leakage case and validation-boundary problems included in the test scenarios.

Finally, the prediction API and frontend were developed as a project demonstration. Additional security, authentication, monitoring, deployment infrastructure, and production-level error handling would be required before using the system in a real industrial environment.
## 13. Conclusion

This project developed a Gradient Boosting based machine learning system with a focus on detecting data leakage and using validation methods that are closer to real-world conditions.

The project started with a baseline Gradient Boosting model and then examined the effect of different validation strategies. Random, temporal, and group validation were performed to understand how the evaluation results can change when the data is separated using different boundaries.

A controlled indirect leakage experiment was also performed using a synthetic target-derived feature. The leakage feature produced a ROC-AUC of 1.0000 and was successfully detected. After removing the feature, the model returned to the clean baseline performance, showing that the unusually high performance was caused by the injected leakage.

The project also included mandatory negative tests covering invalid validation data, leakage, temporal and group split violations, representative transfer failure, and unsafe optimization. All five mandatory negative tests passed and their safe responses were recorded.

Acceptance testing was performed to verify representative operation, boundary and failure conditions, and independent acceptance evidence. The independent acceptance test reproduced the controlled leakage effect and confirmed the clean and recovered model results.

Finally, a FastAPI prediction service and a web-based frontend were developed to demonstrate how the trained model can be used for machine failure prediction.

Overall, the project shows the importance of checking for leakage, validating data using appropriate boundaries, testing failure conditions, and maintaining reproducible evidence when developing machine learning systems.