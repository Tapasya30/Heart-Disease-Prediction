# 🫀 Medical AI: Heart Disease Risk Prediction System

An end-to-end clinical machine learning solution designed to predict heart disease risk using diagnostic biomarkers, demographic data, and cardiovascular indicators. Built with **XGBoost**, domain-specific **Medical Feature Engineering**, and an interactive **Streamlit Web Application**.

---

## 🚀 Live Demo

👉 **Try the Heart Disease Risk Prediction App Given Below**

👉 **HERE --->** https://heart-disease-prediction-7cxbz8o9t6as4lekykymov.streamlit.app/

---

## 📌 Executive Summary & Key Highlights

- **Dataset**: 5,000 patient records containing 34 clinical variables (vitals, lipid panel, ECG findings, lifestyle indicators).
- **Best Model**: **XGBoost Classifier** achieving **84.26% CV Accuracy**, **87.67% CV Recall**, **86.45% CV F1-Score**, and **0.9324 CV ROC-AUC** across 5-Fold Stratified Cross-Validation.
- **Clinical Sensitivity Optimization**: Tuned probability decision threshold from `0.50` to `0.40`, increasing diagnostic sensitivity (**Recall > 92%**) to minimize dangerous False Negatives in medical screening.
- **Interactive Web App**: Fully deployed Streamlit diagnostic dashboard for real-time patient risk evaluation and clinical risk tiering (Low, Moderate, High).

---

## 🩺 Domain-Specific Medical Feature Engineering

Rather than feeding raw metrics directly to models, 8 clinically validated cardiovascular risk ratios were engineered:

1. **Cholesterol Risk Ratio**: Total Cholesterol / HDL
2. **Atherogenic Lipoprotein Ratio**: LDL / HDL
3. **Atherogenic Index of Plasma (AIP)**: log10(Triglycerides / HDL)
4. **Insulin Resistance Marker**: Triglycerides / HDL
5. **Pulse Pressure**: Resting BP - Diastolic BP
6. **Mean Arterial Pressure (MAP)**: Diastolic BP + (Resting BP - Diastolic BP) / 3
7. **Heart Rate Reserve (HRR)**: Maximum Heart Rate - Resting Heart Rate
8. **Waist-to-BMI Ratio**: Waist Circumference / BMI

---

## 📊 Model Benchmark Results (Stratified 5-Fold Cross-Validation)

| Rank | Model | CV Accuracy | CV Recall (Sensitivity) | CV Precision | CV F1-Score | CV ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost** | **0.8426 ± 0.0104** | **0.8767 ± 0.0068** | **0.8526 ± 0.0108** | **0.8645 ± 0.0087** | **0.9324 ± 0.0085** |
| 🥈 | **Gradient Boosting** | 0.8432 ± 0.0131 | 0.8725 ± 0.0082 | 0.8565 ± 0.0143 | 0.8644 ± 0.0108 | 0.9432 ± 0.0078 |
| 🥉 | **AdaBoost** | 0.8396 ± 0.0139 | 0.8456 ± 0.0085 | 0.8708 ± 0.0178 | 0.8580 ± 0.0114 | 0.9472 ± 0.0071 |
| 4 | **Random Forest** | 0.8278 ± 0.0146 | 0.8851 ± 0.0119 | 0.8267 ± 0.0156 | 0.8548 ± 0.0118 | 0.9184 ± 0.0094 |
| 5 | **Logistic Regression** | 0.8268 ± 0.0163 | 0.8491 ± 0.0052 | 0.8490 ± 0.0216 | 0.8489 ± 0.0124 | 0.9161 ± 0.0137 |
| 6 | **SVM** | 0.8194 ± 0.0124 | 0.8533 ± 0.0083 | 0.8353 ± 0.0178 | 0.8441 ± 0.0092 | 0.9096 ± 0.0134 |
| 7 | **Extra Trees** | 0.8180 ± 0.0146 | 0.8620 ± 0.0050 | 0.8277 ± 0.0180 | 0.8444 ± 0.0112 | 0.9088 ± 0.0112 |

---

## 🔍 Top Medical Risk Factors (Interpretability)

Feature importance analysis using XGBoost revealed the top clinical indicators for heart disease prediction:

1. **`ST_Depression`** (Exercise-induced ST depression on ECG)
2. **`Major_Vessels`** (Number of major vessels colored by fluoroscopy)
3. **`Cholesterol_HDL_Ratio`** (Engineered lipid ratio)
4. **`Maximum_Heart_Rate`** (Peak exercise heart rate)
5. **`Thalassemia`** (Blood disorder defect status)
6. **`Age`** & **`Pulse_Pressure`**

