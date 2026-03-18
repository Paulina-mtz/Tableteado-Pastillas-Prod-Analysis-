# Tablet Production Analysis

## Overview
This repository contains a data science project focused on analyzing and reducing tablet friability in a pharmaceutical manufacturing context. The challenge was introduced by PiSA, which shared the industrial problem statement and the operational context of the tableting process. Based on that challenge, the project team developed a predictive analytics workflow to identify the process variables that most strongly influence friability and to support data-driven decision-making.

The project combines data preprocessing, exploratory analysis, machine learning, model interpretation, and an interactive Streamlit app for variable-level analysis.

## Problem Statement
In the tableting stage of pharmaceutical production, friability and hardness are critical quality attributes. According to the challenge presentation, the main need was to identify the parameters that significantly affect tablet friability and define operating values that help reduce friability while maintaining acceptable hardness. The presentation also highlights the relationship between compression force, hardness, and friability, as well as the importance of the tableting operation and its critical parameters. fileciteturn1file0

## Project Objective
The objective of this project is to build predictive models that explain variation in tablet friability and help identify which operational variables should be monitored or adjusted to improve product quality.

## Repository Structure
```text
Tablet_Production_Analysis/
├── app/
│   └── app.py
├── data/
│   └── datos.csv
├── docs/
│   └── PiSA_Challenge_Presentation.pptx
├── src/
│   └── Pastillas_Prod_Analysis.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset
The dataset includes production-related process and quality variables used to model friability. Examples of variables used in the analysis include pre-compression force, compression force, average weight, minimum and maximum weight, thickness, diameter, hardness, and friability.

## Methodology
### 1. Data preprocessing
- Loaded and cleaned the production dataset.
- Renamed columns for consistency.
- Removed unnecessary indexing columns.
- Sorted observations by batch index.
- Scaled the predictor variables for model training.

### 2. Predictive modeling
Three machine learning models were trained to predict friability:
- Random Forest Regressor
- Multilayer Perceptron Regressor
- XGBoost Regressor

### 3. Model evaluation
The models were evaluated using:
- Mean Squared Error (MSE)
- R-squared (R²)

### 4. Explainability and interpretation
To interpret model behavior, the project includes:
- Feature importance from Random Forest and XGBoost
- Permutation importance for the neural network
- SHAP values for model explainability
- Weighted average importance across models
- Partial dependence plots for variable-level interpretation

### 5. Interactive application
A Streamlit app was built to let users:
- Select a variable
- Visualize its relative importance
- Inspect partial dependence behavior
- Read an interpretation of its effect on friability
- Review process suggestions derived from the model outputs

## Key Deliverables
- End-to-end friability prediction workflow
- Variable importance ranking
- Model interpretation with SHAP and partial dependence plots
- Streamlit interface for interactive exploration
- Reproducible Python-based analysis pipeline

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit

## How to Run
### 1. Clone the repository
```bash
git clone <your-repository-url>
cd Tablet_Production_Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analysis script
From the project root:
```bash
python src/Pastillas_Prod_Analysis.py
```

### 4. Launch the Streamlit app
```bash
PYTHONPATH=. streamlit run app/app.py
```

## Notes
- The file in `docs/` corresponds to the presentation provided by PiSA describing the industrial challenge.
- The rest of the repository corresponds to the project solution developed from that challenge.
- If needed, the code can be refactored into functions or notebooks for cleaner reuse and deployment.

## Suggested Future Improvements
- Add a notebook version for easier walkthrough of the analysis.
- Include train/test metrics in a results section.
- Add plots exported as static images.
- Refactor the analysis script into modular functions.
- Deploy the Streamlit app publicly.

## Author
Paulina Martinez Lopez
