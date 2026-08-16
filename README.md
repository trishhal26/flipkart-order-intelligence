# Flipkart Order Intelligence

An end-to-end machine learning and AI project for Flipkart-style e-commerce intelligence.

## Project Parts

### Part 1 — Return Risk Modeling

Built a return-risk prediction pipeline using generated e-commerce order data.

- Dataset: 6,000 orders, 13 features
- Baseline: DummyClassifier
- Logistic Regression with threshold tuning
- Random Forest with GridSearchCV
- Feature importance and permutation importance analysis
- Product-category and payment-method subgroup analysis
- Final model saved as `models/return_risk_model.pkl`
- Random Forest F1-maximising threshold: `t*_rf = 0.47`

### Part 2 — Product Image Categorisation

Transfer-learning image classifier using Fashion-MNIST and a pretrained CNN backbone.

### Part 3 — AI Agent

A LangGraph-based agent that connects the project's prediction and image-classification capabilities into a single workflow.