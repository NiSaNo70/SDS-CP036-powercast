PowerCast ⚡: A Deep Learning App for Electricity Demand Forecasting
Live Application Link: https://sds-cp036-powercast-nima.streamlit.app/

Overview
PowerCast is an end-to-end data science project that forecasts short-term electricity demand for three different zones in Tetuan City, Morocco. The project demonstrates a complete machine learning pipeline, from initial data exploration and feature engineering to model training, optimization, and final deployment as an interactive web application.

The core of the application is a Bidirectional LSTM (Long Short-Term Memory) model built with PyTorch, which is capable of learning complex temporal patterns from historical weather and power consumption data.

Features
Interactive Forecasting: Users can upload the historical dataset, select any date and time, and receive an on-demand forecast for the next 10 minutes.

Multi-Zone Prediction: Separate deep learning models are trained and deployed for each of the three zones, allowing for specialized predictions.

Time-Series Analysis: The project includes a deep dive into time-series feature engineering, using cyclical features to capture daily and weekly seasonality.

Model Interpretability: Uses SHAP (SHapley Additive exPlanations) to understand which features (like temperature, humidity, or time of day) are driving the model's predictions.

Tech Stack
Modeling & Data Science: Python, PyTorch, Scikit-learn, Pandas, NumPy

Web Application & Deployment: Streamlit, Streamlit Community Cloud

Data Source: UCI Machine Learning Repository - Tetouan City Power Consumption
