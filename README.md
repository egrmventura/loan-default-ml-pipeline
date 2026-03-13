# Loan Default Prediction ML Pipeline

## Overview
End-to-end ML pipeline exercise for predicting loan default risk with data engineering, feature pipelines, and model deployment.

Key formula on case-by-case:
EL = PD x LGD x EAD

EL = Expected Loss
PD = Probablity of Default — Probability the borrower will default within a time horizon (usually 12 months)
LGD = Loss Given Default — Percentage of loan lost if default occurs
EAD = Exposure at Default — How much mone is owed when default occurs

## Architecture
Data → Feature Engineering → Model Training → API → Monitoring

## Stack
Python
Pandas
XGBoost
MLflow
Airflow
FastAPI
Docker

## Dataset
LendingClub dataset gathered from OpenIntro resource at:
https://www.openintro.org/data/index.php?data=loans_full_schema

## Pipeline
1. Data Ingestion
2. Data Validation
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Deployment
