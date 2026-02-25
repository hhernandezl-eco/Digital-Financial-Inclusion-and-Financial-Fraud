# {Digital Financial Inclusion and Financial Fraud: Evidence from Self-Reported Complaints on Social Media using Multilingual Transformers Models

## Research Overview
This repository contains the computational and econometric replication materials for the study on the impact of self-reported financial fraud on Digital Financial Inclusion. The methodology integrates natural language processing and panel data econometrics. The code is structured to execute three analytical stages. First, it implements multilingual transformer models to extract embeddings from a corpus of social media complaints in Spanish and Portuguese. Second, it applies dimensionality reduction and density-based clustering to categorize fraud typologies. Third, it executes panel data regression models with fixed effects to estimate the statistical relationship between the constructed index and financial inclusion metrics across the analyzed jurisdictions.

## Repository Structure

The project is organized into modular stages to ensure a clean data pipeline and seamless replication:

├── "data"
│   ├── "raw"
│   │   ├── "dataset_fraude_pt.csv"
│   │   ├── "dataset_fraude_es.csv"
│   │   ├── "keywords.csv"
│   │   ├── "General_Corruption_Index.csv"
│   │   └── "IFI_Final.csv"
│   └── "processed"
│       ├── "dataset_fraude_clasificado.csv"
│       └── "dataset_FRISK_panel.csv"
├── "src"
│   ├── "01_extraccion_embeddings.py"
│   ├── "02_clasificacion_clustering.py"
│   └── "03_analisis_econometrico.py"
├── "results"
│   └── "tablas_regresion.txt"
└── "README.md"
