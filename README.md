# Project3-Deep-Learning-

Team Members: S Deepthi Reddy, Meka Shashank; Context-aware two-tower deep learning recommendation engine for fashion e-commerce, built on the H&M dataset. Uses PySpark for data processing, TensorFlow Recommenders for the model, Redis for caching and FastAPI for real-time recommendations.

## Data
Original source: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

Place transactions_train.csv, customers.csv and articles.csv in data/raw/

## User Data Processing Pipeline

Implemented using PySpark as part of Week 1.

### Tasks Completed

1. Customer data inspection and missing-value analysis.
2. Customer data cleaning and preprocessing.
3. Cold-start user identification.
4. User purchase recency and transaction-count feature generation.
5. Last 10 transaction history generation for each user.
6. User vocabulary generation with unique numerical IDs.

### Output

Processed datasets are saved in the `data/processed/` directory in Parquet format.

### Scripts

All user-processing scripts are available in the `user_pipeline/` directory.
