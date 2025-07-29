# real-estate-price-predictor
House Price Prediction Project

This project uses a supervised machine learning approach to predict house prices based on various features such as location, number of bedrooms, area, and more. The model is trained on a dataset sourced from Kaggle and performs regression analysis to estimate house prices.

📁 Dataset

Source: Kaggle - Housing Prices Dataset

File used: Housing.csv
Size: Medium (~500 entries)
Features include:
Area
Number of Bedrooms
Number of Bathrooms
Furnishing status
Parking, etc.

🧠 Techniques Used

Data Preprocessing
Categorical Encoding with pd.get_dummies
Train-Test Split (80-20)
Linear Regression
Evaluation with Mean Squared Error (MSE) and R^2 Score
Visualization with Matplotlib & Seaborn


🛠 Project Structure

House-Price-Prediction/
├── data/
│   └── Housing.csv
├── notebooks/
│   └── House_Price_Prediction.ipynb
├── src/ (optional for modular code)
├── README.md

📈 Results
Model Used: Linear Regression

Performance:
Mean Squared Error: Depends on dataset
R² Score: Indicates goodness of fit



📊 Visualization
A scatter plot compares predicted prices against actual prices to show how close the model's predictions are.
