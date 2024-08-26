# House Prices Prediction

This project aims to predict house sale prices in Ames, Iowa. Using a dataset from Kaggle, advanced machine learning techniques, particularly regression algorithms like Gradient Boosting and Random Forest, are applied to develop a model that predicts house prices with high accuracy. The project includes data preparation, feature engineering, and model evaluation, with performance assessed using the Root Mean Squared Error (RMSE) metric.

### Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and tools
- **xgboost**: Extreme Gradient Boosting
- **warnings**: Warning management

### Project Steps

1. **Import Data**
   - Loaded the dataset from CSV files:
     ```python
     test = pd.read_csv('test.csv')
     train = pd.read_csv('train.csv')
     df = pd.concat([train, test], axis=0, ignore_index=True)
     ```

2. **Filling Missing Data and Data Conversion**
   - Used KNN Imputer for filling missing values in numerical features.
   - Dropped columns with a large number of missing values.
   - Applied mode and median strategies for categorical and numerical data, respectively.
   - Encoded categorical features using LabelEncoder and One-Hot Encoding.

3. **Feature Engineering**
   - Created new features such as `AgeAtSale`, `RemodelAge`, and `TotalBathrooms`.
   - Applied feature transformations and created dummy variables for categorical columns.
   - Mapped ordinal categorical variables to numerical values.

4. **Modeling**
   - Split the data into training and testing sets.
   - Trained and evaluated various regression models, including Linear Regression, Ridge, Lasso, ElasticNet, Gradient Boosting, XGBRegressor, and others.
   - Selected the Gradient Boosting model based on performance metrics.

5. **Results**
   - The Gradient Boosting model achieved an R-squared score of approximately 0.894, with a RMSE of 24,105.55.
   - Predictions were made on the test dataset and saved in the Kaggle competition format.

   ```python
   predictions_df = pd.DataFrame({
       'Id': test['Id'],
       'SalePrice': tahminler.astype(int)
   })
   predictions_df.to_csv('predictions.csv', index=False)
   ```
   ### Achieved 1076th place out of 4611 participants in the Kaggle competition.
![Kagle-Score](https://github.com/user-attachments/assets/0a984a58-9109-4440-8af9-23ce090f822b)
