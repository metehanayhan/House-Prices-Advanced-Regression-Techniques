[EN]
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

[TR]

# House Prices Prediction

Bu proje, Ames, Iowa'daki evlerin satış fiyatlarını tahmin etmeyi amaçlamaktadır. Kaggle'dan alınan veri seti kullanılarak, Gradient Boosting ve Random Forest gibi ileri düzey regresyon algoritmalarıyla yüksek doğruluk oranına sahip bir model geliştirilmiştir. Proje, veri hazırlama, özellik mühendisliği ve model değerlendirme aşamalarını içerir. Model performansı, **Root Mean Squared Error (RMSE)** metriği ile değerlendirilmiştir.

---

### Kullanılan Kütüphaneler

- **pandas**: Veri işleme ve analiz
- **numpy**: Sayısal işlemler
- **matplotlib**: Veri görselleştirme
- **seaborn**: İstatistiksel veri görselleştirme
- **scikit-learn**: Makine öğrenimi algoritmaları ve araçları
- **xgboost**: Extreme Gradient Boosting
- **warnings**: Uyarı yönetimi

---

### Proje Adımları

1. **Veri Yükleme**
   - CSV dosyaları kullanılarak veri yüklendi:
     ```python
     test = pd.read_csv('test.csv')
     train = pd.read_csv('train.csv')
     df = pd.concat([train, test], axis=0, ignore_index=True)
     ```

2. **Eksik Verilerin Doldurulması ve Veri Dönüşümü**
   - Sayısal özelliklerde eksik değerleri doldurmak için **KNN Imputer** kullanıldı.
   - Çok fazla eksik veri içeren sütunlar kaldırıldı.
   - Kategorik veriler için mod, sayısal veriler için medyan doldurma stratejileri uygulandı.
   - **LabelEncoder** ve **One-Hot Encoding** ile kategorik veriler dönüştürüldü.

3. **Özellik Mühendisliği**
   - `AgeAtSale`, `RemodelAge` ve `TotalBathrooms` gibi yeni özellikler oluşturuldu.
   - Özellik dönüşümleri yapılarak, kategorik sütunlar için dummy değişkenler oluşturuldu.
   - Sıralı kategorik değişkenler sayısal değerlere dönüştürüldü.

4. **Modelleme**
   - Veri, eğitim ve test setlerine ayrıldı.
   - Linear Regression, Ridge, Lasso, ElasticNet, Gradient Boosting, XGBRegressor gibi farklı regresyon modelleri eğitildi ve değerlendirildi.
   - Performans metriklerine göre **Gradient Boosting** modeli seçildi.

5. **Sonuçlar**
   - Gradient Boosting modeli, yaklaşık **0.894 R-squared** skoruna ve **24,105.55 RMSE** değerine ulaştı.
   - Test veri seti üzerinde tahminler yapıldı ve Kaggle yarışması formatında kaydedildi:

   ```python
   predictions_df = pd.DataFrame({
       'Id': test['Id'],
       'SalePrice': tahminler.astype(int)
   })
   predictions_df.to_csv('predictions.csv', index=False)
   ```
   ### Kaggle Yarışması Sonucu: 4611 katılımcı arasında 1076. sıraya yerleşildi.
![Kagle-Score](https://github.com/user-attachments/assets/0a984a58-9109-4440-8af9-23ce090f822b)
