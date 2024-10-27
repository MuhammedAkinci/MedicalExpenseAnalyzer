import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Veriyi yükleyin
data = pd.read_csv("insurance.csv")

# Bmi dağılımını inceleyin
sns.histplot(data['bmi'], bins=20, kde=True)
plt.show()

# Sigara içicisi ve maliyet arasındaki ilişkiyi inceleyin
sns.boxplot(x='smoker', y='charges', data=data)
plt.show()

# Sigara içicisi ve bölge arasındaki ilişkiyi inceleyin
sns.countplot(x='smoker', hue='region', data=data)
plt.show()

# Bmi ve cinsiyet arasındaki ilişkiyi inceleyin
sns.boxplot(x='sex', y='bmi', data=data)
plt.show()

# En fazla çocuğa sahip bölgeyi bulun
max_children_region = data.groupby('region')['children'].sum().idxmax()
print("En fazla çocuğa sahip bölge:", max_children_region)

# Kategorik değişkenleri etiket kodlama ile dönüştürün
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
data['region'] = label_encoder.fit_transform(data['region'])

# One-hot encoding ile kategorik değişkenleri dönüştürün
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Veriyi bağımsız değişkenler (X) ve hedef değişken (y) olarak ayırın
X = data.drop('charges', axis=1)
y = data['charges']

# Monte Carlo simülasyonu parametreleri
n_simulations = 100  # Simülasyon sayısı
linear_mse_scores = []
linear_mae_scores = []
random_forest_mse_scores = []
random_forest_mae_scores = []

for _ in range(n_simulations):
    # Veriyi eğitim ve test kümelerine ayırın
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Veriyi ölçeklendirin
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression modelini eğitin ve doğrulama puanlarını kaydedin
    linear_reg_model = LinearRegression()
    linear_reg_scores = cross_val_score(linear_reg_model, X_train_scaled, y_train, cv=5)
    linear_reg_model.fit(X_train_scaled, y_train)
    y_pred_linear = linear_reg_model.predict(X_test_scaled)
    linear_mse_scores.append(mean_squared_error(y_test, y_pred_linear))
    linear_mae_scores.append(mean_absolute_error(y_test, y_pred_linear))

    # Random Forest modelini eğitin ve doğrulama puanlarını kaydedin
    random_forest_model = RandomForestRegressor(random_state=42)
    random_forest_model.fit(X_train_scaled, y_train)
    y_pred_rf = random_forest_model.predict(X_test_scaled)
    random_forest_mse_scores.append(mean_squared_error(y_test, y_pred_rf))
    random_forest_mae_scores.append(mean_absolute_error(y_test, y_pred_rf))

# Ortalama sonuçları yazdır
print("Linear Regression Ortalama MSE:", np.mean(linear_mse_scores))
print("Linear Regression Ortalama MAE:", np.mean(linear_mae_scores))
print("Random Forest Ortalama MSE:", np.mean(random_forest_mse_scores))
print("Random Forest Ortalama MAE:", np.mean(random_forest_mae_scores))
