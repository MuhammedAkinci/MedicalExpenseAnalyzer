import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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

# Veriyi eğitim ve test kümelerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirin
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Linear Regression modelini eğitin
linear_reg_model = LinearRegression()
linear_reg_scores = cross_val_score(linear_reg_model, X_train_scaled, y_train, cv=5)
linear_reg_mean_score = np.mean(linear_reg_scores)

# Random Forest Regressor modelini eğitin
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_scores = cross_val_score(random_forest_model, X_train_scaled, y_train, cv=5)
random_forest_mean_score = np.mean(random_forest_scores)

print("Linear Regression Ortalama R2 Skoru:", linear_reg_mean_score)
print("Random Forest Ortalama R2 Skoru:", random_forest_mean_score)

# Random Forest için hiper-parametre aralıklarını belirleyin
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search ile en iyi parametreleri bulun
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print("En iyi parametreler:", best_params)

# En iyi parametrelerle Random Forest modelini tekrar eğitin
optimized_random_forest_model = RandomForestRegressor(random_state=42, **best_params)
optimized_random_forest_model.fit(X_train_scaled, y_train)

# Test verisi üzerinde modeli değerlendirin
y_pred = optimized_random_forest_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Ortalama Kare Hata (MSE):", mse)
print("Ortalama Mutlak Hata (MAE):", mae)
