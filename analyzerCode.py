import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
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

# Kategorik değişkenleri etiket kodlama ile dönüştürün
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])

# Veriyi bağımsız değişkenler (X) ve hedef değişken (y) olarak ayırın
X = data.drop(['charges', 'region'], axis=1)
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

# BMI değeri en yüksek ve en düşük olan bölgeler
max_bmi_region = data.groupby('region')['bmi'].mean().idxmax()
min_bmi_region = data.groupby('region')['bmi'].mean().idxmin()
print("BMI değeri en yüksek olan bölge:", max_bmi_region)
print("BMI değeri en düşük olan bölge:", min_bmi_region)

print("---------------------------------------------")

# En fazla çocuğa sahip bölgeyi bulun
max_children_region = data.groupby('region')['children'].sum().idxmax()
min_children_region = data.groupby('region')['children'].sum().idxmin()
print("En fazla çocuğa sahip bölge:", max_children_region)
print("En az çocuğa sahip bölge:", min_children_region)

print("---------------------------------------------")

# En fazla ve en az sigara içicisi olan bölgeler
smoker_counts = data[data['smoker'] == 1].groupby('region')['smoker'].count()
max_smoker_region = smoker_counts.idxmax()
min_smoker_region = smoker_counts.idxmin()
print("En fazla sigara içicisi bulunan bölge:", max_smoker_region)
print("En az sigara içicisi bulunan bölge:", min_smoker_region)

print("---------------------------------------------")

# En fazla ve en az kız (0) olan bölge
female_counts = data[data['sex'] == 0].groupby('region')['sex'].count()
max_female_region = female_counts.idxmax()
min_female_region = female_counts.idxmin()
print("En fazla kız olan bölge:", max_female_region)
print("En az kız olan bölge:", min_female_region)

print("---------------------------------------------")

# En fazla ve en az erkek (1) olan bölge
male_counts = data[data['sex'] == 1].groupby('region')['sex'].count()
max_male_region = male_counts.idxmax()
min_male_region = male_counts.idxmin()
print("En fazla erkek olan bölge:", max_male_region)
print("En az erkek olan bölge:", min_male_region)

print("---------------------------------------------")

# Ortalama Linear Regression ve Random Forest sonuçlarını yazdır 
print("Linear Regression Ortalama MSE:", np.mean(linear_mse_scores))
print("Linear Regression Ortalama MAE:", np.mean(linear_mae_scores))
print("Random Forest Ortalama MSE:", np.mean(random_forest_mse_scores))
print("Random Forest Ortalama MAE:", np.mean(random_forest_mae_scores))

print("---------------------------------------------")

# Kadın ve erkek için bölgelere göre ortalama sağlık sigorta maliyetlerini hesaplayın
mean_charges = data.groupby(['region', 'sex'])['charges'].mean().unstack()

# Sigorta maliyet sonuçlarını yazdır
print("Bölgelere göre erkekler ve kadınlar için ortalama sigorta maliyeti")
regions = ['northeast', 'northwest', 'southeast', 'southwest']
for region in regions:
    if region in mean_charges.index:
        print(f"{region.capitalize()}:")
        female_mean = mean_charges.loc[region, 0]  # Kadınların ortalaması
        male_mean = mean_charges.loc[region, 1]    # Erkeklerin ortalaması
        print(f"Kadın: ${female_mean:,.2f}")
        print(f"Erkek: ${male_mean:,.2f}")