import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sigorta maliyeti model parametreleri (Yuvarlanmış Değerler)
age_coef = 242          # Yaşın etkisi
bmi_coef = 76           # BMI'nin etkisi
children_coef = 357     # Çocuk sayısının etkisi
sex_male_coef = -453    # Erkek olmanın etkisi
smoker_yes_coef = 15000  # Sigara içmenin etkisini düşürme
region_northwest_coef = -85  # Kuzeybatı bölgesinin etkisi
region_norteast_coef = -245 # Kuzeydoğu bölgesinin etkisi
region_southeast_coef = -953 # Güneydoğu bölgesinin etkisi
region_southwest_coef = -1361 # Güneybatı bölgesinin etkisi

# Simülasyon parametreleri
simulations_list = [1, 10, 100, 1000, 10000, 100000]  # Simülasyon sayıları
avg_costs = []  # Ortalama sigorta maliyetlerini saklamak için liste
min_costs = []  # Minimum sigorta maliyetlerini saklamak için liste
max_costs = []  # Maksimum sigorta maliyetlerini saklamak için liste
std_devs = []

# Fonksiyon: Sigorta maliyeti simülasyonu
def insurance_simulation(num_simulations):
    insurance_costs = []
    
    for _ in range(num_simulations):
        # Gerçekçi parametreler
        smoker = 0.12

        bmi = np.random.normal(25, 4)    
        bmi = max(18, min(bmi, 40))  # BMI'nin 18 ile 40 arasında kalmasını sağlamak
        
        age = random.randint(18, 64)    # Gerçekçi yaş aralığı
        children = random.randint(0, 4) # Çocuk sayısı, 0 ile 4 arasında
        region = random.choice(['northwest', 'southeast', 'southwest'])  # Bölge seçimi

        # Sigorta maliyetinin hesaplanması
        insurance_cost = (age_coef * age + 
                          bmi_coef * bmi + 
                          children_coef * children + 
                          sex_male_coef * (1 if random.choice(['male', 'female']) == 'male' else 0) + 
                          smoker_yes_coef * smoker +  # Sigara içme etkisini azaltmak
                          region_northwest_coef * (1 if region == 'northwest' else 0) +
                          region_norteast_coef * (1 if region == 'northeast' else 0) +
                          region_southeast_coef * (1 if region == 'southeast' else 0) +
                          region_southwest_coef * (1 if region == 'southwest' else 0)
                          )
        
        insurance_costs.append(insurance_cost)
    
    # Sonuçların analizi
    average_cost = sum(insurance_costs) / num_simulations
    min_cost = min(insurance_costs)
    max_cost = max(insurance_costs)
    
    return average_cost, min_cost, max_cost, insurance_costs

all_costs = []

# Her bir simülasyon sayısı için ortalama, minimum ve maksimum sigorta maliyetlerini hesapla
for simulations in simulations_list:
    avg_cost, min_cost, max_cost, costs = insurance_simulation(simulations)
    avg_costs.append(avg_cost)
    min_costs.append(min_cost)
    max_costs.append(max_cost)
    all_costs.append(costs)

# Sigorta maliyetlerinin gerçek veriye dayalı analizi
df = pd.read_csv("insurance.csv")
real_avg_cost = df["charges"].mean()
real_avg_min = df["charges"].min()
real_avg_max = df["charges"].max()

# Plotlama
plt.plot(simulations_list, avg_costs, marker='o', label="Simülasyonla Hesaplanan Ortalama Maliyet")
plt.axhline(y=real_avg_cost, color='r', linestyle='--', label="Gerçek Ortalama Maliyet")
plt.xscale('log')  # X eksenini logaritmik skala yap
plt.xlabel('Simülasyon Sayısı (Log Skala)')
plt.ylabel('Ortalama Sigorta Maliyeti')
plt.title('Farklı Simülasyon Sayılarıyla Sigorta Maliyeti Hesaplama')
plt.legend()
plt.show()

for i in range(6):
    plt.figure(figsize=(10, 6))
    plt.hist(all_costs[i], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Sigorta Maliyeti')
    plt.ylabel('Frekans')
    plt.title(f"{simulations_list[i]} Simülasyondan Elde Edilen Sigorta Maliyetlerinin Dağılımı")
    plt.show()

# Simülasyonlar için sonuçları yazdırma
for simulations, avg_cost, min_cost, max_cost in zip(simulations_list, avg_costs, min_costs, max_costs):
    print(f"\nSimülasyon Sayısı: {simulations}")
    print(f"Ortalama Sigorta Maliyeti: {avg_cost}")
    print(f"En Düşük Sigorta Maliyeti: {min_cost}")
    print(f"En Yüksek Sigorta Maliyeti: {max_cost}")

# Gerçek sigorta maliyeti için ekrana yazdırma
print(f"\nGerçek Ortalama Sigorta Maliyeti: {real_avg_cost}")
print(f"Gerçek Minimum Sigorta Maliyeti: {real_avg_min}")
print(f"Gerçek Maksimum Sigorta Maliyeti: {real_avg_max}")

# Bölgelere göre ortalama maliyetlerin hesaplanması
regions = ['northwest', 'northeast', 'southeast', 'southwest']
region_avg_costs = []

for region in regions:
    region_costs = [
        age_coef * random.randint(18, 64) +
        bmi_coef * max(18, min(np.random.normal(25, 4), 40)) +
        children_coef * random.randint(0, 4) +
        sex_male_coef * (1 if random.choice(['male', 'female']) == 'male' else 0) +
        smoker_yes_coef * 0.12 +
        region_northwest_coef * (1 if region == 'northwest' else 0) +
        region_norteast_coef * (1 if region == 'northeast' else 0) +
        region_southeast_coef * (1 if region == 'southeast' else 0) +
        region_southwest_coef * (1 if region == 'southwest' else 0)
        for _ in range(10000)
    ]
    region_avg_costs.append(np.mean(region_costs))

# Barplot oluşturma
plt.figure(figsize=(8, 6))
plt.bar(regions, region_avg_costs, color=['lightblue', 'lightgreen', 'salmon', 'gold'], edgecolor='black')
plt.xlabel('Bölge')
plt.ylabel('Ortalama Sigorta Maliyeti')
plt.title('Bölgelere Göre Ortalama Sigorta Maliyetleri')
plt.show()

print("\n")

# Bölge bazlı ortalamaları ekrana yazdırma
for region, avg_cost in zip(regions, region_avg_costs):
    print(f"{region.capitalize()} Bölgesi Ortalama Sigorta Maliyeti: {avg_cost:.2f}")