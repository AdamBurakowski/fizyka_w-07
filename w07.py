import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Parameters of sample
l = 20e-3  # Length in meters (20 mm)
A = 1e-5   # Crossectional area in square meters (10 mm²)

df = pd.read_csv('data.csv', skiprows=[0, 2], delimiter=';', decimal=',')

# Names for columns that actually make sense
df.columns = ['T_C', 'UH_mV', 'U_V', 'I_mA']

# Unit conversion
df['T_K'] = df['T_C'] + 273.15
df['I_A'] = df['I_mA'] / 1000
df['R'] = df['U_V'] / df['I_A']
df['sigma'] = (l * df['I_A']) / (A * df['U_V'])
df['ln_sigma'] = np.log(df['sigma'])
df['inv_T'] = 1 / df['T_K']

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(df['inv_T'], df['ln_sigma'])

print(f"Równanie prostej: {slope:.4f}x + {intercept:.4f}")

k = 8.625e-5  # eV/K
Eg = -2 * k * slope  # eV
u_Eg = 2 * k * std_err  # eV
print(f"Szerokość przerwy energetycznej: Eg = {Eg:.4f} eV ± {u_Eg:.4f} eV")

# Graphs
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.scatter(df['T_K'], df['sigma'], color='blue', marker='o')
ax1.set_ylabel('σ [S/m]')
# ax1.set_title('Zależność Przewodności σ i Rezystancji R od Temperatury')
ax1.grid(True)

ax2.scatter(df['T_K'], df['R'], color='green', marker='o')
ax2.set_xlabel('Temperatura [K]')
ax2.set_ylabel('R [Ω]')
ax2.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(df['inv_T'], df['ln_sigma'], color='lightblue', label='Dane pomiarowe')
plt.plot(df['inv_T'], intercept + slope * df['inv_T'], 'r--', label='Dopasowanie liniowe')
plt.xlabel('1/T [1/K]')
plt.ylabel('ln(σ) [ln(S/m)]')
# plt.title('Zależność ln(σ) od 1/T')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Calculating Uncertainty of Measurement

# Min and Max current
I_min = min(df['I_A'])  # Lowest value of Current in Ampers
I_max = max(df['I_A'])  # Highest value of Current in Ampers

df['min_sigma'] = (l * I_min) / (A * df['U_V'])
df['max_sigma'] = (l * I_max) / (A * df['U_V'])
df['ln_min_sigma'] = np.log(df['min_sigma'])
df['ln_max_sigma'] = np.log(df['max_sigma'])


# Linear regression
min_slope, min_intercept, *_ = linregress(df['inv_T'], df['ln_min_sigma'])
max_slope, max_intercept, *_ = linregress(df['inv_T'], df['ln_max_sigma'])

min_Eg = -2 * k * min_slope  # eV
max_Eg = -2 * k * max_slope  # eV


print(f"Szerokość przerwy energetycznej: Eg (min) = {min_Eg:.4f} eV, Eg (max) {max_Eg:.4f} eV")


plt.figure(figsize=(8, 5))
plt.scatter(df['inv_T'], df['ln_sigma'], color='lightblue', label='Dane pomiarowe')
plt.plot(df['inv_T'], intercept + slope * df['inv_T'], 'r--', label='Dopasowanie liniowe')
plt.plot(df['inv_T'], min_intercept + min_slope * df['inv_T'], 'g--', label=f'Dopasowanie liniowe dla I = {int(I_min * 1000)} mA')
plt.plot(df['inv_T'], max_intercept + max_slope * df['inv_T'], 'b--', label=f'Dopasowanie liniowe dla I = {int(I_max * 1000)} mA')
plt.xlabel('1/T [1/K]')
plt.ylabel('ln(σ) [ln(S/m)]')
# plt.title('Zależność ln(σ) od 1/T')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
