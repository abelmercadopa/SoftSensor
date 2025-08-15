import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

# Cargar el archivo CSV
dt = pd.read_csv('wic318_.csv')

# Ajustar el modelo de regresión lineal múltiple
mod = smf.ols('WIT318 ~FE005+FE006+FE007+FE008+SIT312B2+LIT447+CV002MO1+CV001MO1I', data=dt).fit()
 

# Imprimir el resumen del modelo
print(mod.summary())

# Predicciones del modelo
dt['Predicted'] = mod.predict(dt)

# Visualización de la significancia de las variables

# Extraer los p-valores y coeficientes del modelo
p_values = mod.pvalues.drop('Intercept')  # Excluye el intercepto para el análisis
coef_values = mod.params.drop('Intercept')

# Normalizar los p-valores para representar intensidad de color (entre 0 y 1)
intensity = 1 - p_values / p_values.max()

# Crear una figura y ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Crear un gráfico de barras para visualizar los coeficientes con color basado en la significancia
bars = ax.bar(p_values.index, coef_values, color=plt.cm.coolwarm(intensity))

# Añadir los valores de los coeficientes encima de las barras
for bar, coef in zip(bars, coef_values):
    height = bar.get_height()
    ax.annotate(f'{coef:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Etiquetas y título
ax.set_xlabel('Variables')
ax.set_ylabel('Coeficiente')
ax.set_title('Significancia de las Variables)')
ax.set_xticklabels(p_values.index, rotation=45, ha='right')

# Mostrar el gráfico
plt.show()

# Gráfico de comparación de salida real vs salida aproximada

# Variables para graficar
X = dt['Timestamp']
Y1 = dt['WIT318']
Y2 = dt['Predicted']

# Configuración del gráfico
plt.figure(figsize=(15,8))
plt.plot(X, Y1, color='blue', linestyle='-', label='Salida Real')
plt.plot(X, Y2, color='red', linestyle='--', label='Salida Aproximada')

# Añadir etiquetas y título
plt.xlabel('Timestamp')
plt.ylabel('WIT318')
plt.title('Señal Real Vs Aproximacion')
plt.legend()

# Mostrar el gráfico
plt.show()
