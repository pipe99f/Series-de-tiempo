import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller

# %%##########
## PUNTO 1
##############
# Series de Inflación de alimentos y precipitación en el aeropuerto ElDorado
inflacion = pd.read_excel("./data/inflacion_alimentos_mensual.xlsx", index_col=0,
                          names=["Fecha", "Valor"],
                          parse_dates=["Fecha"], date_format="%d/%m/%Y", decimal=",")
# noinspection PyTypeChecker
precipitacion = pd.read_excel("./data/total_precipitacion.xlsx", sheet_name="Bogotá",
                              index_col=0, names=["Fecha", "Valor"], usecols="B:C", skiprows=5,
                              nrows=46, parse_dates=["Fecha"], date_format="%Y")

#%% Gráfica Precipitación
sns.lineplot(data=precipitacion, legend=False)
plt.xlabel("Año")
plt.ylabel("Precipitación anual")
plt.show()

#%% Gráfica Inflación
sns.lineplot(data=inflacion, x="Fecha", y="Valor", legend=False)
plt.xlabel("Fecha")
plt.ylabel("Inflación mensual")
plt.show()

# %%##########
## PUNTO 2
##############

# MA de orden 5 y 10 para la serie de Inflación de Alimentos
inflacion["MA_5"] = inflacion["Valor"].rolling(5).mean()
sns.lineplot(data=inflacion, x="Fecha", y="MA_5", legend=False)
plt.ylabel("Inflación diaria")
plt.show()

inflacion["MA_8"] = inflacion["Valor"].rolling(8).mean()
sns.lineplot(data=inflacion, x="Fecha", y="MA_8", legend=False)
plt.ylabel("Inflación diaria")
plt.show()

# MA de orden 2 y 8 para la serie de Precipitación mensual
precipitacion["MA_2"] = precipitacion["Valor"].rolling(2).mean()
sns.lineplot(data=precipitacion, x="Fecha", y="MA_2", legend=False)
plt.ylabel("Precipitación mensual")
plt.show()

precipitacion["MA_8"] = precipitacion["Valor"].rolling(8).mean()
sns.lineplot(data=precipitacion, x="Fecha", y="MA_8", legend=False)
plt.ylabel("Precipitación mensual")
plt.show()

# %%##########
## PUNTO 3
##############
# Función para hacer las simulaciones, graficar y ajustar el modelo ARMA(1,1)
sns.set_style("darkgrid")


def simular_graph_analizar_arma(ar, ma, sigma=1, n=500):
    # Simular Arma
    arma_process = ArmaProcess(ar, ma)
    simul = arma_process.generate_sample(nsample=n, scale=sigma)

    # Gráfica simulación
    # plt.figure(figsize=(12, 6))
    plt.figure()
    plt.plot(simul)
    plt.title("Simulación ARMA")
    plt.show()

    # ACF
    plot_acf(simul, lags=20)
    plt.title("ACF")
    plt.show()

    # PACF
    plot_pacf(simul, lags=20)
    plt.title("PACF")
    plt.show()

    # Ajustar ARMA
    model = ARIMA(simul, order=(1, 0, 1))
    results = model.fit()
    print(results.summary())
    return


# %% Primer modelo ARMA: x_t = 0.9x_{t-1} + w_t - 0.9w_{t-1}
ar = [1, -0.9]
ma = [1, -0.9]
simular_graph_analizar_arma(ar, ma)


# %% Segundo modelo ARMA: X_t = 0.2X_{t-1} + 0.55X_{t-2} + w_t
ar = [1, -0.2, -0.55]
ma = [1]
simular_graph_analizar_arma(ar, ma, sigma=np.sqrt(2.25))

# Tercer modelo ARMA: X_t = w_t + 0.9w_{t-1} - 0.8w_{t-2} - 0.8w_{t-3}
ar = [1]
ma = [1, 0.9, -0.8, -0.8]
simular_graph_analizar_arma(ar, ma, sigma=np.sqrt(9))

# %%##########
## PUNTO 4
##############
data4 = pd.read_excel("./data/HW02-DatosPunto4.xls", sheet_name=0, index_col=0)

#%% Box-Jenkings para la serie 1
serie1 = data4["serie1"]
sns.lineplot(data=serie1)
plt.show()

#%% Box-Jenkings para la serie 2
serie2 = data4["serie2"]
sns.lineplot(data=serie2)
plt.show()

#%% Box-Jenkings para la serie 3
serie3 = data4["serie3"]
sns.lineplot(data=serie3)
plt.show()

#%% Box-Jenkings para la serie 4
serie4 = data4["serie4"]
sns.lineplot(data=serie4)
plt.show()

