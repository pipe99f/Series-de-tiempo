import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

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
data4 = pd.read_excel("./data/HW02-DatosPunto4.xls", sheet_name=0)
# print(data4)
