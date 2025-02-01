import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller

# %%##########
## PUNTO 1
##############
# Series de Inflación de alimentos y precipitación en el aeropuerto ElDorado
inflacion = pd.read_excel(
    "./data/inflacion_alimentos_mensual.xlsx",
    index_col=0,
    names=["Fecha", "Valor"],
    parse_dates=["Fecha"],
    date_format="%d/%m/%Y",
    decimal=",",
)
# noinspection PyTypeChecker
precipitacion = pd.read_excel(
    "./data/total_precipitacion.xlsx",
    sheet_name="Bogotá",
    index_col=0,
    names=["Fecha", "Valor"],
    usecols="B:C",
    skiprows=5,
    nrows=46,
    parse_dates=["Fecha"],
    date_format="%Y",
)

# %% Gráfica Precipitación
sns.lineplot(data=precipitacion, legend=False)
plt.xlabel("Año")
plt.ylabel("Precipitación anual")
plt.show()

# %% Gráfica Inflación
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


## Identificacion
def identificacion(serie, max_diff=10):
    """
    Diferenciar la serie hasta que la prueba ADF rechace la hipótesis nula de no estacionariedad.
    """
    # Estabilizar serie
    valor_minimo = np.min(serie)

    if valor_minimo < 0:
        shift = abs(valor_minimo) + 1  # Add a small constant to ensure positivity
        shifted_series = serie + shift
    else:
        shifted_series = serie

    serie_estabilizada = np.log(shifted_series)

    diff_serie = serie_estabilizada.copy()
    diff_order = 0

    while diff_order <= max_diff:
        # Realizar la prueba ADF
        resultado_adf = adfuller(diff_serie.dropna())
        p_valor = resultado_adf[1]

        # Si p-valor < 0.05, la serie es estacionaria
        if p_valor < 0.05:
            print(f"La serie es estacionaria después de {diff_order} diferenciaciones.")
            print(f"Estadístico ADF: {resultado_adf[0]}, p-valor: {p_valor}")
            return diff_serie, diff_order

        # Si no es estacionaria, diferenciar la serie
        diff_serie = diff_serie.diff().dropna()
        diff_order += 1
    # print(f"No se logró estacionariedad después de {max_diff} diferenciaciones.")

    return diff_serie, valor_minimo


def graficar_acf_pacf(serie):
    # Gráfico de ACF
    plt.figure()
    plt.subplot(1, 2, 1)
    plot_acf(serie, lags=20, ax=plt.gca())
    plt.title("ACF")

    # Gráfico de PACF
    plt.subplot(1, 2, 2)
    plot_pacf(serie, lags=20, ax=plt.gca())
    plt.title("PACF")
    plt.show()


# Estimación y diagnóstico


def ajustar_modelo_arima(serie, orden):
    """
    Ajusta un modelo ARIMA a la serie de tiempo.
    """
    modelo = ARIMA(serie, order=orden)
    resultados = modelo.fit()
    return resultados


def seleccionar_mejor_modelo(serie, ordenes):
    """
    Selecciona el mejor modelo ARIMA basado en el criterio AIC.
    """
    mejor_aic = float("inf")
    mejor_modelo = None
    mejor_orden = None

    for orden in ordenes:
        try:
            resultados = ajustar_modelo_arima(serie, orden)
            aic = resultados.aic
            if aic < mejor_aic:
                mejor_aic = aic
                mejor_modelo = resultados
                mejor_orden = orden
        except:
            continue

    print(f"Mejor orden ARIMA: {mejor_orden}, AIC: {mejor_aic}")
    return mejor_modelo, mejor_orden


def prueba_ljung_box(residuales, lags=10):
    """
    Realiza la prueba de Ljung-Box para verificar si los residuales son ruido blanco.
    """
    resultado = acorr_ljungbox(residuales, lags=lags)
    print("Resultado de la prueba de Ljung-Box:")
    print(resultado)

    # Verificar si los p-valores son mayores que 0.05
    if all(resultado.iloc[:, 1] > 0.05):
        print("Los residuales son ruido blanco (no hay autocorrelación significativa).")
    else:
        print("Los residuales NO son ruido blanco (hay autocorrelación significativa).")


# Ahora creamos una función que presente los resultados de las funciones anteriores


def estimacion_diagnostico(serie, ordenes):
    mejor_modelo, mejor_orden = seleccionar_mejor_modelo(serie, ordenes)
    residuales = mejor_modelo.resid
    prueba_ljung_box(residuales)
    return residuales


# %% Serie 1
data4 = pd.read_excel(
    "./data/HW02-DatosPunto4.xls",
    sheet_name=0,
)

serie1 = data4["serie1"]
date_index = pd.date_range(start="2024-01-01", periods=300, freq="D")
serie1.index = date_index


# Graficar la serie 1
plt.figure()
sns.lineplot(data=serie1)
plt.show()

# %% Preparar la serie
serie1_identificacion, valor_minimo = identificacion(serie1)
graficar_acf_pacf(serie1_identificacion)
# ARIMA (3,0,3) es un buen candidato

# %% Estimar modelo
# La estimación con auto_arima es (2,0,2)
auto_model = auto_arima(
    serie1_identificacion, seasonal=False, stepwise=True, trace=True
)

# Comparamos los dos modelos (3,0,3) y (2,0,2)
ordenes_arima_serie_1 = [(3, 0, 3), (2, 0, 2)]
residuales_serie_1 = estimacion_diagnostico(
    serie1_identificacion, ordenes_arima_serie_1
)

# Graficamos los residuales de (2, 0, 2)
modelo_serie_1 = ARIMA(serie1_identificacion, order=(2, 0, 2), trend="n")
modelo_serie_1 = modelo_serie_1.fit()

modelo_serie_1.plot_diagnostics(figsize=(16, 8))
plt.show()


# %% Pronóstico

forecast = modelo_serie_1.get_forecast(steps=20)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(serie1_identificacion, color="blue")
forecast_index = pd.date_range(
    start=serie1_identificacion.index[-1], periods=20 + 1, freq="D"
)[1:]
plt.plot(forecast_index, forecast_mean, label="Pronóstico", color="red")
plt.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()


# %% Serie 2
serie2 = data4["serie2"]
date_index = pd.date_range(start="2024-01-01", periods=300, freq="D")
serie2.index = date_index


# Graficar la serie
plt.figure()
sns.lineplot(data=serie2)
plt.show()

# %% Preparar la serie
serie2_identificacion, valor_minimo = identificacion(serie2)
graficar_acf_pacf(serie2_identificacion)
# ARIMA (4,0,4) es un buen candidato

# %% Estimar modelo
# La estimación con auto_arima es (0,0,2)
auto_model = auto_arima(
    serie2_identificacion, seasonal=False, stepwise=True, trace=True
)

# %%
# Comparamos los dos modelos (4,0,4) y (0,0,2)
ordenes_arima_serie_2 = [(4, 0, 4), (0, 0, 2)]
residuales_serie_2 = estimacion_diagnostico(
    serie2_identificacion, ordenes_arima_serie_2
)

# Graficamos los residuales de (0, 0, 2)
modelo_serie_2 = ARIMA(serie2_identificacion, order=(0, 0, 2), trend="n")
modelo_serie_2 = modelo_serie_2.fit()

modelo_serie_2.plot_diagnostics(figsize=(16, 8))
plt.show()


# %% Pronóstico
forecast = modelo_serie_2.get_forecast(steps=20)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(serie2_identificacion, color="blue")
forecast_index = pd.date_range(
    start=serie2_identificacion.index[-1], periods=20 + 1, freq="D"
)[1:]
plt.plot(forecast_index, forecast_mean, label="Pronóstico", color="red")
plt.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()


# %% Serie 3
serie3 = data4["serie3"]
date_index = pd.date_range(start="2024-01-01", periods=300, freq="D")
serie3.index = date_index


# Graficar la serie
plt.figure()
sns.lineplot(data=serie3)
plt.show()

# %% Preparar la serie
serie3_identificacion, valor_minimo = identificacion(serie3)
graficar_acf_pacf(serie3_identificacion)
# ARIMA (3,0,4) es un buen candidato

# %% Estimar modelo
# La estimación con auto_arima es (1,0,1)
auto_model = auto_arima(
    serie3_identificacion, seasonal=False, stepwise=True, trace=True
)

# %%
# Comparamos los dos modelos (3,0,4) y (1,0,1)
ordenes_arima_serie_3 = [(3, 0, 4), (1, 0, 1)]
residuales_serie_3 = estimacion_diagnostico(
    serie3_identificacion, ordenes_arima_serie_3
)

# Graficamos los residuales de (1, 0, 1)
modelo_serie_3 = ARIMA(serie3_identificacion, order=(1, 0, 1), trend="n")
modelo_serie_3 = modelo_serie_3.fit()

modelo_serie_3.plot_diagnostics(figsize=(16, 8))
plt.show()


# %% Pronóstico
forecast = modelo_serie_3.get_forecast(steps=20)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(serie3_identificacion, color="blue")
forecast_index = pd.date_range(
    start=serie3_identificacion.index[-1], periods=20 + 1, freq="D"
)[1:]
plt.plot(forecast_index, forecast_mean, label="Pronóstico", color="red")
plt.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# %% Serie 4
serie4 = data4["serie4"]
date_index = pd.date_range(start="2024-01-01", periods=300, freq="D")
serie4.index = date_index


# Graficar la serie
plt.figure()
sns.lineplot(data=serie4)
plt.show()

# %% Preparar la serie
serie4_identificacion, valor_minimo = identificacion(serie4)
graficar_acf_pacf(serie4_identificacion)
# ARIMA (5,0,3) es un buen candidato

# %% Estimar modelo
# La estimación con auto_arima es (1,0,1)
auto_model = auto_arima(
    serie4_identificacion, seasonal=False, stepwise=True, trace=True
)

# %%
# Comparamos los dos modelos (5,0,3) y (2,0,0)
ordenes_arima_serie_4 = [(5, 0, 3), (2, 0, 0)]
residuales_serie_4 = estimacion_diagnostico(
    serie4_identificacion, ordenes_arima_serie_4
)

# Graficamos los residuales de (1, 0, 1)
modelo_serie_4 = ARIMA(serie4_identificacion, order=(2, 0, 0), trend="n")
modelo_serie_4 = modelo_serie_4.fit()

modelo_serie_4.plot_diagnostics(figsize=(16, 8))
plt.show()


# %% Pronóstico
forecast = modelo_serie_4.get_forecast(steps=20)
forecast_mean = forecast.predicted_mean
forecast_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(serie4_identificacion, color="blue")
forecast_index = pd.date_range(
    start=serie4_identificacion.index[-1], periods=20 + 1, freq="D"
)[1:]
plt.plot(forecast_index, forecast_mean, label="Pronóstico", color="red")
plt.fill_between(
    forecast_index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
