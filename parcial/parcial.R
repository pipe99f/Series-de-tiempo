library(dplyr)
library(forecast)
library(tseries)
library(TSA)
library(ggplot2)
library(hwwntest)
library(readxl)

####
## PUNTO 1
####
serie1 <- read_excel("P1.xlsx")
adver <- ts(serie1$Xt, start = c(1984, 1), frequency = 12)
plot(adver, xlab = "Año", ylab = "Gasto")

# Transformación de la serie (Estabilización de la varianza)
adver_ln <- log(adver)
plot(adver_ln, main = "Log de la Serie", xlab = "Año")
adf.test(adver_ln)

# Se obtiene estacionariedad
adver_dif <- diff(adver_ln, 1)
plot(adver_dif, main = "Diff de la serie")
adf.test(adver_dif)

# Identificación del modelo
acf(adver_dif)
pacf(adver_dif)
eacf(adver_dif) # Posible modelo ARMA(2, 4)

auto.arima(adver_dif, trace = T) # Posible modelo ARMA(2, 1)

adver_mod1 <- Arima(adver_dif, order = c(2, 0, 4), include.mean = F)
summary(adver_mod1)
adver_mod2 <- Arima(adver_dif, order = c(2, 0, 1), include.mean = F) 
summary(adver_mod2)  # Mejor modelo por criterio de AICc y BIC

# Diagnostico del modelo
checkresiduals(adver_mod2)
mod2_res <- adver_mod2$residuals

ks.test(mod2_res, "pnorm", mean(mod2_res), sd(mod2_res)) #Kolmogorov-Smirnov test
cpgram(mod2_res, main = "") # Periodograma
bartlettB.test(mod2_res)

####
## PUNTO 2
####

p2 <- read_excel("P2.xls")
desempleo <- ts(p2$DESEMP, start = c(1977, 1), frequency = 4)
plot(desempleo, type = "l")

## Estabilizar varianza
log_desempleo <- log(desempleo)

## Test ADF
adf.test(log_desempleo) # No se rechaza H0

log_desempleo_d1 <- diff(log_desempleo, 1) # Primera diferencia
adf.test(log_desempleo_d1) # Se rechaza H0, la serie es estacionaria en diferencias

## Determinar modelo ARMA adecuado
acf(log_desempleo_d1) # Hay estacionalidad cada año
pacf(log_desempleo_d1)
eacf(log_desempleo_d1) # ARMA(3,4) es un modelo tentativo
# auto.arima(log_desempleo_d1, trace = T)

# modelo_arima <- arima(log_desempleo_d1, order = c(0, 0, 1), seasonal = list(order = c(0, 1, 0), period = 4))
modelo_arima <- Arima(log_desempleo_d1, order = c(3, 0, 4))
plot(residuals(modelo_arima))
acf(residuals(modelo_arima))

tsdiag(modelo_arima) # Parece ser apropiado el modelo a pesar de la estacionalidad

## Pronóstico dentro de muestra
pronosticos_dentro <- fitted(modelo_arima)
rmse_dentro <- sqrt(mean((log_desempleo_d1 - pronosticos_dentro)^2))
print(paste("RMSE dentro de muestra:", rmse_dentro))

## Pronóstico a un año
pronostico_fuera <- forecast(modelo_arima, h = 4)
autoplot(pronostico_fuera)


####
## PUNTO 3
####
set.seed(123)

# Función para simular datos y estimar parámetros
simular_y_estimar <- function(n_simulaciones, n_obs, delta, phi, c) {
  resultados_delta <- numeric(n_simulaciones)
  resultados_phi_modelo1 <- numeric(n_simulaciones)
  resultados_c <- numeric(n_simulaciones)
  resultados_phi_modelo2 <- numeric(n_simulaciones)

  for (i in 1:n_simulaciones) {
    # Simular errores (epsilon_t)
    epsilon <- rnorm(n_obs, mean = 0, sd = 1)

    ###
    # Modelo 1: Y_t = delta*t + phi*Y_{t-1} + epsilon_t
    # Se calculan los valores Y_t a partir del valor incial Y_0 = 0
    Y1 <- numeric(n_obs)
    Y1[1] <- delta + epsilon[1]
    for (t in 2:n_obs) {
      Y1[t] <- delta * t + phi * Y1[t - 1] + epsilon[t]
    }

    # Se crea data frame con las columnas ajustadas para la regresión
    datos_modelo1 <- data.frame(
      Y_t = Y1[-1], # Y_t desde t=2 hasta t=n_obs
      t = 2:n_obs, # Variable de tiempo
      Y_lag = Y1[1:(n_obs - 1)] # Y_{t-1}
    )

    # Estimación por MCO
    modelo1 <- lm(Y_t ~ 0 + t + Y_lag, data = datos_modelo1)
    resultados_delta[i] <- coef(modelo1)["t"]
    resultados_phi_modelo1[i] <- coef(modelo1)["Y_lag"]


    ###
    # Modelo 2: Y_t = c + phi*Y_{t-1} + epsilon_t
    # Se calculan los valores Y_t a partir del valor incial Y_0 = 0
    Y2 <- numeric(n_obs)
    Y2[1] <- c + epsilon[1]
    for (t in 2:n_obs) {
      Y2[t] <- c + phi * Y2[t - 1] + epsilon[t]
    }

    # Se crea data frame con las columnas ajustadas para la regresión
    datos_modelo2 <- data.frame(
      Y_t = Y2[-1], # Y_t desde t=2 hasta t=n_obs
      Y_lag = Y2[1:(n_obs - 1)] # Y_{t-1}
    )

    # Estimación por MCO
    modelo2 <- lm(Y_t ~ 1 + Y_lag, data = datos_modelo2)
    resultados_c[i] <- coef(modelo2)["(Intercept)"]
    resultados_phi_modelo2[i] <- coef(modelo2)["Y_lag"]
  }

  return(list(
    delta_hat = resultados_delta,
    phi_hat_modelo1 = resultados_phi_modelo1,
    c_hat = resultados_c,
    phi_hat_modelo2 = resultados_phi_modelo2
  ))
}

# Simulación
delta <- 0.5
phi <- 0.7
c <- 1.0
n_obs <- 1000 # Tamaño de muestra
n_simulaciones <- 500

resultados <- simular_y_estimar(n_simulaciones, n_obs, delta, phi, c)


# Resultados
media_delta <- mean(resultados$delta_hat)
media_phi1 <- mean(resultados$phi_hat_modelo1)
media_c <- mean(resultados$c_hat)
media_phi2 <- mean(resultados$phi_hat_modelo2)

sesgo_delta <- mean(resultados$delta_hat) - delta
sesgo_phi1 <- mean(resultados$phi_hat_modelo1) - phi
sesgo_c <- mean(resultados$c_hat) - c
sesgo_phi2 <- mean(resultados$phi_hat_modelo2) - phi


cat("Estimación media de delta:", media_delta, "\n")
cat("Estimación media de phi (Modelo 1):", media_phi1, "\n")
cat("Estimación media de c:", media_c, "\n")
cat("Estimación media de phi (Modelo 2):", media_phi2, "\n")

cat("Sesgo del estimador de delta:", sesgo_delta, "\n")
cat("Sesgo del estimador de phi (Modelo 1):", sesgo_phi1, "\n")
cat("Sesgo del estimador de c:", sesgo_c, "\n")
cat("Sesgo del estimador de phi (Modelo 2):", sesgo_phi2, "\n")

# Graficar distribuciones de los estimadores
par(mfrow = c(2, 2))
hist(resultados$delta_hat, main = "Distribución de delta_hat", xlab = "", col = "skyblue")
abline(v = delta, col = "red", lwd = 2)
hist(resultados$phi_hat_modelo1, main = "Distribución de phi_hat (Modelo 1)", xlab = "", col = "skyblue")
abline(v = phi, col = "red", lwd = 2)
hist(resultados$c_hat, main = "Distribución de c_hat", xlab = "", col = "skyblue")
abline(v = c, col = "red", lwd = 2)
hist(resultados$phi_hat_modelo2, main = "Distribución de phi_hat (Modelo 2)", xlab = "", col = "skyblue")
abline(v = phi, col = "red", lwd = 2)


####
## PUNTO 4
####
kalman_filter <- function(y, phi, sigma_w, sigma_v) {
  n <- length(y)
  m <- 5
  x_est <- replicate(n, 0)
  P <- replicate(n, 0)
  K <- replicate(n, 0)
  x_est[1] <- mean(y[1:m])
  P[1] <- var(y[1:m])
  
  for (t in 2:n) {
    x_pred <- phi * x_est[t - 1]
    P_pred <- phi^2 * P[t - 1] + sigma_w^2
    K[t] <- P_pred / (P_pred + sigma_v^2)  
    x_est[t] <- x_pred + K[t] * (y[t] - x_pred)  
    P[t] <- (1 - K[t]) * P_pred
  }
  
  return(list(x_est = x_est, P = P, K = K))
}
# Se simula un AR(1)
x_t <- arima.sim(list(order = c(1,0,0), ar = 0.7), n = 1000)
# Se determinan los valores iniciales
y_t <- x_t 
acf_mod <- acf(y_t, lag.max = 2, plot = F)$acf
phi <- acf_mod[2] / acf_mod[1]
sigma_w <- (1 - phi**2)*var(y_t)/phi
# Se ajusta el modelo
ymode <- makeARIMA(phi, theta = 0, Delta = sigma_w) 
ymode$phi #Valor estimado de phi
ymode$Delta # Valor estimado de sigma^2
