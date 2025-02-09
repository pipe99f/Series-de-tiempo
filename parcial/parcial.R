library(dplyr)



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
