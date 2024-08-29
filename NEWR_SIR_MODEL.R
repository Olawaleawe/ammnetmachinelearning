# Load necessary libraries
install.packages("deSolve")
library(deSolve)
####rm(parameters), rm(list=ls()), dev.off(), Ctrl + Enter runs R, Ctrl + Alt + R - Runs the entire script. Ctrl + Alt + B/E
# Define the SIR model

sir_model <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    dS <- -beta * S * I
    dI <- beta * S * I - gamma * I
    dR <- gamma * I
    return(list(c(dS, dI, dR)))
  })
}

# Initial values
init <- c(S = 999, I = 1, R = 0)
parameters <- c(beta = 0.25, gamma = 0.1)
times <- seq(0, 60, by = 1)

# Solve the SIR model
output <- ode(y = init, times = times, func = sir_model, parms = parameters)
print(output)
output <- as.data.frame(output)


####ANSWERS#######
# Maximum number of infected individuals
max_infected <- max(output[, "I"])
print(max_infected)
max_infected_day <- which.max(output[, "I"])
print(max_infected_day)
# Number of recovered individuals by the end of 60 days
total_recovered <- output[nrow(output), "R"]
print(total_recovered)
# Percentage of population that remains susceptible
final_susceptible <- output[nrow(output), "S"]
susceptible_percentage <- (final_susceptible / 1000) * 100

cat("Maximum number of infected individuals:", max_infected, "\n")
cat("Day with the highest number of infections:", max_infected_day, "\n")
cat("Total number of recovered individuals by day 60:", total_recovered, "\n")
cat("Percentage of the population that remains susceptible after 60 days:", susceptible_percentage, "%\n")
## Define parameters
beta <- 0.25
gamma <- 0.1

# Calculate the basic reproductive rate
R0 <- beta / gamma
print(paste("The basic reproductive rate (R0) is:", R0))

#######BASIC REPRODUCTIVE RATE


# Plot the results
plot(output$time, output$S, type = "l", col = "blue", xlab = "Time", ylab = "Population", ylim = c(0, 1000))
lines(output$time, output$I, col = "red")
lines(output$time, output$R, col = "green")
legend("right", legend = c("Susceptible", "Infectious", "Recovered"), col = c("blue", "red", "green"), lty = 1)

#################INCREASE THE BETA(TRANSMISSION RATE)#######
# Initial values
init <- c(S = 999, I = 1, R = 0)

# Increase transmission rate
parameters <- c(beta = 0.6, gamma = 0.1)  # Increased beta

# Time sequence
times <- seq(0, 60, by = 1)

# Solve the model
output <- ode(y = init, times = times, func = sir_model, parms = parameters)
output <- as.data.frame(output)

# Plot the results
ggplot(data = output, aes(x = time)) +
  geom_line(aes(y = S, color = "Susceptible")) +
  geom_line(aes(y = I, color = "Infectious")) +
  geom_line(aes(y = R, color = "Recovered")) +
  labs(x = "Time", y = "Population", title = "SIR Model with Higher Transmission Rate") +
  scale_color_manual(values = c("blue", "red", "green")) +
  theme_minimal()
#####################################################################################

###By adjusting the initial values, parameters, and time sequence, you can explore how different conditions and assumptions affect the spread of the disease and shape the curves in the SIR model. These modifications can provide valuable insights into the dynamics of infectious diseases under various scenarios.

# Initial values with more infectious individuals
init <- c(S = 990, I = 10, R = 0)  # Increased initial I

# Default parameters
parameters <- c(beta = 0.3, gamma = 0.1)

# Time sequence
times <- seq(0, 60, by = 1)

# Solve the model
output <- ode(y = init, times = times, func = sir_model, parms = parameters)
output <- as.data.frame(output)

# Plot the results
ggplot(data = output, aes(x = time)) +
  geom_line(aes(y = S, color = "Susceptible")) +
  geom_line(aes(y = I, color = "Infectious")) +
  geom_line(aes(y = R, color = "Recovered")) +
  labs(x = "Time", y = "Population", title = "SIR Model with More Initial Infectious Individuals") +
  scale_color_manual(values = c("blue", "red", "green")) +
  theme_minimal()
###############################################################################################################
# Initial values
init <- c(S = 999, I = 1, R = 0)

# Increase recovery rate (shorter infectious period)
parameters <- c(beta = 0.3, gamma = 0.3)  # Increased gamma

# Time sequence
times <- seq(0, 60, by = 1)

# Solve the model
output <- ode(y = init, times = times, func = sir_model, parms = parameters)
output <- as.data.frame(output)

# Plot the results
ggplot(data = output, aes(x = time)) +
  geom_line(aes(y = S, color = "Susceptible")) +
  geom_line(aes(y = I, color = "Infectious")) +
  geom_line(aes(y = R, color = "Recovered")) +
  labs(x = "Time", y = "Population", title = "SIR Model with Shorter Infectious Period") +
  scale_color_manual(values = c("blue", "red", "green")) +
  theme_minimal()




#################################################################################
# Initial values
init <- c(S = 999, I = 1, R = 0)

# Default parameters
parameters <- c(beta = 0.3, gamma = 0.1)

# Longer time sequence
times <- seq(0, 300, by = 1)  # Extended time sequence

# Solve the model
output <- ode(y = init, times = times, func = sir_model, parms = parameters)
output <- as.data.frame(output)

# Plot the results
ggplot(data = output, aes(x = time)) +
  geom_line(aes(y = S, color = "Susceptible")) +
  geom_line(aes(y = I, color = "Infectious")) +
  geom_line(aes(y = R, color = "Recovered")) +
  labs(x = "Time", y = "Population", title = "SIR Model Over Extended Time Period") +
  scale_color_manual(values = c("blue", "red", "green")) +
  theme_minimal()


##################################combined##################
# Define the SIR model function
sir_model <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    dS <- -beta * S * I
    dI <- beta * S * I - gamma * I
    dR <- gamma * I
    return(list(c(dS, dI, dR)))
  })
}

# Time sequence
times <- seq(0, 160, by = 1)

# Define the different sets of parameters
param_sets <- list(
  list(beta = 0.3, gamma = 0.1, title = "Base Parameters (β = 0.3, γ = 0.1)"),
  list(beta = 0.5, gamma = 0.1, title = "Higher Transmission Rate (β = 0.5, γ = 0.1)"),
  list(beta = 0.3, gamma = 0.2, title = "Higher Recovery Rate (β = 0.3, γ = 0.2)"),
  list(beta = 0.5, gamma = 0.2, title = "Higher Transmission and Recovery Rate (β = 0.5, γ = 0.2)")
)

# Initial values
init <- c(S = 999, I = 1, R = 0)
# Set up plotting area
par(mfrow = c(2, 2))  # 2x2 grid for plots

# Loop through each parameter set
for (params in param_sets) {
  # Solve the ODEs with the current set of parameters
  output <- ode(y = init, times = times, func = sir_model, parms = params[1:2])
  
  # Plot the results
  plot(output[, "time"], output[, "S"], type = "l", col = "blue", ylim = c(0, max(output[, "S"])), ylab = "Number of People", xlab = "Time", main = params$title)
  lines(output[, "time"], output[, "I"], col = "red")
  lines(output[, "time"], output[, "R"], col = "green")
  legend("right", legend = c("Susceptible", "Infectious", "Recovered"), col = c("blue", "red", "green"), lty = 1)
}

dev.off()
##########################################################################################################
# Define the SEIR model

# Parameters including N

init <- c(S = 999, E = 1, I = 0, R = 0)
parameters <- c(beta = 0.3, sigma = 0.1, gamma = 0.1, N = 1000)
times <- seq(0, 160, by = 1)

# SEIR model uses N from parameters list

seir_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    dS <- -beta * S * I / N
    dE <- beta * S * I / N - sigma * E
    dI <- sigma * E - gamma * I
    dR <- gamma * I
    return(list(c(dS, dE, dI, dR)))
  })
}

# Run the SEIR model
output <- ode(y = init, times = times, func = seir_model, parms = parameters)


output <- as.data.frame(output)

# Plot the results
plot(output$time, output$S, type = "l", col = "blue", xlab = "Time", ylab = "Population", ylim = c(0, 1000))
lines(output$time, output$E, col = "orange")
lines(output$time, output$I, col = "red")
lines(output$time, output$R, col = "green")
legend("right", legend = c("Susceptible", "Exposed", "Infectious", "Recovered"), col = c("blue", "orange", "red", "green"), lty = 1)

