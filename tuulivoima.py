def read_file(file):
    data = []
    for line in file:
        line = line.strip('\n')
        parts = line.split(";")
        data.append(parts)
    return data


file = open("./tuulivoimapuistot.csv", "r")
wind_farms = read_file(file)
file.close()

file = open("./tuuliturbiinit.csv", "r")
wind_turbines = read_file(file)
file.close()

import matplotlib.pyplot as plt

powers = []
rotor_diameters = []
years = []
hub_heights = []

for i in range(1, len(wind_turbines)):
    power = wind_turbines[i][6]
    rotor_diameter = wind_turbines[i][11]
    hub_height = wind_turbines[i][10]
    year = wind_turbines[i][7]
    if power != "" and rotor_diameter != "" and hub_height != "":
        power = power.replace(",", ".")
        rotor_diameter = rotor_diameter.replace(",", ".")
        hub_height = hub_height.replace(",", ".")
        years.append(float(year))
        powers.append(float(power))
        hub_heights.append(float(hub_height))
        rotor_diameters.append(float(rotor_diameter))


    

import numpy as np

# Linear regression for power and diamter
X = rotor_diameters
y = powers


# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)

# Measure performance
import sklearn.metrics as sm

print( "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
print( "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) )
print( "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) )
print( "Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) )
print( "R2 score =", round(sm.r2_score(y_test, y_test_pred), 2) )

# Plot outputs
plt.scatter(rotor_diameters, powers)
plt.plot(X_test, y_test_pred, color='black', linewidth=1)
plt.xlabel("rotor diameter (m)")
plt.ylabel("power (kW)")
plt.show()
plt.show()


# Power / year

plt.scatter(years,powers)
plt.xlabel("year")
plt.ylabel("power (kW)")
plt.show()

# Rotor diameter / hub height

plt.scatter(rotor_diameters,hub_heights)
plt.xlabel("rotor diameter (m)")
plt.ylabel("hub height (m)")
plt.show()

# Wind farm data
years = []
powers = []
power_plants = []
data = {}

for i in range(1, len(wind_farms)):
    power = wind_farms[i][3]
    power_plant = wind_farms[i][2]
    year = wind_farms[i][1]
    if power != "" and year != "" and power_plant != "":
        power = power.replace(",", ".")
        year = float(year)
        power = float(power)
        power_plant = float(power_plant)
        years.append(year)
        powers.append(power)
        power_plants.append(power_plant)
        
        if year not in data:
            data[year] = power
        else:
            power = data[year] + power
            data[year] = power

# Power / year
plt.scatter(years,powers)
plt.xlabel("year")
plt.ylabel("power of wind farm (MW)")
plt.show()

# turbine amount / pwer
plt.scatter(power_plants,powers)
plt.xlabel("power plants")
plt.ylabel("power of wind farm (MW)")
plt.show()


years = []
powers = []
for year in data:
    years.append(year)
    powers.append(data[year])

# Power / year
plt.bar(years,powers)
plt.xlabel("year")
plt.ylabel("power of built wind farms (MW)")
plt.show()