#################################################################
##################### Problem Set #3 ############################
#################################################################
import pandas as pd
import pandasql
import csv
import urllib2
import os
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')

cwd = os.getcwd()
os.chdir('%s\\Intro Data Science\\Problem Set 3' %(cwd))

df = pd.read_csv('turnstile_data_master_with_weather.csv')

def entries_histogram(turnstile_weather):

    plt.figure()
    turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly'].hist(bins=50)
    turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly'].hist(bins=50)
    return plt

entries_histogram(df)

'''
Does entries data from the previous exercise seem normally distributed?

Ans: NO

Can we run Welch's T test on entries data? why or why not?

Ans: No, we can not as the assumption to perform T test is to have normally distributed data.

'''


def mann_whitney_plus_means(turnstile_weather):

    with_rain = turnstile_weather[turnstile_weather.rain==1]['ENTRIESn_hourly']
    with_rain_mean = with_rain.mean()
    without_rain = turnstile_weather[turnstile_weather.rain==0]['ENTRIESn_hourly']
    without_rain_mean = without_rain.mean()

    U, p = scipy.stats.mannwhitneyu(with_rain,without_rain)

    return with_rain_mean, without_rain_mean, U, p

'''
Is the distribution of the number of entries statistically different between rainy & non rainy days?

Ans: YES

Describe your results and the methods used:

Ans: The p-value calculated using Mann-Whitney method is 0.019 which is a small number. This shows that the
    Null hypothesis could be rejected and the distribution of number of entries between rainy and non rainy
    days are statistically different.

'''

def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()

    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_cost(features, values, theta):

    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):

    m = len(values)
    cost_history=[]

    for i in range(num_iterations):

        predicted_values = np.dot(features,theta)

        theta = theta-alpha/m*np.dot((predicted_values - values),features)

        cost = compute_cost(features,values,theta)
        cost_history.append(cost)

    return theta, pandas.Series(cost_history)

def predictions(dataframe):

    # Select Features (try different features!)
    features = dataframe[['rain', 'precipi', 'Hour', 'meantempi','mintempi','maxtempi']]

    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)

    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)

    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 75 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array,
                                                            values_array,
                                                            theta_gradient_descent,
                                                            alpha,
                                                            num_iterations)

    plot = plot_cost_history(alpha, cost_history)

    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions, plot


def plot_cost_history(alpha, cost_history):

   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

def plot_residuals(turnstile_weather, predictions):

    plt.figure()
    (turnstile_weather['ENTRIESn_hourly'] - predictions).hist(bins=200)
    return plt

def compute_r_squared(data, predictions):

    r_squared = 1 - ((predictions-data)**2).sum()/((data-data.mean())**2).sum()

    return r_squared

