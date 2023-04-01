import numpy as np 
import pickle
import pandas as pd 

kmeans_file = 'kmeans.bin'
model_file = 'model.bin'

# Load the k-means algorithm from disk
with open(kmeans_file, 'rb') as f_in:
    kmeans = pickle.load(f_in)

# Load the model from disk    
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

sample_data = {}

def predict(customer):
    # If we do not use data from the application_with_clusters dataset
    # Then we have to check if they have the same columns 
    customer_columns = ['own_car', 'own_realty', 'num_child', 'income', 'age',
       'years_of_employment', 'mobile', 'work_phone', 'phone', 'email',
       'num_family_numbers', 'is_female', 'is_male', 'Commercial associate',
       'Pensioner', 'State servant', 'Student', 'Working', 'Academic degree',
       'Higher education', 'Incomplete higher', 'Lower secondary',
       'Secondary / secondary special', 'Civil marriage', 'Married',
       'Separated', 'Single / not married', 'Widow', 'Co-op apartment',
       'House / apartment', 'Municipal apartment', 'Office apartment',
       'Rented apartment', 'With parents', 'Government', 'Sales', 'Unemployed',
       'Unknown', 'has_previous_credit', 'count_X', 'count_C', 'months_late',
       'clusters']

    for column in customer.columns:
        if column not in customer_columns:
            return "Error: The column {} is not in the model. Please reformat the data to sample data".format(column)

    # First round of predictions with k-means clustering
    kmeans_prediction = kmeans.predict(customer)
    customer['outcome'] = 0 if kmeans_prediction in [1,10,11,14] else 1 

    # Second round of predictions with the model
    prediction = model.predict(customer)
    print(f"The prediction is {prediction}!")
    return prediction

predict(sample_data)