import pickle
import pandas as pd 

kmeans_file = 'kmeans.bin'
model_file = 'model.bin'

# Load the k-means algorithm from disk
with open(kmeans_file, 'rb') as f_in_kmeans:
    kmeans = pickle.load(f_in_kmeans)

# Load the model from disk    
with open(model_file, 'rb') as f_in_model:
    model = pickle.load(f_in_model)

sample_data = {'own_car': 0,
 'own_realty': 1,
 'num_child': 1,
 'income': 11.630717389475995,
 'age': 30,
 'years_of_employment': 4,
 'mobile': 1,
 'work_phone': 0,
 'phone': 0,
 'email': 0,
 'num_family_members': 2,
 'is_female': 1,
 'is_male': 0,
 'Commercial associate': 0,
 'Pensioner': 0,
 'State servant': 0,
 'Student': 0,
 'Working': 1,
 'Academic degree': 0,
 'Higher education': 0,
 'Incomplete higher': 0,
 'Lower secondary': 0,
 'Secondary / secondary special': 1,
 'Civil marriage': 0,
 'Married': 0,
 'Separated': 0,
 'Single / not married': 1,
 'Widow': 0,
 'Co-op apartment': 0,
 'House / apartment': 1,
 'Municipal apartment': 0,
 'Office apartment': 0,
 'Rented apartment': 0,
 'With parents': 0,
 'Government': 0,
 'Sales': 0,
 'Unemployed': 0,
 'Unknown': 1,
 'has_previous_credit': 0,
 'count_X': 0,
 'count_C': 0,
 'months_late': 0}

def predict(customer):
    # Pandas will throw back an error if it does not fit the data
    # Following the sample data works just fine :) 

    customer = pd.DataFrame(customer, index = [0])

    # First round of predictions with k-means clustering
    kmeans_prediction = kmeans.predict(customer)
    kmeans_outcome = 0 if kmeans_prediction in [1,10,11,14] else 1 

    # Second round of predictions with the model
    prediction = model.predict(customer)

    prediction_string = "is not approved" if prediction == [0] else "is approved"
    print(f"The credit card {prediction_string}!")
    print(f"The kmeans algorithm suggested that this is {prediction[0] == kmeans_outcome}")
    return prediction

predict(sample_data)