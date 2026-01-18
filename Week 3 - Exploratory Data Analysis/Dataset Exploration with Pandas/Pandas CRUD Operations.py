""" C - Create 
    R - Read
    U - Update
    D - Delete """

""" 1. Create: Creating Dataframe """
#   Creating a dataset in Pandas means building a DataFrame which is the main data structure in Pandas. We can create a DataFrame using various methods like reading from a file or directly creating one from Python objects like dictionaries, lists or arrays

""" 1. Creating a DataFrame from a Dictationary """
#   This is one of the easiest and most commonly used methods to create a dataset in Pandas
import pandas as pd
data = {
    "Name": ["Ansh", "Sahil", "Ram"],
    "Age": [21, 20, 41],
    "City": ["Moradabad", "New Delhi", "Chennai"]
}
df = pd.DataFrame(data)
print(df)

""" 2. Creating a DataFrame from Lists """
#   We can also create a DataFrame by combining lists.

import pandas as pd

names = ["Akshit", "Uday", "Sam"]
ages = [25, 30, 35]
cities = ["Gurugram", "New Delhi", "Chicago"]

df = pd.DataFrame({
    "Name": names,
    "Age": ages,
    "City": cities
})

print(df)

""" 3. Creating a DataFrame from a CSV File """
#   We can also create a DataFrame by reading an external file like a CSV. Here we used the random car.csv data.

import pandas as pd

df = pd.read_csv("/content/CAR.csv")
print(df.head())

