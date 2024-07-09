import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# importing data 
cars_data=pd.read_csv("D:\Toyota.csv" ,index_col=0,na_values=["?","???"])

# Removing missing values from the data frame 
cars_data.dropna(axis=0,inplace=True)


                           # Scatter plot  price vs Age of the car
 
plt.scatter(cars_data['Age'], cars_data['Price'], c='red')    
plt.title('scatter plot of the price vs Age of the car')
plt.xlabel('Age(months)')
plt.ylabel('price(Euros)')
plt.show()     

"""  conclusion of above scatter >>The price of the car decreases as age of the car increases""" 


                          
                         # HISTOGRAM 
  
plt.hist(cars_data['KM'], 
         color='green', 
         edgecolor='white', 
         bins=5)

plt.title('Histogram of Kilometer')
plt.xlabel('Kilometer')
plt.ylabel('Frequency')

plt.show()

"""Conclusion >>  Frequency distribution of kilometre of the cars shows that most of 
the cars have travelled between 50000 – 100000 km and there are only few cars
 with more distance travelled"""


                                  # BAR PLOT 
                                  
# Data
counts = [979, 120, 12]
fuelType = ['Petrol', 'Diesel', 'CNG']
index = np.arange(len(fuelType))

# Plotting the bar chart
plt.bar(index, counts, color=['red', 'blue', 'cyan'])

# Adding title and labels
plt.title('Bar plot of fuel types')
plt.xlabel('Fuel Types')
plt.ylabel('Frequency')

# Adding custom tick labels
plt.xticks(index, fuelType, rotation=90)

# Show the plot
plt.show()  

""" Conclusion of Bar plot >>> Bar plot of fueal type shows that most of the cars have petrol
    as fueal type"""                                 
                                  


