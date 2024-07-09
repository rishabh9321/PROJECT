
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# importing data 
cars_data=pd.read_csv("D:\Toyota.csv" ,index_col=0,na_values=["?","???"])

# Removing missing values from the data frame 
cars_data.dropna(axis=0,inplace=True)


                    # SCATTER PLOT
                    
 # Scatter plot of Price vs Age with default arguments 
 
sns.set(style="darkgrid")
sns.regplot(x=cars_data['Age'], y=cars_data['Price'],
             fit_reg=False)
                  
""" conclusion >>  Age og cars increases as price of cars decreases """

                
          
                #  SCATTER PLOT OF Price vs Age BY FuealType


# Scatter plot with Seaborn
sns.lmplot(x='Age', y='Price', data=cars_data, 
           fit_reg=False, hue='FuelType', 
           legend=True, palette="Set1")

plt.show()

""" Conclusion >> There are more cars that are of petrol fueal type and the 
    price is really higher for the cars which has diesel fuel type and the 
    price is comparatively lower for the cars which have CNG fueal type """
    
    
                   # BOX AND WHISKERS PLOT 
                   
# price of the cars for various fueal types 

sns.boxplot(x= cars_data['FuelType'], y= cars_data['Price'])                  
    
""" Conclusion>>> you look at the middle lines of the different fuel types 
    of the cars, the median price of the car is really high when the fuel type
    of the car is petrol And the median values is really low when the fuel type
    is of either diesel or cng . it is very evident that the on an average the
    petrol fuel type has the highest price among the cars from the data set 
    that we have and  also see the  the maximum price of the car is for the 
    diesel fuel type and the minimum value of the car is also for the diesel
    fuel type """
    
    
    
                  # GROUPED BOX AND WHISKERS PLOT
                  
 # grouped box and whiskers plot of Price vs FuealType and Automatic  

sns.boxplot(x='FuelType', y=cars_data['Price'],hue='Automatic',data=cars_data)
plt.show()             

""" Conclusion>>>>    Here whenever the cars fuel type is petrol and the gearbox 
    type is also automatic there are no cars that are available for the automatic
    gearbox when the fuel type is of either diesel or cng ."""
    
    
    

                  
                  
                  
                  
                  
    

