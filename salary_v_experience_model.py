import numpy as np
import matplotlib.pyplot as plt
import time

#The following are stylistic adjusments and do not affect the performance of the model. 
#Do comment these out if it isn't to your liking.
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#181C14',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor': '#697565',
    'axes.labelcolor': '#CFC4B3',
    'xtick.color': '#CFC4B3',
    'ytick.color': '#CFC4B3',
    'text.color': '#ECDFCC',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.family': 'serif'
})

from utils import cost_f, grad_desc, predict, z_norm


data = np.loadtxt("salary_data.csv", delimiter=",", skiprows=1) #Import the dataset 

X_train = data[:,0] #Extract the first column ie features as a 1-D array
y_train = data[:,1] #Extract the second column ie targets as a 1-D array

#Uncomment the following if you wish to see the dataset
# print(f"Features : {X_train},\nTargets:{y_train}")

# #Normalizing the feature values using implemented z-score normalization and calculating the mean and the standard deviation of the dataset
# X_norm,mean,sd = z_norm(X_train)
# #Normalization (z-score) was implemented solely for educational purposes but was not implemented here,
# #as the dataset contains only a single feature that is already well-scaled.
# #For this specific case of a univariate linear regression, normalization offers no significant benefit and may reduce interpretability.
# #Uncomment this entire block to apply normalization and adjust the code accordingly if experimenting with convergence behavior.

w_init, b_init = 0,0 #Initialize the model parameters as w=0, b=0
alpha = 1e-3 #Set the learning rate to a lower value to begin with
iterations = 100000 #Set the number of steps desired to run gradient descent

cost = cost_f(X_train,y_train,w_init,b_init)
cost1 = cost

start_time = time.time()

w,b=grad_desc(X_train,y_train,w_init,b_init,alpha,iterations) #Get the model parameters after running gradient descent

end_time = time.time()

print(f"\033[96mTraining time: {end_time - start_time:.4f} seconds\033[0m")

#Assign the model prediction to a new variable
y_pred = predict(X_train,w,b)

print(f"\033[96mThe cost before gradient descent: {cost1}\033[0m")
cost = cost_f(X_train,y_train,w,b)
print(f"\033[96mThe cost after gradient descent: {cost}\033[0m")

#Plotting the graph and comparing against the dataset
plt.scatter(X_train, y_train, color='#CFC4B3', label='Raw')
plt.plot(X_train, y_pred, color='#7FC97F', label='Prediction')  # Olive green
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Univariate Linear Regression: Salary vs Experience")
plt.legend()
plt.show()