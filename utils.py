#Implementing the Mean Square Error cost function (MSE)
def cost_f (X_train,y_train,w,b):
    cost = 0
    m = X_train.shape[0] 
    for i in range(m):
        model = X_train[i]*w+b
        cost += (model - y_train[i])**2
    cost = cost/(2*m)
    return cost

#Implementing Gradient Descent
def grad_desc (X_train,y_train,w,b,alpha,iterations):
    m = X_train.shape[0]
    for j in range(iterations):
        #Reset the partial derivatives to ensure each iteration uses a fresh gradient value for each parameter
        dj_dw = 0 
        dj_db = 0
        for i in range(m):
            #Compute the common factor in both gradients and hence use it to calculate them afterwards
            error = (w*X_train[i]+b - y_train[i]) 
            dj_dw += error * X_train[i]
            dj_db += error
        #Once what's within the summation has been computed, divide by the number of training examples
        dj_dw = dj_dw/m 
        dj_db = dj_db/m
        #Based on the calculated gradients adjust the model parameters
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        #Print the weight,bias and cost every 10% of the total iterations. 
        #ANSI escape codes are used to color the output for better readability in the terminal
        if j % (iterations // 10) == 0:
            cost_now = cost_f(X_train, y_train, w, b)
            print(f"\033[97mIteration {j:>6}:\033[0m \033[92mw = {w:.4f}, b = {b:.4f}, cost = {cost_now:.4f}\033[0m")
    return w,b #Return the updated model parameters to be used in the implementation process

#Define a function to obtain the prediction made by the model
def predict(X,w,b):
    return X*w+b

def z_norm (X_train):
    mean = 0
    sd = 0

    #Compute Mean
    for i in X_train:
        mean += i
    mean /= len(X_train)

    #Compute Standard Deviation
    for j in X_train:
        sd += (j-mean)**2
    sd /= len(X_train)
    sd = sd**0.5

    #Apply x=score normalization to each value
    X_norm = []
    for k in X_train:
        X_norm.append((k-mean)/sd)
    
    #Return the normalized feature values, the mean and the standard deviation for the dataset
    return X_norm, mean, sd