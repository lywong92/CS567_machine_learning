import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    y = np.where(y==0, -1, y)
    X = np.insert(X, 0, 1, axis=1)
    
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        
        w = np.zeros(D+1)
        b = 0
        
        # calculate perceptron loss gradient
        # gradient = sum(-I(ynwTXn<=0))*ynXn) for all n
        # only misclassified examples (ynwTXn<=0) contribute to gradient
        # update w with w+step_size*avergae gradient
        for i in range(max_iterations):          
            XwT = np.matmul(X, np.transpose(w))
            YXwT = np.multiply(y, XwT)
            matches = np.where(YXwT > 0)
            y_loss = np.delete(y, matches)
            X_loss = np.delete(X, matches, axis=0)
            loss_sum = np.matmul(np.transpose(X_loss), np.transpose(y_loss))
            
            loss_avg = np.divide(loss_sum, N)
            step = np.multiply(step_size, loss_avg)
            w = np.add(w, step)
               
        b = w[0]
        w = w[1:]
        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        
        w = np.zeros(D+1)
        b = 0
        
        # calculate logistic loss gradient
        # gradient = sum(sigmoid(-ynwTXn)*ynXn) for all n
        # update w with w+step_size*avergae gradient
        for i in range(max_iterations):
            XwT = np.matmul(X, np.transpose(w))
            YXwT = np.multiply(y, XwT)
            loss_prob = sigmoid(np.multiply(-1, YXwT))
            loss_sum = np.multiply(loss_prob, y)
            loss_sum = np.matmul(np.transpose(X), np.transpose(loss_sum))
            
            loss_avg = np.divide(loss_sum, N)
            step = np.multiply(step_size, loss_avg)
            w = np.add(w, step)
                
        b = w[0]
        w = w[1:]
        
        ############################################

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    
    value = np.exp(-z)
    value = value + 1
    value = np.power(value, -1)
    
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b)
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        
        preds = np.zeros(N)
        y = np.matmul(X, np.transpose(w))
        
        # predict based on sign
        for i in range(len(y)):
            preds[i] = 1 if y[i] > 0 else 0
            
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        
        preds = np.zeros(N)        
        y = np.matmul(X, np.transpose(w))
        
        # predict based on probability using sigmoid
        for i in range(len(y)):
            preds[i] = 1 if sigmoid(y[i]) > 0.5 else 0
        
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    X = np.insert(X, D, 1, axis=1)
    
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        
        w = np.zeros((C, D+1))
        b = np.zeros(C)
        
        # calculate gradient with softmax function
        # sgd randomly picks one sample n
        # softmax matrix = (e^(wjTXn)/sum(e^wkTXn, k in C))*XnT, j in C
        # for class y[n], subtract 1 from corresponding softmax matrix value
        # update w with w-step*softmax matrix
        for i in range(max_iterations):
            
            n = np.random.choice(N)
            yn = y[n]
            
            g_numer = np.transpose(np.matmul(w, np.transpose(X[n])))
            g_numer = np.exp(g_numer - np.max(g_numer))
            g_denom = np.sum(g_numer)
            g = np.divide(g_numer, g_denom)  
            g[yn] -= 1
            
            g = np.reshape(g, (C, 1))
            Xn = np.reshape(X[n], (1, D+1))
            step = np.matmul(g, Xn)
            step = np.multiply(step_size, step)
            
            w = np.subtract(w, step)
        
        b = w[:,D]
        w = np.delete(w, D, axis=1)
     
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        
        w = np.zeros((C, D+1))
        b = np.zeros(C)
        
        # same procedures as sgf except using all samples
        # update w with average softmax gradient
        for i in range(max_iterations):
                        
            g_numer = np.matmul(w, np.transpose(X))
            x_max = np.tile(np.amax(g_numer, axis=0), (C, 1))
            g_numer = np.exp(np.subtract(g_numer, x_max))
            g_denom = np.tile(np.sum(g_numer, axis=0), (C, 1))
            g = np.divide(g_numer, g_denom)
            g[y, np.arange(N)] -= 1
            
            step = np.matmul(g, X) / N
            step = np.multiply(step_size, step)
            
            w = np.subtract(w, step)
        
        b = w[:,D]
        w = np.delete(w, D, axis=1)
        
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
        
    X = np.insert(X, D, 1, axis=1)
    w = np.insert(w, D, b, axis=1)
    y = np.matmul(w, np.transpose(X))
    
    # predict based on max value of class
    preds = np.argmax(y, axis=0)
    
    ############################################

    assert preds.shape == (N,)
    return preds




        



