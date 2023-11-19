# Student Names and UUN:  Aryan Verma (s2512060), Chloe Stipinovich (s2614706), Hrushikesh Vazurkar (s2550941)
# https://github.com/infoaryan/NeuralNet_R

# CONTRIBUTIONS:
# Aryan completed the netup function and ensured sound logic for derivative calculations.
# Chloe completed the backward and train functions, and the commenting.
# Hrushikesh completed the forward function and the results formulation.
# We all contributed towards debugging and ensuring sound results,
# all team members contributed towards the assignment fairly.

# INTRODUCTION:
# Neural networks (nn) are a branch of machine learning used to train a model
# to fit some data. A nn is comprised of interconnected nodes (h) which are 
# organized into layers, weights (W) connecting these nodes, and offset 
# values (b). 

# The assignment consisted of a classification task in which numeric variables
# are used to predict what class an observation belongs to. The input layer
# thus has a node for each input variable and the output layer (L) has a node
# for each possible class. The default  constructed nn has 4 input nodes, 3 
# output nodes and two hdden layers of size 8 and 7. The node values are 
# calculated as a linear combination of the nodes in the previous layer and a 
# non-linear transformation of the results, known as an activation fucntion. 
# The ReLU activation function was utilised in this assignment defined as:
#           h^{l+1}_j = max(0, W^{l}_j h^{l} + b^{l}_j)
# In the notatin above, the superscript refers to the layer of the nn and the 
# subscript refers to the position of the node in that layer. 
# The probability of a class was then defined according to the softmax function:
#           p_k = exp(h^{L}_k) / sum(exp(h^{L}_j))

# The model was trained by first defining a Loss Function:
#           Loss = - sum(log(p_k_i)) / n
# Where n is the number of training data points. We then adjust the parameters
# of the nn (the W and b) to minimize this Loss Function. We note that the
# number of parameters in the model far outweighed the number of data points, 
# and so stochastic gradient descent (SGD) was used in the training of the 
# model. To achieve this, we randomly select a small subset of data, a 
# minibatch (mb), of the training data. We run these data through the defined
# nn and update the parameters according to the average gradient of the Loss 
# Function over the mb points.

# The basic procedure used to train the nn is as follows:
# Step 1: Define the training and testing sets.
# Step 2: Randomly initialize the parameters.
# Step 3: Complete the following steps nstep times:
#           > Randomly select a mb from the training data.
#           > For each data point (x_i) with corresponding class (k_i) in mb:
#               - Forward pass x_i through the nn to find all h values.
#               - Compute the derivative of the loss (dLoss) for k_i wrt W and b, using the chain rule:
#                   First compute dLoss_i wrt h^{L}
#                   Then compute dLoss_i wrt all other layers of h by working backwards
#                   Finally, compute the dLoss_i wrt the W and b.
#           > Calculate the average derivative of the loss for the mb.
#           > Update the parameters using the average derivatives:
#           > W = W - eta * dW
#           > b = b - eta * db (where eta is the learning rate)

# For the Iris data set used, the input variables (x_i) were the Sepal Length, 
# Sepal Width Petal Length and Petal Width of flowers and the classes (k) 
# were the three different species (1) setosa, (2) versicolor and 
# (3) virginica of the flowers encoded into numerical values 1, 2 and 3.

# The trained model was then tested to see how well it performed on the 
# training set as well as how well it performed on unseen data (the test set).
# The misclassification rate (proportion misclassified) was calculated and 
# the results can be seen below.

######################################################################################

# INPUT:      d, a vector of numbers giving the number of nodes in
#             each layer of a nn.
# OUTPUT:     nn, a neural network with nodes, h, weights, W, and 
#             offset values b. 
# PURPOSE:    Construct and initialize a nn with nodes, h, weights, W, and
#             offset values b. The nodes are initialized to 0 and the W and b 
#             values are initialized with draws from a Uniform(0, 0.2).
netup <- function(d) {
  
  # Check that the nn has at least 1 hidden layer
  if (length(d) < 2) {
    stop("A neural network should have at least 1 hidden layer.")
  }
  
  # Initialize the network as a list
  nn <- list()
  
  # Initialize the lists for nodes, weights, and offset
  h <- list(); W <- list(); b <- list()
  
  # For each layer of the nn:
  #   Initialize the values of the nodes, h, to 0
  #   Initialize the values of the weights, W, and offsets, b, to be
  #   Uniform(0, 0.2)
  for (l in 1:(length(d))) {
    
    h[[l]] <- numeric(d[l]) # Initialize h
    if(l != length(d)){
      W[[l]] <- matrix(runif(d[l + 1] * d[l], 0, 0.2), # Initialize W
                       nrow = d[l+1], ncol = d[l])
      b[[l]] <- matrix(runif(d[l + 1], 0, 0.2), # Initialize b
                       nrow = d[l+1])
    }
  }
  
  nn$h <- h; nn$W <- W; nn$b <- b
  
  return(nn) 
}

# INPUT:      nn, a nn defined by netup.
#             inp, input values for a particular data point (1x4).
# OUTPUT:     nn, the neural network after forward propagation, i.e. with
#             updated node values according to the W and b defined in nn.
# PURPOSE:    Given a nn with defined W and b, and a set of input values,
#             this function will calculate the values of the nodes at each
#             layer of the nn, applying a ReLU activation function:
#             h^{l+1} = max(0, W^l %*% h^l + b^l)
forward <- function(nn, inp) {
  
  h <- nn$h; W <- nn$W; b <- nn$b 
  inp <- matrix(inp[1,], nrow = 1) # ensure inp is the correct form
  h[[1]] <- t(inp) # feed inp values into first layer of nn
  for(i in 1:(length(h)-1)){
    h[[i+1]] <- W[[i]]%*%h[[i]] + b[[i]] # Calculate h^{l+1} = W^l %*% h^l + b^l
    h[[i+1]][h[[i+1]] < 0] <- 0 # Apply the ReLU transformation
  }
  nn$h <- h
  return(nn)
}

# INPUT:      nn, a nn which has been passed through forward.
#             k, the class values corresponding to the inputs used to define
#             nn.
# OUTPUT:     nn, the same nn which was given as an input but with added
#             elements, dh, dW and db the derivative of the Loss Function wrt
#             the nodes, Weights and offsets respectively.
# PURPOSE:    To perform back-propagation by calculating the derivative of the Loss
#             Function wrt the nodes, Weights, and offsets.
backward <- function(nn, k) {
  
  h <- nn$h; W <- nn$W; b <- nn$b
  L <- length(h) # define the number of layers in nn
  k <- k[,1] # ensure k has the correct dimensions

  # initialize derivatives
  dh <- vector("list", length = L)
  dW <- vector("list", length = L-1)
  db <- vector("list", length = L-1)
  
  for (l in L:1){
    if (l == L){ # final layer
      # Compute the derivative of the loss wrt the final layer
      dh_L <- exp(h[[l]]) / colSums(exp(h[[l]])) 
      dh_L[k,] <-  dh_L[k,] - 1
      dh[[l]] <- dh_L
      
    }
    else{
      d_l_plus_1 <- dh[[l+1]] # derivative of the loss wrt h^{l+1}
      d_l_plus_1[h[[l+1]] < 0] <- 0 # set dh values to 0 where h values are <0
      #d_l_plus_1 <- apply(dh[[l+1]], 1:2,
      #                    function(x){ max(0,x) })
      dh[[l]] <- t(W[[l]]) %*% d_l_plus_1 # derivative of loss wrt h
      db[[l]] <- d_l_plus_1 # derivative of loss wrt b
      dW[[l]] <- d_l_plus_1 %*% t(h[[l]]) # derivative of loss wrt W
    }
  }
  
  nn$dh <- dh
  nn$dW <- dW
  nn$db <- db
  return(nn)
}

# INPUT:      list 1, a list of matrices from train
#             list 2, a list of matrices from train
#             operator, the operation to be applied either 'add' or 'subtract' 
# OUTPUT:     result_list, a list of matrices
# PURPOSE:    This function performs elementwise matrix addition/subtraction
#             on a list of matrices
elementwise_manipulation <- function(list1, list2, operator) {
  result_list <- list() # empty list for results

  # Add the elements of the matrices in list1 and list2
  if (operator == 'add'){
    for (i in 1:length(list1)) {
      result_list[[i]] <- list1[[i]] + list2[[i]] 
    }
  }

  # Subtract the elements of the matrix list2 from list1
  if (operator == 'subtract'){
    for (i in 1:length(list1)) {
      result_list[[i]] <- list1[[i]] - list2[[i]] # subtract matrices elementwise
    }
  }
  return(result_list)
}

# INPUT:      nn, a neural network defined by netup
#             inp, the input data for training from 
#             k, the corresponding true class values
#             eta, the learing rate with default 0.01
#             mb, the mini-batch size with default 10
#             nstep, the number of iterations with defualt 10000
# OUTPUT:     nn, a trained neural network
# PURPOSE:    This function trains a nn with ReLU activation functions at each
#             node with a softmax activation applied to the final node,
#             and a negative log probability loss function. The function uses
#             stochastic gradient decent for training.
train <- function(nn, inp, k, eta = 0.01, mb = 10, nstep = 10000) {
  
  # initialize loss value
  loss_values <- c()
  
  for (step in 1:nstep) {

    # Randomly sample a small subset of the data
    indices <- sample(1:nrow(inp), mb, replace = TRUE)
    inp_mb <- inp[indices, ]
    k_mb <- k[indices]
    
    # Initialize subset loss
    mb_loss <- 0 
    
    for (item in 1:mb){
      
      # Isolate one data point
      inp_item <- matrix(inp_mb[item,], nrow = 1)
      k_item <- matrix(k_mb[item], nrow = 1)

      # Pass data point through the forward and backward function
      nn_forward <- forward(nn, inp_item)
      nn_backward <- backward(nn_forward, k_item)

      # Add derivatives to a stored cumulative value
      if (item == 1) { # initialize cumulative weights and offsets
        cum_dW <- nn_backward$dW
        cum_db <- nn_backward$db
      }
      else { # add current weights to cumulative weights and offsets
        cum_dW <- elementwise_manipulation(cum_dW, nn_backward$dW, 'add')
        cum_db <- elementwise_manipulation(cum_db, nn_backward$db, 'add')
      }
      
      # Isolate final layer
      L <- length(nn_forward$h)
      final_layer <- nn_forward$h[[L]]
      
      # Compute probabilities
      p <- exp(final_layer)/ colSums(exp(final_layer))
      if (!all.equal(colSums(p), 1)){
        print("Something is wrong; your probabilities do not sum to 1.")
      }
      
      # Store sum of negative log probability for the class of item
      mb_loss <- mb_loss - log(p[k_item,]) 
    }
    
    # Calculate the loss of the whole subset as the average
    loss = mb_loss/mb 

    # Store this mini batch loss value
    loss_values <- c(loss_values, loss)
    
    # Calculate the average of the derivative wrt the W and b
    dW_average <- lapply(cum_dW, function(x){x/mb}) 
    db_average <- lapply(cum_db, function(x){x/mb}) 
    
    # Update the W and b with the average of the derivatives wrt the W and b
    # W = W - eta * dW
    # b = b - eta * db
    nn$W <- elementwise_manipulation(nn$W, lapply(dW_average, function(x){x * eta}), 'subtract') 
    nn$b <- elementwise_manipulation(nn$b, lapply(db_average, function(x){x * eta}), 'subtract') 
  }
  return(nn)
}

# INPUT:      k_hat, vector of predicted class labels
#             k_true, vector of true class labels
# OUTPUT:     confusion_matrix, a confusion matrix of k_hat and k_true
# PURPOSE:    Generating a confusion matrix given some predicted 
#             and true class labels.
cnf_generator <- function(k_hat, k_true){

  # Initialize matrix
  cnf_matrix <- matrix(0, nrow=3, ncol=3)

  for(i in 1:length(k_hat)){
    pred_val <- k_hat[i]
    actual_val <- k_true[i]
    cnf_matrix[pred_val, actual_val] <- cnf_matrix[pred_val, actual_val] + 1
  }
  
  return(cnf_matrix)
}

# INPUT:      inp, input variables
#             nn, defined nn
# OUTPUT:     k_hat, predicted class labels
# PURPOSE:    Finds the predicted class labels given some input values
#             and a defined nn
prediction_calculator <- function(inp, nn){

  # initialize predictions vector
  k_hat <- c() 

  # L = number of layers in nn
  L <- length(nn$h) 

  for (i in 1:nrow(inp)){
    nn <- forward(nn, matrix(inp[i,], nrow = 1)) # pass inputs through nn
    final_layer <- nn$h[[L]] # Isolate final layer
    k_hat <- c(k_hat, which.max(final_layer)) # prediction = argmax of final layer
  }
  return(k_hat)
}

# INPUT:      k_hat, vector of predicted class labels
#             k_true, vector of true class labels
# OUTPUT:     missclass, the misclassification raate
# PURPOSE:    Calculates the miss-classification rate given the true classes 
#             and the predicted values
missclass_calculator <- function(k_hat, k_true){

  # identify and count the differences
  differences <- k_hat != k_true 
  num_differences <- sum(differences) 

  # Calculate the proportion
  missclass <- num_differences / length(k_hat) 
  return(missclass)
}

# Extract dependent variables: inp
inp <- as.matrix(iris[, 1:4])

# Extract class labels as numerics 1,2,3 for the three species
k <- matrix(as.numeric(iris$Species), ncol=1)

# Define training and testing sets
test_indices <- seq(5, nrow(iris), by = 5)
inp_train <- inp[-test_indices, ]
k_train <- k[-test_indices, ]
inp_test <- inp[test_indices, ]
k_test <- k[test_indices, ]

# Set the seed for pseudo-random number generation
set.seed(6)

# Define number of nodes in each layer of nn
nn <- netup(c(4,8,7,3))

# Train nn on the training data
nn_trained <- train(nn, inp_train, k_train,
                    eta = 0.01, mb = 10, nstep = 10000)

# RESULTS:

# misclassification rate pre-processing on the training data
k_hat_pre_train <- prediction_calculator(inp_train, nn)
missclass_pre_train <- missclass_calculator(k_hat_pre_train, k_train)
missclass_pre_train

# misclassification rate post-processing on the training data
k_hat_training <- prediction_calculator(inp_train, nn_trained)
missclass_post_train <- missclass_calculator(k_hat_training, k_train)
missclass_post_train

# As seen above, the misclassification rate from the pre trained 
# nn to the post trained nn decreased from 0.66 to 0.025.

# misclassification rate on the testing data
k_hat_test <- prediction_calculator(inp_test, nn_trained)
missclass_test <- missclass_calculator(k_hat_test, k_test)
missclass_test

# There are no misclassification in the testing data.

# Confusion matrix for training set
cnf_generator(k_hat_training, k_train)

# Confusion matrix for testing set
cnf_generator(k_hat_test, k_test)
