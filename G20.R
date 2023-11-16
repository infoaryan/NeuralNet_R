# Group 20
# Student Names and UUN:  Aryan Verma(s2512060)
#                         Chloe Stipinovich(s2614706)
#                         Hrushikesh Vazurkar(s2550941)


# INPUT:      d, a vector of numbers giving the number of nodes in
#             each layer of a nn.
# OUTPUT:     nn, a neural network with nodes, h, weights, W, and 
#             bias values b. 
# PURPOSE:    Construct and initialize a nn with nodes, h, weights, W, and
#             bias values b. The nodes are initialized to 0 and the W and b 
#             values are initialized with elements drawn from a Uniform(0, 0.2).
netup <- function(d) {
  
  # Check that the nn has at least 1 hidden layer
  if (length(d) < 2) {
    stop("A neural network should have at least 1 hidden layer.")
  }
  
  # Initialize the network as a list
  nn <- list()
  
  # Initialize the lists for nodes, weights, and bias
  h <- list(); W <- list(); b <- list()
  
  # For each layer of the nn:
  #   Initialize the values of the nodes, h, to 0
  #   Initialize the values of the weights, W, and biases, b, to be
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
#             the nodes, Weights and biases respectively.
# PURPOSE:    To perform back-propagation by calculating the derivative of the Loss
#             Function wrt the nodes, Weights, and biases.
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

# INPUT:      
# OUTPUT:     
# PURPOSE:    
# Function which performs matrix summation in a list of matrices
sum_lists_elementwise <- function(list1, list2) {
  
  result_list <- list() # empty list for results
  for (i in 1:length(list1)) {
    result_list[[i]] <- list1[[i]] + list2[[i]]
  }
  return(result_list)
}

# Function which performs matrix summation in a list of matrices
subtract_lists_elementwise <- function(list1, list2) {
  
  result_list <- list() # empty list for results
  for (i in 1:length(list1)) {
    result_list[[i]] <- list1[[i]] - list2[[i]]
  }
  return(result_list)
}

# INPUT:      
# OUTPUT:     
# PURPOSE:    
# Training function
train <- function(nn, inp, k, eta = 0.01, subset_size = 10, nstep = 10000) {

  loss_values <- c()
  
  for (step in 1:nstep) {
    # Randomly sample a small subset of the data
    indices <- sample(1:nrow(inp), subset_size, replace = TRUE)
    inp_subset <- inp[indices, ]
    k_subset <- k[indices]
    
    subset_loss <- 0 # initialize
    
    for (item in 1:subset_size){
      inp_item <- matrix(inp_subset[item,], nrow = 1)
      k_item <- matrix(k_subset[item], nrow = 1)
      nn_forward <- forward(nn, inp_item)
      nn_backward <- backward(nn_forward, k_item)
      if (item == 1) { # initialize cumulative weights and biases
        cum_dW <- nn_backward$dW
        cum_db <- nn_backward$db
      }
      else { # add current weights to cumulative weights and biases
        cum_dW <- sum_lists_elementwise(cum_dW, nn_backward$dW)
        cum_db <- sum_lists_elementwise(cum_db, nn_backward$db)
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
      subset_loss <- subset_loss - log(p[k_item,]) 
  
    }
    
    loss = subset_loss/subset_size # Find the loss of the whole subset by taking the average of the cumulative loss
    loss_values <- c(loss_values, loss) # store this loss value
    
    dW_average <- lapply(cum_dW, function(x){x/subset_size}) # find the average of the derivative wrt the weights
    db_average <- lapply(cum_db, function(x){x/subset_size}) # find the average of the derivative wrt the bias
    
    nn$W <- subtract_lists_elementwise(nn$W, lapply(dW_average, function(x){x * eta})) # update the weight of the nn
    nn$b <- subtract_lists_elementwise(nn$b, lapply(db_average, function(x){x * eta})) # update the bias of the nn
    
    # Print the loss for every 1000 steps
    if (step %% 1000 == 0) {
     cat("Step:", step, "  Loss:", loss, "\n")
    }
    
  }
  
  nn$loss_values <- loss_values
  return(nn)
}

# INPUT:      
# OUTPUT:     
# PURPOSE:    
# Function generating a confusion matrix
cnf_generator <- function(k_hat, k_true){
  confusion_matrix <- matrix(0, nrow=3, ncol=3)
  sample_size <- length(k_hat)
  
  for(i in 1:sample_size){
    pred_val <- k_hat[i]; actual_val <- k_true[i]
    confusion_matrix[pred_val, actual_val] <- confusion_matrix[pred_val, actual_val] + 1
  }
  
  confusion_matrix
}

# INPUT:      
# OUTPUT:     
# PURPOSE:    
# Training function
# function which finds the predicted class labels given 
# some input values and a defined nn
prediction_calcuator <- function(inp, nn){
  k_hat <- list() # initialize predictions vector
  L <- length(nn$h) # L = number of layers in nn
  for (i in 1:nrow(inp)){
    nn <- forward(nn, matrix(inp[i,], nrow = 1)) # pass inputs through nn
    final_layer <- nn$h[[L]] # Isolate final layer
    k_hat <- c(k_hat, which.max(final_layer)) # prediction = argmax of final layer
  }
  return(k_hat)
}

# INPUT:      
# OUTPUT:     
# PURPOSE:    
# Training function
# calculates the miss-classification rate given the true classes 
# and the predicte values
missclass_calculator <- function(k_hat, k_true){
  differences <- k_hat != k_true # identify the differences
  num_differences <- sum(differences) # count the differences
  missclass <- num_differences / length(k_hat) # Calculate the proportion
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
                    eta = 0.01, subset_size = 10, nstep = 10000)

###### RESULTS ##########

# misclassification rate pre-processing on the training data
k_hat <- prediction_calcuator(inp_train, nn)
missclass_pre_train <- missclass_calculator(k_hat, k_train)
missclass_pre_train

# misclassification rate post-processing on the training data
k_hat <- prediction_calcuator(inp_train, nn_trained)
missclass_post_train <- missclass_calculator(k_hat, k_train)
missclass_post_train

# As seen above, the misclassification rate from the pre trained 
# nn to the post trained nn decreased from 0.66 to 0.025.

# misclassification rate on the testing data
k_hat <- prediction_calcuator(inp_test, nn_trained)
missclass_test <- missclass_calculator(k_hat, k_test)
missclass_test

# There are no misclassification in the testing data

# Confusion matrix for training set
cnf_generator(k_hat_training, k_train)

# Confusion matrix for testing set
cnf_generator(k_hat_test, k_test)
