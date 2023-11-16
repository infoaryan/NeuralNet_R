netup <- function(d) {
  # Check if the d has at least 2 elements 
  if (length(d) < 2) {
    stop("A neural network should have at least 1 hidden layer.")
  }
  
  # Initialize the network as a list
  nn <- list()
  
  # Initialize the lists for nodes, weights, and bias
  h <- list()
  W <- list()
  b <- list()
  
  set.seed(111) 
  
  # Loop through each layer dim and initialize the values
  for (l in 1:(length(d))) {
    # Initialize nodes for this layer that will contain node values
    h[[l]] <- numeric(d[l])
    
    if(l != length(d)){
      # Initialize weight matrix W[[l]] with random values from U(0, 0.2)
      W[[l]] <- matrix(runif(d[l + 1] * d[l], 0, 0.2), nrow = d[l+1], ncol = d[l])
      
      # Initialize offset vector b[[l]] with random values from U(0, 0.2)
      b[[l]] <- matrix(runif(d[l + 1], 0, 0.2), nrow = d[l+1])
    }
  }
  
  nn$h <- h
  nn$W <- W
  nn$b <- b
  
  return(nn)
}

forward <- function(nn, inp) {
  h <- nn$h; W <- nn$W; b <- nn$b
  inp <- matrix(inp[1,], nrow = 1)
  h[[1]] <- t(inp)
  for(i in 1:(length(h)-1)){
    h[[i+1]] <- apply(W[[i]]%*%h[[i]], 2,
                      function(x){ x + b[[i]] })
    h[[i+1]] <- apply(h[[i+1]], 1:2,
                      function(x){ max(0,x) })
  }
  nn$h <- h
  return(nn)
}


# Function to perform backward pass and update weights and biases

backward <- function(nn_forward, k_forward) {
  
  h <- nn_forward$h; W <- nn_forward$W; b <- nn_forward$b
  L <- length(h); L
  k <- k_forward[,1]

  # initialize derivatives
  dh <- vector("list", length = L); dh
  dW <- vector("list", length = L-1); dW
  db <- vector("list", length = L-1); db
  
  for (l in L:1){
    if (l == L){
      # i = 3
      #dh_L <- matrix()
      dh_L <- exp(h[[l]]) / colSums(exp(h[[l]])); dh_L
      dh_L[k,] <-  dh_L[k,] - 1; dh_L
      dh[[l]] <- dh_L
      
    }
    else{
      d_l_plus_1 <- dh[[l+1]]
      d_l_plus_1[h[[l+1]] < 0] <- 0
      #d_l_plus_1 <- apply(dh[[l+1]], 1:2,
      #                    function(x){ max(0,x) })
      dh[[l]] <- t(W[[l]]) %*% d_l_plus_1; dh
      db[[l]] <- d_l_plus_1; db
      dW[[l]] <- d_l_plus_1 %*% t(h[[l]]); dW
    }
  }
  
  nn_forward$dh <- dh
  nn_forward$dW <- dW
  nn_forward$db <- db
  return(nn_forward)
  
}

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

# Training function
train <- function(nn, inp, k, eta = 0.01, subset_size = 10, nstep = 10000) {

  set.seed(2)
  loss_values <- c()
  loss <- 0
  
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
      L <- length(nn_forward$h); L
      final_layer <- nn_forward$h[[L]]
      
      # Compute probabilities
      p <- exp(final_layer)/ colSums(exp(final_layer))
      if (all.equal(colSums(p), 1) == FALSE){
        print("Something is wrong; your probabilities do not sum to 1.")
      }
      
      # Store sum of negative log probability for the class of item
      subset_loss <- subset_loss - log(p[k_item,]) 
      
      #loss <- loss_function(k_subset, final_layer)
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
  
  return(nn)
}

# Function generatiung a confusion matrix
cnf_generator <- function(k_pred, k_actual){
  confusion_matrix <- matrix(0, nrow=3, ncol=3)
  sample_size <- length(k_pred)
  
  for(i in 1:sample_size){
    pred_val <- k_pred[i]; actual_val <- k_actual[i]
    confusion_matrix[pred_val, actual_val] <- confusion_matrix[pred_val, actual_val] + 1
  }
  
  confusion_matrix
}

# Extract dependent variables: inp
inp <- as.matrix(iris[, 1:4]); inp

# Extract class labels as numerics 1,2,3 for the three species
k <- matrix(as.numeric(iris$Species), ncol=1); k

# Define training and testing sets
test_indices <- seq(5, nrow(iris), by = 5)
inp_training <- inp[-test_indices, ]; k_training <- k[-test_indices, ]
inp_testing <- inp[test_indices, ]; k_testing <- k[test_indices, ]

# Construct Neural Net
nn <- netup(c(4,8,7,3))

# Train Neural Net on training data
nn_trained <- train(nn, inp_training, k_training,
                    eta = 0.01, subset_size = 10, nstep = 10000)

nn_trained

# Test how well training set classified data
k_hat_training <- k_training * 0 # initialize prediction vector
L <- length(nn_trained$h); L
for (i in 1:nrow(inp_training)){
  nn_trained_i <- forward(nn_trained, matrix(inp_training[i,], nrow = 1))
  # Isolate final layer
  final_layer <- nn_trained_i$h[[L]]
  k_hat_training[i] <- which.max(final_layer)
}
k_hat_training

# Test how well the trained nn will perform on the test data
k_hat_test <- k_testing * 0 # initialize prediction vector
L <- length(nn_trained$h); L
for (i in 1:nrow(inp_testing)){
  nn_trained_i <- forward(nn_trained, matrix(inp_testing[i,], nrow = 1))
  # Isolate final layer
  final_layer <- nn_trained_i$h[[L]]
  k_hat_test[i] <- which.max(final_layer)
}

# Confusion matrix for training set
cnf_generator(k_hat_training, k_training)

# Confusion matrix for testing set
cnf_generator(k_hat_test, k_testing)


