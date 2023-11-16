netup <- function(d) {
  # Check if the d has at least 2 elements 
  if (length(d) < 2) {
    stop("Should have more than 2 length")
  }
  
  # Initialize the network as a list
  network <- list()
  
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
  
  network$h <- h
  network$W <- W
  network$b <- b
  
  return(network)
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
    #h[[i+1]] <- apply(W[[i]]%*%h[[i]] + b[[i]], 1:2, function(x){ max(0,x) })
  }
  nn$h <- h
  return(nn)
}

####
# Function to perform backward pass and update weights and biases

backward <- function(nn, k) {
  h <- nn$h; W <- nn$W; b <- nn$b
  L <- length(h); L
  k<- k[,1]
  #factor_labels <- factor(k)
  # Perform one-hot encoding using model.matrix
  #one_hot_matrix <- model.matrix(~factor_labels - 1)
  #one_hot_matrix <- matrix(one_hot_matrix, ncol = ncol(one_hot_matrix), nrow = nrow(one_hot_matrix))

  # initialize derivatives
  dh <- vector("list", length = L); dh
  dW <- vector("list", length = L-1); dW
  db <- vector("list", length = L-1); db
  
  for (l in L:1){
    if (l == L){
      # i = 3
      dh_L <- exp(h[[l]]) / colSums(exp(h[[l]])); dh_L
      dh_L[k,] <-  dh_L[k,] - 1; dh_L
      dh[[l]] <- dh_L
    }
    else{
      d_l_plus_1 <- apply(dh[[l+1]], 1:2,
                          function(x){ max(0,x) })
      dh[[l]] <- t(W[[l]]) %*% d_l_plus_1; dh
      db[[l]] <- d_l_plus_1; db
      dW[[l]] <- d_l_plus_1 %*% t(h[[l]]); dW
    }
  }
  
  network <- list()
  network$h <- h
  network$W <- W
  network$b <- b
  network$dh <- dh
  network$dW <- dW
  network$db <- db
  return(network)
  
}

#loss_function <- function(k_subset, h[[length(h)]]){
  
#}

sum_lists_elementwise <- function(list1, list2) {
  
  result_list <- list() # empty list for results
  for (i in 1:length(list1)) {
    result_list[[i]] <- list1[[i]] + list2[[i]]
  }
  return(result_list)
}

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
      if (item == 1) {
        #print(item)
        cum_dW <- nn_backward$dW
        cum_db <- nn_backward$db
      }
      else {
        #print(item)
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
    if (step %% 100 == 0) {
     cat("Step:", step, "  Loss:", loss, "\n")
    }
  }
  
  return(nn)
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
                    eta = 0.01, subset_size = 10, nstep = 1000)

nn_trained

############## the code below this point is not complete but can be used while we debug ########

# Test how well training set classified data
k_hat <- k_training * 0 # initialize prediction vector
for (i in 1:nrow(inp_training)){
  nn_forward_trained <- forward(nn_trained, matrix(inp_training[i,], nrow = 1))
  
}
nn_forward_trained <- forward(nn_trained, inp_training)

nn_forward_trained$h



# temp for testing ######
#nn <- netup(c(4,8,7,3))
nn <- netup(c(4,2,2,3))
#inp <- matrix(c(5.1, 3.5, 1.4, 0.2), nrow=1)
#inp <- matrix(c(1,2,3,4,
#               5,6,7,8), nrow=2)
inp <- as.matrix(iris[, 1:4]); inp
k <- matrix(as.numeric(iris$Species), ncol=1); k
subset_size = 10
#subset_size = 3
eta = 0.1
nstep = 10000
#nstep = 10



#nn <- netup(c(4,8,7,3))
nn <- netup(c(4,4,4,2))
#inp <- matrix(c(5.1, 3.5, 1.4, 0.2), nrow=1)
inp <- matrix(c(5.1, 3.5, 1.4, 0.2,
                5.1, 3.5, 1.4, 0.2), nrow=2)
k <- matrix(c(1,2), nrow=1)
nn_forward <- forward(nn, inp)
nn_backward <- backward(nn_forward,k)

subset_size = 2
eta = 0.01

