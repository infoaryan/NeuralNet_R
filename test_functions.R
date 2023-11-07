# This file is for testing some relevant functions
# Ignore this


# netup - A function 
netup <- function(d) {
  # Check if the d has at least 2 elements 
  if (length(d) < 2) {
    stop("Should have more than 2 lenght")
  }
  
  # Initialize the network as a list
  network <- list()
  
  # Initialize the lists for nodes, weights, and offsets
  h <- list()
  W <- list()
  b <- list()
  
  set.seed(111) 
  
  # Loop through each layer dim and initialie the values
  for (l in 1:(length(d))) {
    # Initialize nodes for this layer that will contain node values
    h[[l]] <- numeric(d[l])
    
    if(l != length(d)){
      # Initialize weight matrix W[[l]] with random values from U(0, 0.2)
      W[[l]] <- matrix(runif(d[l] * d[l + 1], 0, 0.2), nrow = d[l], ncol = d[l + 1])
      
      # Initialize offset vector b[[l]] with random values from U(0, 0.2)
      b[[l]] <- runif(d[l + 1], 0, 0.2)
    }
  }
  
  network$h <- h
  network$W <- W
  network$b <- b
  
  return(network)
}

my_network <- netup(c(4,2,2,3))
print(my_network)
