SpectralClustering <- function(X, NumClust) {
  # ======================================================================= #
  # DESCRIPTION: This function implements normalized spectral clustering
  # INPUT: 
  #     X = (N x d) matrix of covariates, each row corresponds to one unit
  #     NumClust = number of desired clusters
  # OUTPUT:
  #     CC = length N vector of indices mapping units to clusters
  # ======================================================================= #
  
  ### 1. SUBFUNCTIONS
  GaussianSimilarities <- function(X,V = 1) {
    # DESCRIPTION: Computes Gaussian Similarity Matrix
    # INPUT: X = (N x d) matrix of covariates
    #        V = variance parameter
    # OUTPUT: KK = (N x N) similarity matrix
    DD <- as.matrix(dist(X), method = "euclidean")^2 # matrix of squared Euclidean distances 
    KK <- exp(-DD/V/2)                               # matrix of Gaussian similarities
    return(KK)
  }
  
  GraphLaplacian <- function(K, normalized = TRUE){
    # DESCRIPTION: Computes graph Laplacian matrix from similarity matrix K
    # INPUT: K = (N x N) similarity matrix
    # OUTPUT: L = (N x N) graph Laplacian
    # NOTE: normalized Laplacian is the default option, set FALSE for unnormalized case
    stopifnot(nrow(K) == ncol(K))
    
    deg <- colSums(K) # degrees of vertices
    n   <- nrow(K)    # number of observations
    
    if (normalized) {
      Dm1 <- diag(1/sqrt(deg)) # D^{-1/2}
      return(diag(n) - Dm1 %*% K %*% Dm1)
    }
    else {
      return(diag(deg) - K)
    }
  }
  # ======================================================================= #
  
  ### 2. MAIN OPERATIONS
  K    <- GaussianSimilarities(X)                  # similarity matrix
  L    <- GraphLaplacian(K)                        # compute Laplacian matrix, default is normalized
  eigs <- eigen(L, symmetric = TRUE)               # compute eigenvectors/values of L
  
  V    <- eigs$vectors[,(nrow(L) - NumClust):(nrow(L)-1)] # array of eigenvectors corresponding to C smallest eigenvalues
  CC   <- kmeans(V, NumClust)                             # run K-means on eigenvector array
  
  return(CC$cluster) 
}