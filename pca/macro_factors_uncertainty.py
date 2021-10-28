#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
McCracken & Ng (2016) and Jurado, Ludvigson, Ng (2015) Replication Codes
@author: victorsellemi
"""

#%% paths and imports

import numpy             as np    
import numpy.linalg      as LA   
from statsmodels.tsa.stattools import adfuller




#%% data functions

def remove_outliers(X):
    """ 
    Python analog to MATLAB function from McCracken & Ng (2016)
     
    DESCRIPTION:
        This function takes a set of series aligned in the columns of a 
        NumPy array and replaces outliers with nan
    
    INPUT:
        X = dataset (one series per column)
    
    OUTPUT:
        Y = dataset with outliers replaced with NaN
        n = number of outliers found in each series
        
    NOTES: 
        1) Outlier definition: a data point is an outlier if 
        abs(x-median) >= 10*interquartile range
        2) This function replaces outliers for series with missing values as well
            
    """
    
    # calculate median of each series
    median_X = np.nanmedian(X, axis = 0)
    
    # Repeat median of each series over all data points in the series
    median_X_mat = np.kron(np.ones((X.shape[0],1)), median_X)

    # Calculate quartiles 
    Q = []
    for t in range(X.shape[1]):
        Q.append(np.nanpercentile(X[:,t], [25, 50, 75], interpolation = 'midpoint'))
    Q = np.array(Q)

    # Calculate interquartile range (IQR) of each series
    IQR = Q[:,2] - Q[:,0]

    # Repeat IQR of each series over all data points in the series
    IQR_mat = np.kron(np.ones((X.shape[0],1)),IQR)
    
    # Determine outliers 
    Z = abs(X - median_X_mat)
    Z[np.isnan(Z)] = 0
    outlier = Z > (10*IQR_mat)


    # Replace outliers with NaN
    Y = X
    Y[outlier] = np.nan
    
    # count number of outliers
    n = np.sum(outlier, axis = 0)
    
    return Y,n 

def data_transformations(rawdata, tcode):
    """
    Python analog to MATLAB function from McCracken & Ng (2016)
    
    DESCRIPTION:
        this function transforms raw data based on each series transformation code.
        
    INPUT: 
        rawdata = raw data
        tcode   = transformation code for each series
        
    OUTPUT: 
        yt      = transformed data
        
    SUBFUNCTION:
        transxf: transforms a single series as specified by a transformation code
    """
    
    # subfunction
    def transxf(x, tcode):
        """
        DESCRIPTION:
            This function transforms a single series (in a column vector) as specified
            by a given transformation code.
            
        INPUT: 
            x = series to be transformed
            tcode = transformation code (1-7)
            
        OUTPUT: 
            y = transformed series as a column vector            
        """
        # number of observations (including missing values)
        n = x.shape[0]
        
        # value close to zero
        small = 1e-6
        
        # allocation output variable
        y = np.nan*np.ones((n,1))
        
        # transformations
        
        # Case 1, no transformation
        if tcode == 1: 
            y = x
            
        # Case 2, first difference: x(t) - x(t-1)
        if tcode == 2: 
            y[1:] = x[1:] - x[:-1]
        
        # Case 3, second difference: [x(t) - x(t-1)] - [x(t-1) - x(t-2)]
        if tcode == 3: 
            y[2:] = x[2:] - 2*x[2:-1] + x[:-2]
        
        # Case 4, natural log: ln(x)
        if tcode == 4: 
            y[x <= small] = np.nan
            y[x > small] = np.log(x[x > small])
        
        # Case 5, first difference of natural log: ln(x(t)) - ln(x(t-1))
        if tcode == 5: 
            if min(x) > small:
                x = np.log(x)
                y[1:] = x[1:] - x[:-1]
        
        # Case 6, second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
        if tcode == 6:
            if min(x) > small:
                x = np.log(x)
                y[2:] = x[2:] - 2*x[1:-1] + x[:-2]
    
        # Case 7, first difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)    
        if tcode == 7: 
            y1 = (x[1:] - x[:-1]) / x[:-1]
            y[2:] = y1[1:] - y1[:-1]
            
        return y
    
    # initialize output variable
    yt = np.zeros(rawdata.shape)
    
    # number of series kept 
    N = rawdata.shape[1]
    
    # perform transformation using subfunction transxf
    for i in range(N):
        temp = transxf(np.expand_dims(rawdata[:,i],axis=1), tcode[i])
        yt[:,i] = temp.flatten()
    
    return yt
    
def obtain_factors(x, kmax, jj, DEMEAN):
    """
    Python analog to MATLAB function from McCracken & Ng (2016)
    
    DESCRIPTION: 
        This function estimates a set of factors for a given dataset using principal 
        component analysis. The number of factors is determined by an information
        criterion specified by the user. Missing values in the original dataset are handled 
        using an iterative expectation maximization algorithm
        
    INPUTS: 
        x    = dataset (one series per column)
        
        kmax = an integer indicating the maximum number of factors to be estimated 
                if kmax = 99, number of factors is 8
                
        jj   = an integer indicating the information criterion used for selecting the number of factors
                1 (information criterion PC_p1)
                2 (information criterion PC_p2)
                3 (information criterion PC_p3) 
                
        DEMEAN = an integer indicating the type of transformation
                performed on each series in x before the factors are
                estimated; it can take on the following values:
                          0 (no transformation)
                          1 (demean only)
                          2 (demean and standardize)
                          3 (recursively demean and then standardize)
        
    OUTPUTS:
        ehat    = difference between x and values of x predicted by
                     the factors
                     
        Fhat    = set of factors
        
        lamhat  = factor loadings
        
        ve2     = eigenvalues of x3'*x3 (where x3 is the dataset x post
                     transformation and with missing values filled in)
    
        x2      = x with missing values replaced from the EM algorithm
        
    SUBFUNCTIONS:
        baing() - selects number of factors
        
        pc2() - runs principal components analysis
        
        minindc() - finds the index of the minimum value for each column of a 
        given matrix
        
        transform_data() - performs data transformation
        
    BREAKDOWN OF THE FUNCTION:
        
        Part 1: Check that inputs are specified correctly.
        
        Part 2: Setup.
        
        Part 3: Initialize the EM algorithm -- fill in missing values with
             unconditional mean and estimate factors using the updated
             dataset.

        Part 4: Perform the EM algorithm -- update missing values using factors,
             construct a new set of factors from the updated dataset, and
             repeat until the factor estimates do not change.
        
    NOTES:
        Details for the three possible information criteria can be found in the
        paper "Determining the Number of Factors in Approximate Factor Models" by
        Bai and Ng (2002).
        
        The EM algorithm is essentially the one given in the paper "Macroeconomic
        Forecasting Using Diffusion Indexes" by Stock and Watson (2002). The
        algorithm is initialized by filling in missing values with the
        unconditional mean of the series, demeaning and standardizing the updated
        dataset, estimating factors from this demeaned and standardized dataset,
        and then using these factors to predict the dataset. The algorithm then
        proceeds as follows: update missing values using values predicted by the
        latest set of factors, demean and standardize the updated dataset,
        estimate a new set of factors using the demeaned and standardized updated
        dataset, and repeat the process until the factor estimates do not change.

    """
    
    # PART 0: SUBFUNCTIONS
    
    def minindc(x):
        """
        DESCRIPTION:
             This function finds the index of the minimum value for each column of a
             given matrix. The function assumes that the minimum value of each column
             occurs only once within that column. The function returns an error if
             this is not the case.

        INPUT:
            x   = matrix 
        OUTPUT:
            pos = column vector with pos(i) containing the row number
                corresponding to the minimum value of x(:,i)
        """
        # number of rows and columns of x
        nrows = x.shape[0]
        ncols = x.shape[1]
        
        # preallocate memory for output array
        pos = np.zeros((ncols, 1))
        
        # create column vector 1:nrows
        seq = np.expand_dims(np.arange(1,nrows+1, dtype = np.float64), axis=1)
        
        # find the index of the minimum value of each columns of x
        
        for i in range(ncols):
            
            # minimum value of column i
            min_i = min(x[:,i])
            
            # column vector containing the row number corresponding to the minimum
            # value of x[:,i] in that row and zeros elsewhere
            colmin_i = seq * ((x[:,i] - min_i) == 0).reshape(seq.shape)
            
            # produce an error if the minimum value occurs more than once
            if sum(colmin_i >0) > 1:
                raise ValueError('Minimum value occurs more than once.')
                
            # obtain the index of the minimum value by taking the sum of the column vector 
            pos[i] = sum(colmin_i)
            
        return pos
            
            
    def baing(X, kmax, jj):
        """
        DESCRIPTION:
            This function determines the number of factors to be selected for a given
            dataset using one of three information criteria specified by the user.
            The user also specifies the maximum number of factors to be selected.
            
        INPUTS:
            X       = dataset (one series per column)
            kmax    = an integer indicating the maximum number of factors
                        to be estimated
            jj      = an integer indicating the information criterion used 
                      for selecting the number of factors; it can take on 
                      the following values:
                            1 (information criterion PC_p1)
                            2 (information criterion PC_p2)
                            3 (information criterion PC_p3)    
        OUTPUTS:
            ic1     = number of factors selected
            chat    = values of X predicted by the factors
            Fhat    = factors
            eigval  = eivenvalues of X'*X (or X*X' if N>T)
        
        SUBFUNCTIONS USED:
            
            minindc() - finds the index of the minimum value for each column of a
                given matrix

        BREAKDOWN OF THE FUNCTION:
            Part 1: Setup.
            
            Part 2: Calculate the overfitting penalty for each possible number of
                factors to be selected (from 1 to kmax).
            
            Part 3: Select the number of factors that minimizes the specified
                information criterion by utilizing the overfitting penalties
                calculated in Part 2.
            
            Part 4: Save other output variables to be returned by the function (chat,
                Fhat, and eigval). 

        """
        # PART 1: SETUP
        
        # observations per series (i.e., number of rows)
        T = X.shape[0]
        
        # number of series (i.e., number of columns)
        T = X.shape[1]
        
        # total number of observations
        NT = N * T
        
        # number of rows + columns
        NT1 = N + T
        
        # -----------------------------------
        
        # PART 2: OVERFITTING PENALTY
        # determine penalty for overfitting based on selected information criterion
        
        # allocate memory for overfitting penalty
        CT = np.zeros((1,kmax))
        
        # array containing possible number of factors that can be selected (1:kmax)
        ii = np.arange(1,kmax+1,1)
        
        # the smaller of N and T
        GCT = min(N,T)
        
        # calculate penalty based on criterion determined by jj
        
        # criterion PC_p1
        if jj == 1: 
            CT[0,:] = np.log(NT / NT1) * ii * NT1 / NT
            
        # criterion PC_p2
        if jj == 2:
            CT[0,:] = (NT1 / NT) * np.log(min(N,T)) * ii
            
        # criterion PC_p3
        if jj == 3:
            CT[0,:] = ii * np.log(GCT) / GCT
        
        # -----------------------------------
            
        # PART 3: SELECT NUMBER OF FACTORS
        # perform principal component analysis on the dataset and select the number
        # of factors that minimizes the specified information criterion.
        
        # RUN PRINCIPAL COMPONENTS ANALYSIS 
            
        # get components loadings and eigenvalues
        if T < N:
            
            # singular value decomposition
            ev, eigval, V = LA.svd(X.T.dot(X))
            
            # components
            Fhat0 = np.sqrt(T) * ev
            
            # loadings
            Lambda0 = X.T.dot(Fhat0) / T
        
        if T >= N: 
            
            # singular value decomposition
            ev, eigval, V = LA.svd(X.T.dot(X))
            
            # loadings
            Lambda0 = np.sqrt(N) * ev
            
            # components
            Fhat0 = X.dot(Lambda0) / N 
            
        # SELECT NUMBER OF FACTORS
        
        # preallocate memory
        Sigma = np.zeros((1,kmax+1)) # sum of squared residuals divided by NT
        IC1 = np.zeros((CT.shape[0], kmax + 1)) # information criterion value
        
        # loop through all possibilities for the number of factors
        for i in np.arange(kmax-1,-1,-1):
            
            # identify factors as first i components
            Fhat = Fhat0[:,:i]
            
            # identify facrtor loadings as first i loadings
            lambd = Lambda0[:,:i]
            
            # predict X using i factors
            chat = Fhat.dot(lambd.T)
            
            # residuals from predicting X using the factors
            ehat = X - chat
            
            # sum of squared residuals divided by NT
            Sigma[0,i] = np.nanmean(np.sum(ehat * ehat / T, axis = 0), axis = 0)
            
            # value of the information criterion when using i factors
            IC1[:,i] = np.log(Sigma[0,i]) + CT[:,i]
            
        #  sum of squared residuals when using no factors to predict X (i.e., fitted values are set to 0)
        Sigma[0,kmax] = np.nanmean(np.sum(X*X/T, axis = 0), axis = 0)
        
        # value of the information criterion when using no factors
        IC1[:, kmax] = np.log(Sigma[0,kmax])
        
        # number of factors that minimizes the information criterion
        ic1 = minindc(IC1.T)
        
        # Set ic1=0 if ic1>kmax (i.e., no factors are selected if the value of the 
        #information criterion is minimized when no factors are used)
        ic1 = ic1 * (ic1 <= kmax)
        
        # -----------------------------------
            
        # PART 4 : SAVE OTHER OUTPUT
        
        # factors and loadings when number of factors set to kmax
        Fhat = Fhat0[:,:kmax-1] # factors
        Lambda = Lambda0[:,:kmax-1] # factor loadings
        
        # predict X using kmax factors
        chat = Fhat.dot(Lambda.T)
        
        # get the eigenvalues corresponding to X'*X (or X*X' if N>T)
        eigval = eigval
        
        return ic1,chat,Fhat,eigval
        
    def pc2(X, nfac): 
        """

        DESCRIPTION:
            This function runs principal component analysis.
        INPUTS:
            X      = dataset (one series per column)
            nfac   = number of factors to be selected
        OUTPUTS:
          chat   = values of X predicted by the factors
          fhat   = factors scaled by (1/sqrt(N)) where N is the number of
                   series
          lambda = factor loadings scaled by number of series
          ss     = eigenvalues of X'*X 
        """
        # Number of series in X (i.e. number of columns)
        N = X.shape[1]
    
        # Singular value decomposition: X'*X = U*S*V'
        U,S,V = LA.svd(X.T.dot(X))
    
        # Factor loadings scaled by sqrt(N)
        lambd = U[:,:int(nfac-1)] * np.sqrt(N)
    
        # Factors scaled by 1/sqrt(N) (note that lambda is scaled by sqrt(N))
        fhat = X.dot(lambd) / N
    
        # estimate initial dataset X using the factors (note that U' = inv(U))
        chat = fhat.dot(lambd.T)
    
        # identify eigenvalues of X'*X
        ss = S
    
        return chat,fhat,lambd,ss

    def transform_data(x2, DEMEAN):
        """

    DESCRIPTION:
         This function transforms a given set of series based upon the input variable DEMEAN. 
         The following transformations are possible:

            1) No transformation.
            2) Each series is demeaned only (i.e. each series is rescaled to have a
                mean of 0).  
            3) Each series is demeaned and standardized (i.e. each series is
                rescaled to have a mean of 0 and a standard deviation of 1).
            4) Each series is recursively demeaned and then standardized. For a
            given series x(t), where t=1,...,T, the recursively demeaned series
            x'(t) is calculated as x'(t) = x(t) - mean(x(1:t)). After the
            recursively demeaned series x'(t) is calculated, it is standardized by
            dividing x'(t) by the standard deviation of the original series x. Note
            that this transformation does not rescale the original series to have a
            specified mean or standard deviation.

     INPUTS:
           x2      = set of series to be transformed (one series per
                     column); no missing values;
           DEMEAN  = an integer indicating the type of transformation
                     performed on each series in x2; it can take on the
                     following values:
                           0 (no transformation)
                           1 (demean only)
                           2 (demean and standardize)
                           3 (recursively demean and then standardize) 

     OUTPUTS:
         
           x22     = transformed dataset
           mut     = matrix containing the values subtracted from x2
                     during the transformation
           sdt     = matrix containing the values that x2 was divided by
                     during the transformation

        """
        # number of observations in each series (i.e., number of rows in x2)
        T = x2.shape[0]
    
        # number of series (i.e., number of columns in x2)
        N = x2.shape[1]
    
        # perform transofmration based on type determined by 'DEMEAN'
        # CASE 0, no transformation
        if DEMEAN == 0:
            mut = np.kron(np.ones((T,1)), np.zeros((1,N)))
            sdt = np.kron(np.ones((T,1)), np.ones((1,N)))
            x22 = x2
    
        # CASE 1, each series is demeaned only
        if DEMEAN == 1: 
            mut = np.kron(np.ones((T,1)), np.mean(x2, axis = 0))
            sdt = np.kron(np.ones((T,1)), np.ones((1,N)))
            x22 = (x2 - mut)
        
        # CASE 2, Each series is demeaned and standardized     
        if DEMEAN == 2: 
            mut = np.kron(np.ones((T,1)), np.mean(x2, axis = 0))
            sdt = np.kron(np.ones((T,1)), np.std(x2, axis = 0))
            x22 = (x2 - mut) / sdt
        
        # CASE 3, each series is recursively demeaned and then standardized
        if DEMEAN == 3:
            mut = np.nan * np.zeros(x2.shape)
            for t in range(T):
                mut[t,:] = np.mean(x2[:t,:], axis = 0)
                sdt = np.kron(np.ones((T,1)), np.std(x2, axis = 0))
                x22 = (x2 - mut) / sdt
        
        return x22, mut, sdt


    # ---------------------------------------------------------------------- #

    # PART 1: CHECKS
    
    #  Check that x is not missing values for an entire row
    if sum(np.sum(np.isnan(x), axis = 1) == x.shape[1]) > 0 :
        raise ValueError('Input x contains entire row of missing values')
        
    #  Check that x is not missing values for an entire column
    if sum(np.sum(np.isnan(x), axis = 0) == x.shape[0]) > 0 :
        raise ValueError('Input x contains entire column of missing values')
        
    #  Check that kmax is an integer between 1 and the number of columns of x, or 99
    if (not (kmax <= x.shape[1] and kmax >= 1 and np.floor(kmax) == kmax)) or kmax == 99:
        raise ValueError('Input kmax is specified incorrectly')

    # Check that jj is one of 1, 2, 3
    if (jj != 1) and (jj != 2) and (jj != 3):
        raise ValueError('Input jj is not specified correctly')
        
    # Check that DEMEAN is one of 0, 1, 2, 3
    if (DEMEAN != 0) and (DEMEAN != 1) and (DEMEAN != 2) and (DEMEAN != 3):
        raise ValueError('Input DEMEAN is not specified correctly')
        
    # ---------------------------------------------------------------------- #
        
    # PART 2: SETUP
    
    # maximum number of iterations for the EM algorithm
    maxit = 50
    
    # number of observations per series (i.e., number of rows)
    T = x.shape[0]
    
    # number of series in x (i.e., number of columns)
    N = x.shape[1]
    
    # set error to arbitrarily high number
    err = 999
    
    # set iteration counter to 0
    it=0
    
    # locate missing values in x
    x1 = np.isnan(x)
    
    # ---------------------------------------------------------------------- #
    # PART 3: INITIALIZE EM ALGORITHM
    # Fill in missing values for each series with the unconditional mean of
    # that series. Demean and standardize the updated dataset. Estimate factors
    # using the demeaned and standardized dataset, and use these factors to
    # predict the original dataset.
    
    # get unconditional mean of the non-missing values of each series
    mut = np.kron(np.ones((T,1)), np.nanmean(x, axis = 0))
    
    # replace missing values with unconditioinal mean
    x2 = x
    x2[np.isnan(x)] = mut[np.isnan(x)]
    
    # Demean and standardize data using subfunction transform_data()
    # x3  = transformed dataset
    # mut = matrix containing the values subtracted from x2 during the
    #     transformation
    # sdt = matrix containing the values that x2 was divided by during the
    #      transformation
    x3,mut,sdt = transform_data(x2, DEMEAN)

    # If input 'kmax' is not set to 99, use subfunction baing() to determine
    # the number of factors to estimate. Otherwise, set number of factors equal
    # to 8
    if kmax != 99:
        icstar,_,_,_ = baing(x3,kmax,jj)
    if kmax == 99:
        icstar = 8

    # Run principal components on updated dataset using subfunction pc2()
    #   chat   = values of x3 predicted by the factors
    #   Fhat   = factors scaled by (1/sqrt(N)) where N is the number of series
    #   lamhat = factor loadings scaled by number of series
    #   ve2    = eigenvalues of x3'*x3 
    chat,Fhat,lamhat,ve2  = pc2(x3,icstar);

    # Save predicted series values
    chat0=chat;
    
    # ---------------------------------------------------------------------- #

    # PART 4: PERFORM EM ALGORITHM
    # Update missing values using values predicted by the latest set of
    # factors. Demean and standardize the updated dataset. Estimate a new set
    # of factors using the updated dataset. Repeat the process until the factor
    # estimates do not change.

    # Run while error is large and have yet to exceed maximum number of
    # iterations
    while (err> 0.000001) and (it <maxit):
    
    # INCREASE ITERATION COUNTER
    
    # Increase iteration counter by 1
        it += 1;
    
    #Display iteration counter, error, and number of factors
        print(f'Iteration {it} obj {err} IC {icstar}')

        # UPDATE MISSING VALUES
    
        # Replace missing observations with latest values predicted by the
        # factors (after undoing any transformation)
        for t in range(T):
            for j in range(N):
                if x1[t,j] == 1:
                    x2[t,j] = chat[t,j] * sdt[t,j] + mut[t,j]
                else:
                    x2[t,j] = x[t,j]    

                # ESTIMATE FACTORS
    
                # Demean/standardize new dataset and recalculate mut and sdt using
                # subfunction transform_data()
                #   x3  = transformed dataset
                #   mut = matrix containing the values subtracted from x2 during the
                #         transformation
                #   sdt = matrix containing the values that x2 was divided by during 
                #         the transformation
        x3,mut,sdt = transform_data(x2,DEMEAN);
    
        # Determine number of factors to estimate for the new dataset using
        # subfunction baing() (or set to 8 if kmax equals 99)
    
        if kmax != 99:
            icstar,_,_,_ = baing(x3, kmax, jj)
    
        if kmax == 99:
            icstar = 8
    

        # Run principal components on the new dataset using subfunction pc2()
        #   chat   = values of x3 predicted by the factors
        #   Fhat   = factors scaled by (1/sqrt(N)) where N is the number of 
        #            series
        #   lamhat = factor loadings scaled by number of series
        #   ve2    = eigenvalues of x3'*x3 
        chat,Fhat,lamhat,ve2  = pc2(x3,icstar);

        # CALCULATE NEW ERROR VALUE
    
        # Calculate difference between the predicted values of the new dataset
        # and the predicted values of the previous dataset
        diff=chat-chat0;
    
        # The error value is equal to the sum of the squared differences
        # between chat and chat0 divided by the sum of the squared values of
        # chat0
        v1 = diff.flatten('C')
        v2 = chat0.flatten('C')
        err = (v1.T.dot(v1)) / (v2.T.dot(v2))
    
        # Set chat0 equal to the current chat
        chat0=chat
    
        # Produce warning if maximum number of iterations is reached
        if it==maxit: 
            raise Warning('Maximum number of iterations reached in EM algorithm')
        
    # FINAL DIFFERNECE
        
    #  Calculate the difference between the initial dataset and the values 
    # predicted by the final set of factors
    ehat = x - chat*sdt-mut
    
    return ehat,Fhat,lamhat,ve2,x2

def mrsq(Fhat, lamhat, ve2, series):
    """
    DESCRIPTION:
        This function computes the R-squared and marginal R-squared from
        estimated factors and factor loadings.

    INPUTS:
           Fhat    = estimated factors (one factor per column)
           lamhat  = factor loadings (one factor per column)
           ve2     = eigenvalues of covariance matrix
           series  = series names

    OUTPUTS:
           R2      = R-squared for each series for each factor
           mR2     = marginal R-squared for each series for each factor
           mR2_F   = marginal R-squared for each factor
           R2_T    = total variation explained by all factors
           t10_s   = top 10 series that load most heavily on each factor
           t10_mR2 = marginal R-squared corresponding to top 10 series
                     that load most heavily on each factor 
    """
    # N = number of series, ic = number of factors
    N, ic = lamhat.shape
    
    # preallocate memory for output
    R2 = np.nan * np.ones((N,ic))
    mR2 = np.nan * np.ones((N,ic))
    t10_s = []
    t10_mR2 = np.nan * np.ones((10,ic))
    
    # compute R-squared and marginal R-squared for each series for each factor
    for i in range(ic):
        R2[:,i] = np.var(Fhat[:,:i].dot(lamhat[:,:i].T), axis = 0).T
        mR2[:,i] = np.var(np.expand_dims(Fhat[:,i],axis=1).dot(np.expand_dims(lamhat[:,i],axis=1).T), axis = 0).T
    
    # compute marginal R-squared for each factor
    mR2_F = ve2 / np.sum(ve2)
    mR2_F = mR2_F[:ic]
    
    # compute total variance explained by all factors
    R2_T = np.sum(mR2_F)
    
    # sort series by marginal R-squared for each factor
    vals = np.sort(mR2)[::-1]
    ind = np.argsort(mR2)[::-1]
    
    # get top 10 series that load most heavily on each factor and the 
    # corresponding marginal R-squared values
    for i in range(ic):
        t10_s.append(series[ind[:10,i]])
        t10_mR2[:,i] = vals[:10,i]
        
    return R2,mR2,mR2_F,R2_T,t10_s,t10_mR2



def DickeyFullerTest(rawdata):
    """
    This function determines if series in a dataset are stationary
    INPUT: 
        rawdata = raw data matrix with series as columns
    OUTPUT:
        tcode = list of numbers with as many entries as series,
        an entry of 5 indicates that series is not stationary
        an entry of 1 indicates that series is stationary
        determination based on Dickey-Fuller Test Statistic with 5% critical region
    """
    tcode = []
    
    for i in range(rawdata.shape[1]):
        test_series = rawdata[:,i]
        adf_test = adfuller(test_series)
        if adf_test[0] < adf_test[4]['5%']:
            tcode.append(1)
        else:
            tcode.append(5)
            
    return tcode
                


def RemoveNaNColumns(data, series):
    """
    removes all columns that contain nothing but nan entries
    OUTPUT:
        new data frame, number of columns removed
    """
    bad_ind = np.where(np.sum(np.isnan(data), axis = 0) == data.shape[0])[0]
    n = len(bad_ind)
    full_ind = np.arange(0,data.shape[1],1)
    good_ind = np.delete(full_ind, bad_ind)
    good_series = series[good_ind]
    
    return data[:,good_ind], n, good_series
