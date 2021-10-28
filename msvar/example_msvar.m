%% SIMPLE EXAMPLE - MSVAR(2,2)
%  the basic model is a 2 dimensional MSVAR 
%      Y_{t} = mu_{s_t} A_{s_t} * Y_{t-1} + Sigma_{s_t} * epsilon_{s_t}

%% SETUP

% number of state variables 
N = 2; 

% number of MS states 
K = 2; 

% basis vectors 
e = eye(N);

% autoregressive matrix (N x N x K)
AMat = zeros(N,N,K); 
AMat(:,:,1) = diag([0.469; 0.92]);
AMat(:,:,2) = diag([0.469; 0.92]);

% means (N x K)
muYMat = zeros(N,K); 
muYMat(:,1) = 2.*[0.0125; 0.085];
muYMat(:,2) = [0.0125; 0.085];

% shocks
SigMat = zeros(N,N,K); 
SigMat(:,:,1) = diag([0.001, 0.02]); 
SigMat(:,:,2) = 2.*diag([0.001, 0.02]); 

% transition matrix
P = zeros(K);

% Regime 1 transition probabilities (crisis times)
crisisDuration = 5*252;
P(1,1) = 1-1/crisisDuration;
P(1,2) = 1-P(1,1);

% Regime 2 transition probabilities (normal regime)
nDuration =  5*252;
P(2,2) = 1-1/nDuration;
P(2,1) = 1-P(2,2);

% stationary distribution
[V,D] = eigs(P');
DVec  = diag(D);
piVec = V(:, abs(DVec-1) == min(abs(DVec-1)));
piVec = piVec ./ sum(piVec);

%% CONDITIONAL VARIANCE-COVARIANCE MOMENTS

% conditional variance-covariance matrix
% note: vec(varY) = (I - kron(A,A))^{-1} * vec(SS')
A1 = AMat(:,:,1); A2 = AMat(:,:,2);
S1 = SigMat(:,:,1) * SigMat(:,:,1)'; S2 = SigMat(:,:,2) * SigMat(:,:,2)';
VarY1 = reshape((eye(N*N) - kron(A1, A1)) \ S1(:), N,N);
VarY2 = reshape((eye(N*N) - kron(A2, A2)) \ S2(:), N,N);

% predefine useful selection vectors
S1 = e(:,1);
S2 = e(:,2);

% variance and standard deviation of state variable 1
var_y1_s1 = S1' * VarY1 * S1; 
std_y1_s1 = sqrt(var_y1_s1);
var_y1_s2 = S1' * VarY2 * S1; 
std_y1_s2 = sqrt(var_y1_s2);

% conditional variance and standard deviation of state variable 2
var_y2_s1 = S2' * VarY1 * S2; 
std_y2_s1 = sqrt(var_y2_s1);
var_y2_s2 = S2' * VarY2 * S2; 
std_y2_s2 = sqrt(var_y2_s2);

% conditional covariance of state variables
cov_y1_y2_s1 = S1' * VarY1 * S2; 
cov_y1_y2_s2 = S1' * VarY2 * S2; 

% conditional correlation of state variables
corr_y1_y2_s1 = cov_y1_y2_s1 / (std_y1_s1 * std_y2_s1);
corr_y1_y2_s2 = cov_y1_y2_s2 / (std_y1_s1 * std_y2_s2);

%% UNCONDITIONAL VARIANCE/COVARIANCE MOMENTS - Bianchi (2015)

% first order moments
H = P';  % eq. (4)

C = blkdiag(muYMat(:,1), muYMat(:,2));
Omega = blkdiag(AMat(:,:,1), AMat(:,:,2)) * kron(H,eye(N));
q = (eye(N*K) - Omega) \ (C * piVec);         % eq. (3)
OmegaTilde = [Omega, C*H; zeros(K, N*K), H];  % eq. (5)

w = repmat(eye(N),1,K);
mu = w * q;
mu2 = mu * mu';

wtilde = [w, zeros(N,K)];
qtilde = [q; piVec];
mutp1 = wtilde * OmegaTilde * qtilde;

c1 = muYMat(:,1); c2 = muYMat(:,2);
cc1 = kron(c1,c1); cc2 = kron(c2,c2);
V1 = SigMat(:,:,1); V2 = SigMat(:,:,2);
VV1 = kron(V1,V1); VV2 = kron(V2,V2);
A1 = AMat(:,:,1); A2 = AMat(:,:,2);
AA1 = kron(A1,A1); AA2 = kron(A2,A2);
DAC1 = kron(A1,c1) + kron(c1,A1); DAC2 = kron(A2,c2) + kron(c2,A2);
Ik = eye(N);

Xi = blkdiag(AA1,AA2) * kron(H,eye(N^2));
VV = blkdiag(VV1 * Ik(:), VV2 * Ik(:));
DAC = blkdiag(DAC1,DAC2) * kron(H,eye(N));
cc = blkdiag(cc1,cc2);
Vc = VV + cc;

Q = (eye(2*N^2) - Xi) \ (DAC * q(:) + Vc * piVec); % eq (7)
W = repmat(eye(N*N),1,K);
M = W * Q;

vecVar = M - mu2(:);

% unconditional variance-covariance matrix
VarY = reshape(vecVar,N,N);

% unconditional variance of state 1
VarY1 = S1' * VarY * S1; 
VarY2 = S2' * VarY * S2; 

%% UNCONDITIONAL AUTOCORRELATION

% see Proposition 3 from Bianchi
IA1 = kron(eye(N), A1); IA2 = kron(eye(N), A2);
Ic1 = kron(eye(N), c1); Ic2 = kron(eye(N), c2);
Xi1l = blkdiag(IA1, IA2) * kron(H,eye(N*N));
Ic = blkdiag(Ic1, Ic2);
Xi1tilde = [Xi1l, Ic*kron(H,eye(N)); zeros(N*2,N*4), kron(H,eye(N))];
Qttp1 = Xi1tilde * [Q; q]; % eq. 13

% first order autocovariance matrix
AC1 = reshape(Qttp1(1:N*2) + Qttp1((N*2+1):(N*4)), N,N) - mutp1 * mu';

% correlation
Acorr1 = AC1 ./ sqrt(diag(VarY) * diag(VarY)');

% unconditional first-order autocorrelation
AC1_y1 = S1' * Acorr1 * S1 ;
AC1_y2 = S2' * Acorr1 * S2 ;

