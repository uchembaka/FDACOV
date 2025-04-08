function [R, sig2, OUTs] = LRCov(data, r, p, lambda, basis_type, omega, isSparse, mu_nbasis, nRegGrid, W, dispIter, LSQ)

% LRCov function for estimating smooth covariance surface,

% inputs
% data: \sum_i^n m_i x 3 matrix (obsID, time, observation) 

% r: Rank size

% p: Number of basis functions

% lambda: Smoothing parameter

% basis_type: Type of basis function. Can be 'bspline' or 'fourier'.
% (Default is bspline).

% omega: For selecting p automatically. 

% isSparse: Logical for if data is sparse;

% mu_nbasis: Number of basis used to estimate the mean function.

% nRegGrid: Output fixed grid size if data is irregular.

% W: Weights for improving estimate. The default is Null.

% dispIter: Logical for displaying optimization iteration.

% LSQ: Logical to use Least Squares loss for sparse and/or irregular data

if nargin < 2 || isempty(r)
    r = [];
end

if nargin < 3 || isempty(p)
    p = [];
end

if nargin < 4 || isempty(lambda)
    lambda = [];
end

if nargin < 5 || isempty(basis_type)
    basis_type = 'bspline';
end

if nargin < 6 || isempty(omega)
    omega = 0.01;
end

if nargin < 7 || isempty(isSparse)
    isSparse = checkSparse(data);
end

if nargin < 8 || isempty(mu_nbasis)
    mu_nbasis = [];
end

if nargin < 9 || isempty(nRegGrid)
    nRegGrid = 50;
end

if nargin < 10 || isempty(W)
    W = [];
end

if nargin < 11 || isempty(dispIter)
    dispIter = 0;
end

if isSparse
    M = nRegGrid;
    if nargin < 12 || isempty(LSQ)
        LSQ = 1;
    end
else
    M = length(unique(data(:,2)));
    LSQ = 1;
end

if dispIter
    iter = 'iter';
else
    iter = 'off';
end

options = optimset('Large',  'off', 'Display', iter, ...
'TolFun', 1e-6, 'GradObj','on', 'Hessian', 'off',   ...
'maxit',  2000,  'DiffMaxChange', 1e-6, 'UseParallel', 1);

grid = linspace(0, 1, M)';
[S, SiCell, SiID, mu, p, basis, penmat, W] = LR_pars_init(data, p, basis_type, omega, isSparse, grid, mu_nbasis, W, LSQ);

if isSparse
    if LSQ
        [R, cvec, r, GCV, W, lambda] = fitR_spLSQ(data, SiCell, S, SiID, grid, r, p, basis, lambda, penmat, omega, W, options );
    else
        [R, cvec, r, GCV, lambda] = fitR_spLLK(data, SiCell, S, grid, r, p, basis, lambda, penmat, omega, W, options );
    end
else
    [R, cvec, r, GCV, lambda] = fitR(S, data, M, r, p, basis, lambda, penmat, options, omega);
end

sig2 = exp(cvec(end));

OUTs.GCV = GCV;
OUTs.coeffs = cvec;
OUTs.r = r;
OUTs.p = p;
OUTs.mu = mu;
OUTs.RRTx = R*R';
OUTs.RRTy = R*R' + sig2*eye(M);
OUTs.evalGrid = grid;
OUTs.lambda = lambda;
OUTs.isSparse = isSparse;

if isSparse && LSQ
    OUTs.W = W;
end

end

function isSparse = checkSparse(data)
Eta = 0;
IDs = unique(data(:,1));
n = length(IDs);
M = length(unique(data(:,2)));
for i = 1:n
    datai = data(data(:,1) == IDs(i),:);
    Eta = Eta+size(datai, 1);
end
isSparse = Eta/M/n <= 0.75;
end


function R = get_R(cvec, B, M, r,p)
R = zeros(M, r);
indB = cumsum([1, repelem(p,r)]);
for i = 1:r
    C = cvec(indB(i):i*p);
    R(:,i) = B*C;
end
end
