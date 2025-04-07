function [L, sig2, OUTs] = PDCov(data, lambda1, lambda2, k, covmat, BW, err, diag_cvec0, isSparse, dispIter)
% PDCov function for estimate smooth positive definite covariance matrix,

% inputs
% data: nxM matrix with n subjects and M size grid points. The matrix can
% be sparse

% k: Number of basis of the first column of the Cholesky factor

% covmat: Logical for if data is a covariance matrix. Default is 0 (data
% matrix)

% BW: Specify bandwidth for banded estimate. Default is M

% err: Logical for if data includes error. Default is 1

% diag_cvec0: Initial value for diagonal coefs. If a single number, it's
% repeated M times otherwise should be vector of size M. Default is
% log(0.01)

% isSparse: Logical for if datamat is sparse;

% dispIter: Logical for displaying optimization iteration



if nargin < 2 || isempty(lambda1)
    lambda1 = 10.^(-7:1);
end

if nargin < 3 || isempty(lambda2)
    lambda2  = 10.^(2:5);
end

M = size(data, 2);

if nargin < 4 || isempty(k)
    k = [];
end

if nargin < 5 || isempty(covmat)
    covmat = 0;
end

if nargin < 6 || isempty(BW)
    BW = M;
end

if nargin < 7 || isempty(err)
    err = 1;
end

if nargin < 8 || isempty(diag_cvec0)
    diag_cvec0 = [];
end

if nargin < 9 || isempty(isSparse)
    isSparse = checkSparse(data, covmat);
end

if nargin < 10 || isempty(dispIter)
    dispIter = 0;
end

if BW == 0
    BW = M;
end

if isempty(diag_cvec0)
    [S, mu, B, Rmat, cvec0, k, grid, Sind, isSparse, dind] = Chol_pars_init(data, k, covmat, BW, err, isSparse);
else
    [S, mu, B, Rmat, cvec0, k, grid, Sind, isSparse , dind] = Chol_pars_init(data,k, covmat, BW, err, isSparse);
    if isscalar(diag_cvec0)
        diag_cvec0 = repelem(diag_cvec0, M);
    end
    if length(diag_cvec0) ~= M
        error('diag ncoefs ~= length diag_coefs');
    end
    cvec0([1,dind]) = diag_cvec0; 
end

GCVs = zeros(length(lambda1), 1);
GCVs2 = zeros(length(lambda2), 1);
cvecCell = cell(max(length(lambda1), length(lambda2)), 1);

if ~err
    disp('smoothing parameters set to zero')
    lambda1 = 0; lambda2 = 0;
end

if length(lambda1) > 1
    if isempty(gcp('nocreate'))
        parpool;
    end
    disp('Selecting Lambda1')
    if length(lambda2) > 1
        l1_tmp = 1e-7;
    else
        l1_tmp = lambda2;
    end

    if ~isempty(gcp('nocreate'))
        parfor i = 1:length(lambda1)
            if isSparse
                [~, cvec, ~, gcv] = fitL_sparse(S,cvec0, Sind, grid, B, Rmat, k, lambda1(i), l1_tmp, BW, err, dispIter);
            else
                [~, cvec, ~, gcv] = fitL(S, cvec0, B, Rmat, k, lambda1(i), l1_tmp, BW, err, dispIter);
            end
            GCVs(i) = gcv;
            cvecCell{i} = cvec;
        end
    else
        warning('Parpool is off')
        for i = 1:length(lambda1)
            if isSparse
                [~, cvec, ~, gcv] = fitL_sparse(S,cvec0, Sind, grid, B, Rmat, k, lambda1(i), l1_tmp, BW, err, dispIter);
            else
                [~, cvec, ~, gcv] = fitL(S, cvec0, B, Rmat, k, lambda1(i), l1_tmp, BW, err, dispIter);
            end
            GCVs(i) = gcv;
            cvecCell{i} = cvec;
        end        
    end
    minGCV = GCVs == min(GCVs);
    lambda1 = lambda1(minGCV);
    lambda1 = lambda1(1);
    % cvec = cvecCell{minGCV};
end

if length(lambda2) > 1
    if isempty(gcp('nocreate'))
        parpool;
    end
    disp('Selecting Lambda2')    
    if ~isempty(gcp('nocreate'))
        parfor i = 1:length(lambda2)
            if isSparse
                [~, cvec, ~, ~, ~, gcv2] = fitL_sparse(S,cvec0, Sind, grid, B, Rmat, k, lambda1, lambda2(i), BW, err, dispIter);
            else
                [~, cvec, ~, ~, ~, gcv2] = fitL(S, cvec0, B, Rmat, k, lambda1, lambda2(i), BW, err, dispIter);
            end
            GCVs2(i) = gcv2;
            cvecCell{i} = cvec;
        end
    else
        warning('Parpool is off')
        for i = 1:length(lambda2)
            if isSparse
                [~, cvec, ~, ~, ~, gcv2] = fitL_sparse(S,cvec0, Sind, grid, B, Rmat, k, lambda1, lambda2(i), BW, err, dispIter);
            else
                [~, cvec, ~, ~, ~, gcv2] = fitL(S, cvec0, B, Rmat, k, lambda1, lambda2(i), BW, err, dispIter);
            end
            GCVs2(i) = gcv2;
            cvecCell{i} = cvec;
        end        
    end
    % minGCV = GCVs2 == min(GCVs2);
    lambda2 = lambda2(GCVs2 == min(GCVs2));
    lambda2 = lambda2(1);
    % cvec = cvecCell{minGCV};
end

if isSparse
    [L, cvec, ~, rgcv, ~, argcv, trH, trHd] = fitL_sparse(S, cvec0, Sind, grid, B, Rmat, k, lambda1, lambda2, BW, err, dispIter);
else
    [L, cvec, ~, rgcv, ~, argcv, trH, trHd] = fitL(S, cvec0, B, Rmat, k, lambda1, lambda2, BW, err, dispIter);
end


if err
    sig2 = exp(cvec(end));
    % refine non-informative eigenvalues of L
    CL = cvec0; CL(dind) = log(0.5);
    L = fitL(L*L', CL(1:end-1), B, Rmat, k, 0,0,BW, 0, 0);
else
    sig2 = 0;
end

OUTs.GCV = rgcv;
OUTs.aGCV = argcv;
OUTs.k = k;
OUTs.coeffs = cvec;
OUTs.lambda1 = lambda1;
OUTs.lambda2 = lambda2;
OUTs.mu = mu;
% OUTs.GCV1 = GCVs;
% OUTs.GCV2 = GCVs2;
OUTs.DF1 = trH;
OUTs.DF2 = trHd;
OUTs.S = S;
LLTx = L*L';
OUTs.LLTx = LLTx;
OUTs.LLTy = LLTx + sig2*eye(M);
OUTs.Sparse = isSparse;

if isSparse && ~covmat
    OUTs.Sind = Sind;
    n = size(data, 1);
    xhat = zeros(n, M);
    LLT = LLTx;
    LLTy = OUTs.LLTy;
    for i = 1:n
        Yi = data(i,:);
        ind = ~isnan(Yi);
        Cxy = LLT(:,ind);
        Cy = LLTy(ind, ind);
        xhat(i,:) = Cxy*inv(Cy)*(Yi(ind)' - mu(ind)) + mu;
    end
    OUTs.xHat = xhat;
end


end

function isSparse = checkSparse(data, covmat)
if covmat
    isSparse = any(any(isnan(data)));
else
    Eta = 0;
    [n, M] = size(data);
    for i = 1:n
        Eta = Eta+sum(~isnan(data(i,:)));
    end
    isSparse = Eta/M/n <= 0.75;
end
end
