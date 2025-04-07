function [mu, fits, basis, fdobj] = LR_get_mean(data, estGrid, type, knots, lambdaVec)

if nargin < 4
    knots = length(estGrid);
end

if nargin < 5
    lambdaVec = linspace(1e-4, 1, 20);
end

grid = estGrid;
raw_data = data(:,3);
wk_ind = data(:,2);

if strcmp(type, 'bspline')
    basis = create_bspline_basis([0,1],knots);
else
    basis = create_fourier_basis([0,1], knots);
end
nl = length(lambdaVec);
gcv_vec = zeros(nl,1);
for i = 1:nl
    tmp_fdPar = fdPar(basis, 2, lambdaVec(i));
    [~, ~, gcv] = smooth_basis(wk_ind, raw_data, tmp_fdPar);
    gcv_vec(i) = gcv;
end
lam = lambdaVec(gcv_vec == min(gcv_vec));
fdParobj = fdPar(basis, 2, lam);
fdobj = smooth_basis(wk_ind, raw_data, fdParobj);
mu = eval_basis(grid, basis)*getcoef(fdobj);
fits = eval_basis(wk_ind, basis)*getcoef(fdobj);
end