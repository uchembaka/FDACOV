function mu = Chol_get_mean(data, knots, lambdaVec)
M = size(data, 2);
if nargin < 2
    knots = M;
end
if nargin < 3
    lambdaVec = linspace(1e-3, 1, 20);
end
grid = linspace(0,1, M);
raw_mean = mean(data,1,'omitnan');
wk_ind = ~isnan(raw_mean);

basis = create_bspline_basis([0,1],knots);
nl = length(lambdaVec);
gcv_vec = zeros(nl,1);
for i = 1:nl
    tmp_fdPar = fdPar(basis, 2, lambdaVec(i));
    [~, ~, gcv] = smooth_basis(grid(wk_ind), raw_mean(wk_ind), tmp_fdPar);
    gcv_vec(i) = gcv;
end
lam = lambdaVec(gcv_vec == min(gcv_vec));
fdParobj = fdPar(basis, 2, lam);
fdobj = smooth_basis(grid(wk_ind), raw_mean(wk_ind), fdParobj);
mu = eval_basis(grid, basis)*getcoef(fdobj);
end
