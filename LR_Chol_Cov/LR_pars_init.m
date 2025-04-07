function [S, SiCell, SiID, mu, p, basis, penmat, W] = LR_pars_init(data, p, basis_type, omega, isSparse, evalGrid, mu_nbasis, W, LSQ)

if isempty(mu_nbasis)
    mu_nbasis = length(evalGrid);
end
M = length(evalGrid);
[mu, fits] = LR_get_mean(data, evalGrid, basis_type, mu_nbasis);
cY = data(:,3) - fits;
data = [data,cY];

if isSparse
    if isempty(W) || LSQ
        [SiCell, S, SiID, W] = get_S(data);
    else
        [SiCell, S] = get_S(data);
        W = []; SiID = [];
    end
else
    data_mat = iDTY2mat(data(:, [1,2,4]));
    S = cov(data_mat,"partialrows");
    SiCell = []; SiID = [];
end


if isempty(p) || length(p) > 1
    disp('nbasis selection');
    if isempty(p)
        p = 5:round(0.5*M);
    end
     mrsse = zeros(length(p),1);
     prev_mrsse = 0;
    for i = 1:length(p)
        mui = LR_get_mean(data, evalGrid, basis_type ,p(i), 10.^(-6:1));
        mrsse(i) = rmse(mui,mu, 'all');
        curr_mrsse = mrsse(i);
        if abs(prev_mrsse - curr_mrsse) <= omega || i == length(p)
            p = p(i);
            break
        else
            prev_mrsse = mrsse(i);
        end 
    end
    mu = mui;
    p = min(p+1, M);
end

if strcmp(basis_type, 'fourier')
    if mod(p,2) == 0
        p = p-1;
        p = max(p,3);
    end
end
disp(['p = ',num2str(p)]);

if strcmp(basis_type, 'bspline')
    basis = create_bspline_basis([0,1], p);
else
    basis = create_fourier_basis([0,1], p);
end
penmat = eval_penalty(basis, 2);

end

function [SiCell, S, SiID, W, Mi, xy, vecS] = get_S(data)
IDs = unique(data(:,1));
n = length(IDs);
SiCell = cell(n,1);
W = cell(n,1);
S = [];
SiID = [];
Mi = zeros(n,1);
xy = zeros(0,2); vecS = [];
for i = 1:n
    ind = find(data(:,1) == IDs(i));
    cYi = data(ind, 4);
    SiCell{i} = kron(cYi, cYi');   
    S = [S; vech(SiCell{i})];
    vecS = [vecS; SiCell{i}(:)];
    mi = length(ind);
    xy = [xy; [repelem(data(ind, 2), mi),repmat(data(ind, 2), mi, 1)]];
    Mi(i) = mi;
    W{i} = diag(repelem(1, (mi^2+mi)/2));
    SiID = [SiID, repelem(i,  (mi^2+mi)/2)];
end
end
