function [S, mu, B, Rmat, cvec0, k, grid, Sind, isSparse, dind] = Chol_pars_init(data, k, covmat, BW, err, isSparse)
M = size(data,2);
grid = linspace(0, 1, M);

if isSparse
    if covmat
        [S, Sind] = get_data_vec(data, grid', [], 1, 1);
        mu = [];
    else
        mu = Chol_get_mean(data, k);
        [S, Sind] = get_data_vec(data, grid', mu, 1, 0);
    end
    if isempty(k)
        k = min(15, BW);
    end
    
else
    if covmat
        S = data;
        mu = []; Sind = [];
    else
        S = cov(data - mean(data, 'omitnan'), 'partialrows');
        mu = Chol_get_mean(data, k);
        Sind = [];
        isSparse = any(any(isnan(S)));
        if isSparse
            [S, Sind] = get_data_vec(data, grid', [], 1, 1);
        end
    end
    
    if isempty(k)
        k = getK(S, err);
    end
end

fullPar = (k == BW); % full parameterization of L
k = [repelem(k, M-BW), round((flip(1:BW)/(BW))*k)];
k = max(k, 1);
if (M > 3)
    k(k < 4) = 3;
    k(end-2:end) = [3,2,1];
end


B = cell(M,1);
Rmat = cell(M+1, 1);
for i = 1:M
    gbase = min(BW+i-1, M);
    grid_range = grid(i:gbase);
    
    if fullPar
        ln = length(grid(i:gbase));
        B{i} = sparse(eye(ln));
        if k(i) > 2
            Rmat{i} = diff(eye(ln),2)'*diff(eye(ln),2);
        else
            Rmat{i} = zeros(k(i),k(i));
        end
    else
        if i < M
            brng = [grid(i), grid(gbase)];
            basisobj = create_bspline_basis(brng,k(i));
            B{i} = eval_basis(grid_range,basisobj);
        else
            B{i} = 1;
        end
        
        if k(i) > 4
            Rmat{i}     = eval_penalty(basisobj, 2);
        elseif k(i) > 2 && k(i) < 5 
            Btmp=eval_basis(grid_range,basisobj,2);
            R_a = zeros(size(Btmp, 2));
            for idx = 1:size(Btmp, 2)
                R_a(idx, :) = trapz(grid(i:M), Btmp(:, idx) .* Btmp);
            end
            Rmat{i}= R_a;
        else
            Rmat{i} = zeros(k(i),k(i));
        end
    end
end
Rmat{M+1} = diff(eye(M-1),2)'*diff(eye(M-1),2);

% Initialize coefs
rng("default")
cvec0 = repelem(0.01, sum(k))';
dind = cumsum(k)+1;
dind(end) = [];
cvec0([1, dind]) = log(err * 0.01 + ~err * 0.5);
if err
    cvec0 = [cvec0; log(0.1)];
end


end


