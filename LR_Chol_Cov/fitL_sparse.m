function[L, cvec, GCV, RGCV, aGCV, aRGCV, trH, trHd] = fitL_sparse(S,cvec0, Sind, grid, B, Rmat, k, lambda1, lambda2, BW, err, dispIter)
M = length(grid);
ind_c = [0, cumsum(k)];
L=zeros(M,M);

if dispIter
    iter = 'iter';
else 
    iter = 'off';
end

options=optimoptions('fminunc','Algorithm','quasi-newton','Display', iter, ...
'TolFun', 1e-6, 'GradObj','on', 'Hessian', 'off',   ...
 'MaxIter',  2000, 'MaxFunctionEvaluations', 1000000);

try 
    cvec = fminunc(@Chol_sp_loss, cvec0, options, S, grid, Sind, lambda1.*ones(M,1), lambda2, k, B, Rmat, BW, err, 1);
catch
    cvec0 = jitter(cvec0);
    cvec = fminunc(@Chol_sp_loss, cvec0, options, S, grid, Sind, lambda1.*ones(M,1), lambda2, k, B, Rmat, BW, err, 1);
end

for i = 1:M
    Lbase = min(BW+i-1, M);
    C = cvec((ind_c(i)+1):ind_c(i+1));
    C(1) = exp(C(1));
    L(i:Lbase, i) = B{i}*C;
end
[GCV, RGCV, aGCV, aRGCV, trH, trHd] = RoGCV(B, lambda1, lambda2, cvec, Rmat, M, S, L, ind_c, Sind, k, BW, err, 1);

end


% Aux functions

function [GCV, RGCV, aGCV, aRGCV, trH, trHd] = RoGCV(B, lambda, lambda2, cvec, Rmat, M, S, L, ind_c, Sind, k, BW, err, weight_lambda)
trH = 0; trH2 = 0; nlpts = (M.^2+M)/2; npts = length(S);

if npts < 100
    gamma = .2;
else
    gamma = .3;
end

for i = 1:M
    Lbase = min(BW+i-1, M);
    C = cvec((ind_c(i)+1):ind_c(i+1));
    C(1) = exp(C(1));
    L(i:Lbase, i) = B{i}*C;

    if weight_lambda
        wLambda = ((abs((k(i)/k(1))-1)+1)*lambda);
    else
        wLambda = lambda;
    end
    Hi = B{i}*((B{i}'*B{i}+ wLambda.* Rmat{i})\B{i}');
    trH = trH + trace(Hi);
    trH2 = trH2 + trace(Hi^2);

end
I = eye(M-1);
Hd = inv(I+ lambda2.* Rmat{M+1});
trHd = trace(Hd);

if err
    sig2 = exp(cvec(end));
else
    sig2 = 0;
end

Shat = L*L' + sig2*eye(M);
vShat = vech(Shat);
vShat = vShat(Sind);
Res = (S-vShat);

MSE = sum(Res.^2)/npts;
GCV = MSE / (1-(trH/npts)).^2;
RGCV = (gamma+(1-gamma)*trH2)*GCV;

aGCV = MSE / (1-1*(trHd/(M-1))).^2;
aRGCV = (gamma+(1-gamma)*trace(Hd^2))*aGCV;
end


