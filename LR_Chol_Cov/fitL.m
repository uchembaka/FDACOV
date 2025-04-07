function[L, cvec, GCV,  RGCV, aGCV, aRGCV, trH, trHd] = fitL(S,cvec0, B, Rmat, k, lambda1, lambda2, BW, err, dispIter)
M = size(S,1);
ind_c = [0, cumsum(k)];

if dispIter
    iter = 'iter';
else 
    iter = 'off';
end

options=optimoptions('fminunc','Algorithm','quasi-newton','Display', iter, ...
'TolFun', 1e-6, 'GradObj','on', 'Hessian', 'off',   ...
 'MaxIter',  2000, 'MaxFunctionEvaluations', 1000000);
try 
    cvec = fminunc(@Chol_loss, cvec0, options, S, lambda1.*ones(M,1), lambda2, k, B, Rmat, BW, err);
catch
    cvec0 = jitter(cvec0);
    cvec = fminunc(@Chol_loss, cvec0, options, S, lambda1.*ones(M,1), lambda2, k, B, Rmat, BW, err);
end

L = getL(cvec, B, BW, M, ind_c);
[GCV, RGCV, aGCV, aRGCV, trH, trHd] = RoGCV(B, lambda1, lambda2, cvec, Rmat, M, S, L, ind_c, k, BW, err);

% if err % refine estimate
%     dind = cumsum(k)+1;dind(end) = [];
%     CL = cvec; CL(dind) = log(0.5);
%     L = getL(fminunc(@Chol_loss, CL(1:end-1), options, L*L', 0.*ones(M,1), 0, k, B, Rmat, BW, 0), B, BW, M ,ind_c);
% end

end



function L = getL(cvec, B, BW, M, ind_c)
L=zeros(M,M);
for i = 1:M
    Lbase = min(BW+i-1, M);
    C = cvec((ind_c(i)+1):ind_c(i+1));
    C(1) = exp(C(1));
    L(i:Lbase, i) = B{i}*C;
end
end

function vech = vech(mat)
    n = size(mat, 2);
    vech = mat(tril(ones(n),0)==1);
end


function [GCV, RGCV, aGCV, aRGCV, trH, trHd] = RoGCV(B, lambda, lambda2, cvec, Rmat, M, S, L, ind_c, k, BW, err)
trH = 0; trH2 = 0; nlpts = (M.^2+M)/2;
if nlpts < 100
    gamma = .2;
else
    gamma = .3;
end

for i = 1:M
    Lbase = min(BW+i-1, M);
    C = cvec((ind_c(i)+1):ind_c(i+1));
    C(1) = exp(C(1));
    L(i:Lbase, i) = B{i}*C;

    wLambda = ((abs((k(i)/k(1))-1)+1)*lambda);
    Hi = B{i}*((B{i}'*B{i} + wLambda.* Rmat{i})\B{i}');
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

MSE = (sum(sum((vech(S)-vech(Shat)).^2))) / nlpts;
GCV = MSE / (1-1*(trH/nlpts)).^2;
RGCV = (gamma+(1-gamma)*trH2)*GCV;

aGCV = MSE / (1-1*(trHd/(M-1))).^2;
aRGCV = (gamma+(1-gamma)*trace(Hd^2))*aGCV;
end
