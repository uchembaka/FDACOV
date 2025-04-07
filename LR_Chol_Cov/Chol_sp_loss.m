function [f,DLoss] = Chol_sp_loss(cvec, S, grid, Sind, lambda1, lambda2, k, B, Rmat, BW, err, weight_lambda)
M = length(grid);
L = zeros(M,M);
ind_c = [0, cumsum(k)];
pen = 0;
DC=zeros(M,1);
nlpts = (M^2+M)/2;
[~, Lmat] = trilByVecCol(1:nlpts, M);
wLambda = zeros(length(cvec)-1,1);

for i = 1:M
    Lbase = min(BW+i-1, M);
    C = cvec((ind_c(i)+1):ind_c(i+1));
    C(1) = exp(C(1));
    L(i:Lbase, i) = B{i}*C;
    if weight_lambda
        wLambda((ind_c(i)+1):ind_c(i+1)) = (abs((k(i)/k(1))-1)+1)*lambda1(i);
    else
        wLambda((ind_c(i)+1):ind_c(i+1)) = lambda1(i);
    end
    pen = pen+(abs((k(i)/k(1))-1)+1)*lambda1(i).*((C'*Rmat{i}*C));
    DC(i) = C(1);
end

if err
    sig2 = exp(cvec(end));
else
    sig2 = 0;
end

Shat = L*L' + sig2*eye(M);
vShat = vech(Shat);
vShat = vShat(Sind);
Res = (S-vShat);
f = sum(Res.^2)+pen+lambda2.*(DC(2:end)'*Rmat{M}*DC(2:end));

if nargout > 1
    K     = length(cvec);
    DLoss = zeros(1,K);
    diag_ind = ind_c+1; diag_ind(1:end-1);
    for j = 1:M
        Lbase = min(BW+j-1, M);
        Bj = B{j};
        Bj1 = Bj*(DC(j).*[1;zeros(size(Bj, 2)-1, 1)]);
        Bj = [Bj1, Bj(:,2:end)];
        LTDRDc = kron(Bj', L(j:Lbase, j)');
        LDRTDc = kron(L(j:Lbase, j)', Bj');
        
        gmat = LTDRDc + LDRTDc;
        
        curr_ind = vech(reshape(1:(Lbase-j+1)^2, Lbase-j+1, Lbase-j+1));
        vgmat = gmat(:,curr_ind);
        curr_dat_ind = vech(Lmat(j:Lbase, j:Lbase));
        curr_sind = ismember(curr_dat_ind, Sind);
        vgmat = vgmat(:, curr_sind);
        DLoss(1, (ind_c(j)+1):ind_c(j+1)) = -2.*(vgmat*Res(ismember(Sind, curr_dat_ind)));
    end

    if err
        exp1 = ones(K-1, 1);
        exp1(diag_ind(1:end-1)) = exp(cvec(diag_ind(1:end-1))); 
        fullRmat = blkdiag(Rmat{1:M});
        expCvec = cvec(1:end-1); expCvec(diag_ind(1:end-1)) = exp(cvec(diag_ind(1:end-1)));
        DLoss(1,1:end-1) = DLoss(1, 1:end-1) + (wLambda.* (((fullRmat+fullRmat')*expCvec).*exp1))';
        DLoss(1, diag_ind(2:end-1)) = DLoss(1, diag_ind(2:end-1)) +  (lambda2.*(((Rmat{M+1}+Rmat{M+1}')*DC(2:end)).*DC(2:end)))';
        vIsig2 = vech(tril(sig2*eye(M)));
        vIsig2 = vIsig2(Sind);
        DLoss(end) = -2.*sum( Res.*vIsig2);
    else
        exp1 = ones(K, 1);
        exp1(diag_ind(1:end-1)) = exp(cvec(diag_ind(1:end-1))); 
        fullRmat = blkdiag(Rmat{1:M});
        expCvec = cvec(1:end); expCvec(diag_ind(1:end-1)) = exp(cvec(diag_ind(1:end-1)));
        DLoss(1,1:end) = DLoss(1, 1:end) + (wLambda.* (((fullRmat+fullRmat')*expCvec).*exp1))';
        DLoss(1, diag_ind(2:end-1)) = DLoss(1, diag_ind(2:end-1)) +  (lambda2.*(((Rmat{M+1}+Rmat{M+1}')*DC(2:end)).*DC(2:end)))';
    end

end


end


function [LSigma, Lmat] = trilByVecCol(vec, M)
        LSigma = tril(ones(M));
        Lmat = LSigma;
        LSigma(LSigma == 1) = vec;
        Lmat(Lmat == 1) = 1:(M^2+M)/2;
end