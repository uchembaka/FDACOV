function [f,DLoss] = Chol_loss(cvec, S, lambda1, lambda2, k, B, Rmat, BW, err)
M = size(S,2);
L = zeros(M,M);
ind_c = [0, cumsum(k)];
pen = 0;
DC=zeros(M,1);
wLambda = zeros(length(cvec)-1,1);
ind_c1 = ind_c+1;
cvec(ind_c1(1:end-1)) = min(50, cvec(ind_c1(1:end-1)));

for i = 1:M
    Lbase = min(BW+i-1, M);
    C = cvec((ind_c(i)+1):ind_c(i+1));
    C(1) = exp(C(1));
    L(i:Lbase, i) = B{i}*C;
    wLambda((ind_c(i)+1):ind_c(i+1)) = (abs((k(i)/k(1))-1)+1)*lambda1(i);
    pen = pen+(abs((k(i)/k(1))-1)+1)*lambda1(i).*((C'*Rmat{i}*C));
    DC(i) = C(1);
end

if err
    sig2 = exp(cvec(end));
else
    sig2 = 0;
end

Shat = L*L' + sig2*eye(M);
Res = (S - Shat);
f = sum(sum(Res.^2))+pen+lambda2.*(DC(2:end)'*Rmat{M+1}*DC(2:end));

if nargout > 1
    K = length(cvec);
    DLoss = zeros(1,K);
    diag_ind = ind_c+1;
    for j = 1:M
        Lbase = min(BW+j-1, M);
        Bj = B{j};
        Bj1 = Bj*(DC(j).*[1;zeros(size(Bj, 2)-1, 1)]);
        Bj = [Bj1, Bj(:,2:end)];
        LTDRDc = kron(Bj', L(j:Lbase, j)');
        LDRTDc = kron(L(j:Lbase, j)', Bj');
        
        gmat = LTDRDc + LDRTDc;
        vRes = Res(j:Lbase, j:Lbase);
        DLoss(1, (ind_c(j)+1):ind_c(j+1)) = -2.*(gmat*vRes(:));
    end
    if err
        exp1 = ones(K-1, 1);
        exp1(diag_ind(1:end-1)) = exp(cvec(diag_ind(1:end-1))); 
        fullRmat = blkdiag(Rmat{1:M});
        expCvec = cvec(1:end-1); expCvec(diag_ind(1:end-1)) = exp(cvec(diag_ind(1:end-1)));
        DLoss(1,1:end-1) = DLoss(1, 1:end-1) + (wLambda.* (((fullRmat+fullRmat')*expCvec).*exp1))';
        DLoss(1, diag_ind(2:end-1)) = DLoss(1, diag_ind(2:end-1)) +  (lambda2.*(((Rmat{M+1}+Rmat{M+1}')*DC(2:end)).*DC(2:end)))';
        DLoss(end) = sum(sum(2.*Res.*(-sig2*eye(M))));
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