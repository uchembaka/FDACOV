function [R, cvec, r, GCV, W, lambda] = fitR_spLSQ(data, SiCell, S, SiID, grid, r, p, basis, lambda, penmat, omega, W, options )

M = length(grid);
%rank selection
if isempty(r) || length(r) > 1
    disp('rank selection');
    if isempty(r)
        r = 1:round(0.2*M);
    end
     mrsse = zeros(length(r),1);
     prev_mrsse = 0;
    for i = 1:length(r)
        cvec0 = abs(rand(p*r(i)+1,1));
        cvec = fminunc(@loss, cvec0, options, SiCell, data, r(i), p, basis, penmat, 0, W);
        Shat = get_Shat(cvec,data,basis, r(i), p);
        mrsse(i) = rmse(Shat, S, 'all');
        curr_mrsse = mrsse(i);
        if abs(prev_mrsse - curr_mrsse) <= omega || i == length(r)
            r = r(i);
            break
        else
            prev_mrsse = mrsse(i);
        end 
    end
    disp(['r = ',num2str(r)]);
end

gcvs  = []; 
cvec = [];
%lambda selection
if isempty(lambda) || length(lambda) > 1

    if isempty(gcp('nocreate'))
        parpool;
    end
    disp('lambda selection ...');
    if isempty(lambda)
        lambda = 10.^linspace(-6, 1, 15);
    end
    gcvs = zeros(length(lambda),1);
    cvecGCV = cell(length(lambda), 1);
    if ~isempty(gcp('nocreate'))
        parfor i = 1:length(lambda)
            cvec0 = abs(rand(p*r+1,1));
            cvec = fminunc(@loss, cvec0, options, SiCell, data,r, p, basis, penmat, lambda(i), W);
            cvecGCV{i} = cvec;
            Shat = get_Shat(cvec,data,basis, r, p);
            SSE = sum((S-Shat).^2);
            trH = traceH(data, basis, penmat, lambda(i), r, W);
            gcvs(i) = (SSE*length( Shat)) / (length( Shat) - trH)^2;
        end
    else
        for i = 1:length(lambda)
            disp(['lambda ',num2str(i),'/',num2str(length(lambda))]);
            cvec0 = abs(rand(p*r+1,1));
            cvec = fminunc(@loss, cvec0, options, SiCell, data,r, p, basis, penmat, lambda(i), W);
            cvecGCV{i} = cvec;
            Shat = get_Shat(cvec,data,basis, r, p);
            SSE = sum((S-Shat).^2);
            trH = traceH(data, basis, penmat, lambda(i), r, W);
            gcvs(i) = (SSE*length( Shat)) / (length( Shat) - trH)^2;
        end
    end
    gcvs = round(gcvs,4);
    minGCV = find(gcvs == min(gcvs), 1,'first');
    lam = lambda(minGCV);
    cvec = cvecGCV{minGCV};
    if lam == lambda(end)
        warning('Last value in lambda grid selected')
    end
    lambda = lam;
    disp(['lambda_index = ',num2str(minGCV)]);
end

if isempty(cvec)
    cvec0 = abs(rand(p*r+1,1));
    cvec = fminunc(@loss, cvec0, options, SiCell, data,r, p, basis, penmat, lambda, W);
end

cvec = fminunc(@loss, cvec, options, SiCell, data,r, p, basis, penmat, lambda, W); % attempt to ensure convergence
R = get_R(cvec, basis, M, r, p);
Shat = get_Shat(cvec,data,basis, r, p);
if isempty(gcvs)
    SSE = sum((S-Shat).^2);
    trH = traceH(data, basis, penmat, lambda, r, W);
    GCV = (SSE*length( Shat)) / (length( Shat) - trH)^2;
else
    GCV = gcvs;
end
W = get_weight(Shat, S, SiID);

end


%Loss function 
function [f, g, h]= loss(cvec, SiCell, data, r, p, basis, P, lambda, W)
n = length(unique(data(:,1)));
indB = cumsum([1, repelem(p,r)]);
Pen = 0; f=0; m2 = 0;
if nargout > 1
    g = zeros(p*r+1,1);
    ind_c = cumsum([0, repelem(p,r)]);
end

if nargout > 2
    h = zeros(p*r+1, p*r+1);
    Dn = repelem(p,p*r);
end

for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    Ri = zeros(mi, r);
    wi = W{i};
    for j = 1:r
        C = cvec(indB(j):j*p);
        Ri(:,j) = Bi*C;
%         Pen = Pen+lambda.*(C'*P*C);
    end
    sig2 = exp(cvec(end));
    Sihat = Ri*Ri' + sig2*eye(mi);
    vSihat = vech(Sihat);
    vSi = vech(SiCell{i});
    Resi = wi*(vSi-vSihat);
    fi = sum(Resi.^2);
    f = f+fi/mi;
    m2 = m2 + mi;

    if nargout > 1
        gi = zeros(p*r+1,1);
%         fullR = [];
        vech_ind = vech(tril(reshape(1:mi^2, mi, mi)));
        for j = 1:r
            RTDRDc = kron(Bi', Ri(:,j)');
            RDRTDc = kron(Ri(:,j)', Bi');
            Dc_diff = - RDRTDc - RTDRDc;
            vech_Dc = Dc_diff(:,vech_ind)*wi;
            gi((ind_c(j)+1):ind_c(j+1)) = 2.*(vech_Dc*Resi);
%             fullR = blkdiag(fullR, P);
        end
%         gi(1:end-1)  = gi(1:end-1)+lambda.*(2.*fullR)*cvec(1:end-1);
        vIsig2 = vech(tril(sig2*eye(mi)));
        vIsig2 = wi*-vIsig2;
        gi(end) = 2.*(sum(Resi.*vIsig2));
        g = g+gi/mi;
    end

    if nargout > 2
        indBH  = repelem(1:r, p);
        hi = zeros(p*r, p*r);
        ind_1 = 1:mi; 
        for j = 1:p*r
            Dcvecj = zeros(Dn(j),1);
            Dcvecj(j-(ind_c(indBH(j)))) = 1;
            DRDcj = Bi*Dcvecj;
            DRTDcj = Dcvecj'*Bi';
            for k = 1:j
                if indBH(j) ~= indBH(k)
                    continue
                end
                Dcveck = zeros(Dn(k),1);
                Dcveck(k-(ind_c(indBH(k)))) = 1;
                DRDck = Bi*Dcveck;
                DRTDck = Dcveck'*Bi';
                Rk = Ri(ind_1, indBH(k));
                hterm2 = (-(DRDck*DRTDcj)-(DRDcj*DRTDck));
                vhterm2 = wi*vech(hterm2);
                hterm3 = ((-(Rk*DRTDck) -(DRDck*Rk'))*(-(Rk*DRTDcj) -(DRDcj*Rk')));
                vhterm3 = wi*vech(hterm3);
                hi(j, k) = 2.* sum((Resi.*vhterm2) + vhterm3);
                hi(k, j) = h(j, k);
            end
        end
        Isig2 = sig2*eye(mi);
        Isig22 = Isig2*Isig2;
        vIsig22 = wi*vech(tril(Isig22));
        hi(p*r+1, p*r+1) = 2.*(sum(Resi.*vIsig2-vIsig22));
        h = h+hi/mi;
    end


end

fullR = mat2cell(repmat(P, r,1),repelem(p,r), p);
fullR = blkdiag(fullR{:});
f = f + lambda.*cvec(1:end-1)'*fullR*cvec(1:end-1);
if nargout > 1
    g(1:end-1)  = g(1:end-1) + lambda.*(2.*fullR)*cvec(1:end-1);
end
if nargout > 2
    h(1:end-1, 1:end-1) = h(1:end-1, 1:end-1) +lambda.*(2.*fullR);
end





end

function R = get_R(cvec, basis, M, r,p)
R = zeros(M, r);
B = eval_basis(linspace(0,1,M), basis);
indB = cumsum([1, repelem(p,r)]);
for i = 1:r
    C = cvec(indB(i):i*p);
    R(:,i) = B*C;
end
end

%Aux functions
function vech = vech(mat)
    n = size(mat, 2);
    vech = mat(tril(ones(n),0)==1);
end




function [trH, trH2, DtrH] = traceH(data, basis, R, l, r, W)
BB = [];
n = length(unique(data(:,1)));
m2 = 0;
for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    wi = W{i};
    
    BB = [BB;Bi];
    m2 = m2 + mi;
end
H = BB*((BB'*BB + l.*R)\(BB'));
trH = trace(H)*r;
trH2 = trace(H^2)*r;
DtrH = trH-trH2;
end


function W = get_weight(Shat, S, SiID)
n = length(unique(SiID));
W = cell(n,1);
WiMax = zeros(n, 1);
for i = 1:n
    ind = SiID == i;
    CiHat = Shat(ind);
    Ci = S(ind);
    ri = CiHat - Ci;
    cov_ri = ri*ri'+diag(repelem(0.0001, length(ri)));
    W{i} = inv(cov_ri);
    WiMax(i) = max(max(W{i}));
end
W = cellfun(@(x) x/mean(WiMax), W, 'UniformOutput', false);
W = cellfun(@(x) x/max(max(x)), W, 'UniformOutput', false);
end

function [Shat] = get_Shat(cvec, data,basis, r, p)
Shat = [];
indB = cumsum([1, repelem(p,r)]);
n = length(unique(data(:,1)));
for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    Ri = zeros(mi, r);
    for j = 1:r
        C = cvec(indB(j):j*p);
        Ri(:,j) = Bi*C;
    end
%     sig2 = exp(cvec(end));
    Sihat = Ri*Ri';% + sig2*eye(mi);
    Shat = [Shat;vech(Sihat)];
end
end

