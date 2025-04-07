function [R, cvec, r, GCV, lambda] = fitR(S, data, M, r, p, basis, lambda, penmat, options, omega)
B = eval_basis(linspace(0, 1, M), basis);
n = length(unique(data(:,1)));
%Parameter selection
%rank selection
if isempty(r) || length(r) > 1
    if n/M > 1 || any(any(isnan(iDTY2mat(data))))
    % if any(any(isnan(S))) || n/M > 1 || any(any(isnan(iDTY2mat(data))))
        disp('rank selection');
        if isempty(r)
            r = 1:round(0.2*M);
        end
         mrsse = zeros(length(r),1);
         prev_mrsse = 0;
        for i = 1:length(r)
            cvec0 = abs(rand(p*r(i)+1,1));
            cvec = optimC(cvec0, S, r(i), p, B, penmat, 0, options);
            R = get_R(cvec, B, M, r(i), p);
            mrsse(i) = rmse(R*R',S, 'all');
            curr_mrsse = mrsse(i);
            if abs(prev_mrsse - curr_mrsse) <= omega || i == length(r)
                r = r(i);
                break
            else
                prev_mrsse = mrsse(i);
            end 
        end
    else
        r = getNPC_DonohoGavish(iDTY2mat(data));
    end
disp(['r = ',num2str(r)]);
end

gcvs = [];
cvec = [];
%lambda selection
if isempty(lambda) || length(lambda) > 1

    if isempty(gcp('nocreate'))
        parpool;
    end
    disp('lambda selection ...');
    if isempty(lambda)
        lambda = 10.^(-7:1);
    end
    gcvs = zeros(length(lambda),1);
    cvecCell = cell(length(lambda), 1);

    if ~isempty(gcp('nocreate'))
        parfor i = 1:length(lambda)
            cvec0 = abs(rand(p*r+1,1));       
            cvec = optimC(cvec0, S, r, p, B, penmat, lambda(i), options);
            R = get_R(cvec, B, M, r, p);
            RRT = R*R';
            SSE = sum(sum((S-RRT).^2));
            H = Hmat(B, penmat, lambda(i), r);
            gcvs(i) = (SSE*M.^2) / (M.^2 - trace(H))^2;
            cvecCell{i} = cvec;
        end
    else
        for i = 1:length(lambda)
            disp(['lambda ',num2str(i),'/',num2str(length(lambda))]);
            cvec0 = abs(rand(p*r+1,1));
            cvec = optimC(cvec0, S, r, p, B, penmat, lambda(i), options);
            R = get_R(cvec, B, M, r, p);
            RRT = R*R';
            SSE = sum(sum((S-RRT).^2));
            H = Hmat(B, penmat, lambda(i), r);
            gcvs(i) = (SSE*M.^2) / (M.^2 - trace(H))^2;
            cvecCell{i} = cvec;
        end
    end
    gcvs = round(gcvs,4); %for cases with equal gcv upto 4 dp
    minGCV = find(gcvs == min(gcvs), 1,'first');
    lam = lambda(minGCV);
    if lam == lambda(end)
        warning('Last value in lambda grid selected')
    end
    lambda = lam;
    cvec = cvecCell{minGCV};
end

if isempty(cvec)
    cvec0 = abs(rand(p*r+1,1));
    cvec = optimC(cvec0, S, r, p, B, penmat, lambda, options);
end
cvec = optimC(cvec, S, r, p, B, penmat, lambda, options);
R = get_R(cvec, B, M, r, p);

if isempty(gcvs)
    RRT = R*R';
    SSE = sum(sum((S-RRT).^2));
    H = Hmat(B, penmat, lambda, r);
    GCV = (SSE*M.^2) / (M.^2 - trace(H))^2;
else
    GCV = gcvs;
end
end


function [f, g, h]= loss(cvec, S, r, p, B, P, lambda)
indB = cumsum([1, repelem(p,r)]);
M = size(S,2);
R = zeros(M, r);
Pen = 0;
for i = 1:r
    C = cvec(indB(i):i*p); 
    R(:,i) = B*C;
    Pen = Pen+lambda.*(C'*P*C);
end
sig2 = exp(cvec(end));
Shat = R*R'+ sig2*eye(M);
Res = S-Shat;
f = sum(sum((Res).^2))+Pen;
% f = f/M^2;

if nargout > 1 && M > 2500
    g = zeros(p*r,1);
    ind_c = cumsum([0, repelem(p,r)]);
    indB  = repelem(1:r, p);
    Dn = repelem(p,p*r);
    fullR = [];
      for j = 1:p*r
        Dcvec = zeros(Dn(j),1);
        Dcvec(j-(ind_c(indB(j)))) = 1;
        ind_1      = 1:M; 
        g(j) = 2.*sum(sum( Res.*( -R(ind_1,indB(j))*(Dcvec')*(B') -...
                B*(Dcvec)*(R(ind_1,indB(j))') )));

        if j <= r
            fullR = blkdiag(fullR, P);
        end
      end
      
      g  = g+lambda.*(2.*fullR)*cvec(1:end-1);
      g = [g;sum(sum(2.*Res.*(-sig2*eye(M))))];
%       g = g./M^2;
end

if nargout > 1 && M <= 2500
    g = zeros(p*r,1);
    ind_c = cumsum([0, repelem(p,r)]);
    fullR = [];
    vRes = Res(:);
    for j = 1:r
        RTDRDc = kron(B', R(:,j)');
        RDRTDc = kron(R(:,j)', B');
        g((ind_c(j)+1):ind_c(j+1)) = 2.*((-RDRTDc - RTDRDc)*vRes);
        fullR = blkdiag(fullR, P);
    end
      g  = g+lambda.*(2.*fullR)*cvec(1:end-1);
      g = [g;sum(sum(2.*Res.*(-sig2*eye(M))))];
%       g = g./M^2;
end

if nargout > 2
    h = zeros(p*r, p*r);
    Dn = repelem(p,p*r);
    indB  = repelem(1:r, p);
    ind_1 = 1:M; 
    for j = 1:p*r
        Dcvecj = zeros(Dn(j),1);
        Dcvecj(j-(ind_c(indB(j)))) = 1;
        DRDcj = B*Dcvecj;
        DRTDcj = Dcvecj'*B';
        for k = 1:j
            if indB(j) ~= indB(k)
                continue
            end
            Dcveck = zeros(Dn(k),1);
            Dcveck(k-(ind_c(indB(k)))) = 1;
            DRDck = B*Dcveck;
            DRTDck = Dcveck'*B';
            Rk = R(ind_1, indB(k));
            h(j, k) = 2.* sum(sum( (Res.*(-(DRDck*DRTDcj)-(DRDcj*DRTDck))) + ...
                ((-(Rk*DRTDck) -(DRDck*Rk'))*(-(Rk*DRTDcj) -(DRDcj*Rk'))) ));
            h(k, j) = h(j, k);
        end
    end
    h = h+lambda.*(2.*fullR);
%     h = h./M^2;
end


end

function R = get_R(cvec, B, M, r,p)
R = zeros(M, r);
indB = cumsum([1, repelem(p,r)]);
for i = 1:r
    C = cvec(indB(i):i*p);
    R(:,i) = B*C;
end
end


function trH = Hmat(B, R, l,r)
trH = 0;
for i = 1:r
    Hi = B*((B'*B + l.*R)\B');
    trH = trH + trace(Hi);
end
end

function [GCV, RGCV, R1GCV] = RoGCV(B, P, r, lambda, S, RRT, M)
trH = 0; trH2 = 0; 

for i = 1:r
    Hi = B*((B'*B + lambda.*P)\B');
    trH = trH + trace(Hi);
    trH2 = trH + trace(Hi^2);
end
npts = r*M;
if npts < 100
    gamma = .2;
else
    gamma = .3;
end
GCV = (sum(sum((S-RRT).^2)))*M^2 / (2.*(npts - trH))^2;
RGCV = (gamma+(1-gamma)*trH2)*GCV;
trH12 = (trH - trH2)/lambda;
R1GCV = (gamma+(1-gamma)*trH12)*GCV;
end


function npc = getNPC_DonohoGavish(X)
[n, m] = size(X);
beta = n/m;

if beta > 1 || beta < 1e-3
     warning('Approximation for \\beta(\\omega) may be invalid.')
end
omega_beta = .56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43;

y = svd(X);
rankY = min( find(cumsum(y(y>0))/sum(y(y>0)) > .995) );
y_med = median(y);
npc = min(max(1, sum(y > omega_beta * y_med)),  rankY);
end

function cvec = optimC(cvec0, S, r, p, B, P, lambda, options)
try
   cvec = fminunc(@loss, cvec0, options, S, r, p, B, P, lambda);
catch
    cvec0 = jitter(cvec0);
   cvec = fminunc(@loss, cvec0, options, S, r, p, B, P, lambda);
end
end
