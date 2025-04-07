function [RMSE] = RMSE4EIG(A,B)
if isrow(A)
    A = A';
end
if isrow(B)
    B = B';
end

D    = A-B;
SQE  = D.^2;
MSE  = mean(SQE(:));
RMSEpos = sqrt(MSE);

D    = A+B;
SQE  = D.^2;
MSE  = mean(SQE(:));
RMSEneg = sqrt(MSE);


RMSE = min(RMSEpos, RMSEneg);
end