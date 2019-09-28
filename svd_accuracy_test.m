% Test accuracy of SVD under extreme scale
% Run
% $ /share/apps/matlab/2019a/bin/matlab -nojvm
% Or
% $ LD_PRELOAD=$PWD/lapack_intercept_sdd2svd.so /share/apps/matlab/2019a/bin/matlab -nojvm

n=2501;
nmag = -16;           % negative magnitude to test
randn('state', 234);
[u0, ~] = qr(randn(n));
[v0, ~] = qr(eye(n) + 10^nmag * randn(n));
v0 = v0 .* diag(v0);
s0 = logspace(0, nmag, n);
a = u0 * diag(s0) * v0';
s = svd(a);
s(end)

% sdd: 2500, -20
% 3.0904e-22

% sdd: 2501, -20
% 1.4688e-22

% svd: 2500, -20
% 3.0904e-22

% svd: 2501, -20
% 6.0459e-22


