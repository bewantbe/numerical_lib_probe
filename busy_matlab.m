% Keep matlab busy, that's it.
function busy_matlab(n, k_max)
if exist('OCTAVE_VERSION', 'builtin')
  flushstdout = @() fflush(stdout);
else
  flushstdout = @() 0;
end
if ~exist('k_max', 'var')
  k_max = 2^48-1;   % upper bound of loop variable in matlab
  k_max = 2^31-3;   % max for Octave 3.8.2
end
if ~exist('n', 'var')
  n = 4000;
end
gflo = n^3*2;
v = rand(n,1);  v = v/norm(v);
u = rand(n,1);  u = u/norm(u);
% a (not very) random orthogonal matrix
a = eye(n) - 2*u*u' - 2*v*v' + 4*u*(u'*v)*v';
c = a;
t00 = tic;
for k = 1 : k_max
  tic;
  c = c*a;
  t = toc;
  s = datestr(now);
  fprintf('%s, t=%.3f, #%d, GFLOPS=%.1f.\n', s, t, k, gflo/t/1e9);
  flushstdout();
end
t11 = toc(t00);
fprintf('%s, t=%.3f / %d, ave GFLOPS=%.1f.\n', s, t, k, k_max*gflo/t11/1e9);