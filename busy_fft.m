% Keep matlab busy, that's it.
function busy_fft(n, m, k_max)
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
  m = 100;
end
gflo = 5*n*log2(n) *m;
a = randn(n,m) + 1i * randn(n,m);
t00 = tic;
for k = 1 : k_max
  tic;
  a = fft(a);
  t = toc;
  s = datestr(now);
  fprintf('%s, t=%.3f, #%d, GFLOPS=%.1f.\n', s, t, k, gflo/t/1e9);
  flushstdout();
end
t11 = toc(t00);
fprintf('%s, t=%.3f / %d, ave GFLOPS=%.1f.\n', s, t, k, k_max*gflo/t11/1e9);