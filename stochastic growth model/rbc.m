clear

format compact

nz = 35;            % Grid size 
nk = 1000;  % grid points + 1

%% Initialize the parameters
beta = 0.95;      % Discount rate
eta = 2;           % Risk aversion parameter
alpha = 0.33;      % Technology parameter
delta = 0.1;      % Depreciation rate
rho = 0.90;        % Tech. shock persistence
sigma = 0.01;     % Tech. shock st. dev.

%% Grid for productivity z
muz = 0;
width = 4.2; 

[z, pdfz] = tauchen(nz,muz,rho,sigma,width);
z = exp(z);

%% Grid for capital k
kstar = (alpha/(1/beta - (1-delta)))^(1/(1-alpha)); % steady state k
cstar = kstar^(alpha) - delta*kstar;
istar = delta*kstar;
ystar = kstar^(alpha);

kmin = 0.25*kstar;
kmax = 4*kstar;
grid = (kmax-kmin)/(nk-1);

k = kmin:grid:kmax;
k = k';
kpgrid = k';

c0 = zeros(nz,nk);     % total wealth
for iz=1:nz
  c0(iz,:) = z(iz)*k.^alpha + (1-delta)*k;
end

%% run solvers Matlab native

[v, time]  = rbc_solver(c0,kpgrid,pdfz);


%% run solver Mex

codegen -report rbc_solver -args {c0,kpgrid,pdfz} -o rbc_solver1000

[v, time]  = rbc_solver1000(c0,kpgrid,pdfz);


%% run solver Matlab mex parfor

codegen -report rbc_solver_mexParfor -args {c0,kpgrid,pdfz} -o rbc_solver_mexParfor1000

[v, time]  = rbc_solver_mexParfor1000(c0,kpgrid,pdfz);


%% run solver Mex CUDA

reset(gpuDevice)

cfg = coder.gpuConfig('mex');
cfg.GenerateReport = true;

c0 = gpuArray(c0);
kpgrid = gpuArray(kpgrid);
pdfz = gpuArray(pdfz);

codegen -config cfg rbc_solver_mexgpu -args {c0,kpgrid,pdfz} -o rbc_solver_mexgpu1000

[v, time]  = rbc_solver_mexgpu1000(c0,kpgrid,pdfz);




