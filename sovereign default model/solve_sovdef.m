clear

format compact

ny = 35;   % number of transitory shock grids
nb = 1000;   % # of grid points for debt 

%%
phi0 = -0.35;
phi1 = 0.4403;
rstar = 0.01;  %quarterly risk-free interest rate (Chatterjee and Eyigungor, AER 2012). 
theta = 0.0385; %probability of reentyy (USG  and Chatterjee and Eyigungor, AER 2012). 
sigg = 2; %intertemporal elasticity of consumption substitution 
betta = 0.90;  %discount factor, from Na, Schmitt-Grohe, Uribe, and Yue (2014)

rhoy = 0.931654648380119; %mean reversion of log prod  %% 0.86 from data
sdy = 0.036960096814455;  %stdev of log prod shock 
width = 4.2; 
muy = 0; %long run mean of log prod
[ygrid, pdfy] = tauchen(ny,muy,rhoy,sdy,width);

y = exp(ygrid);

ya =  y - max(0,phi0*y + phi1*y.^2); %output in autarky
ua = (ya.^(1-sigg)-1) / (1-sigg);

%debt grid
bupper = 2;    blower = 0;
b = blower:(bupper-blower)/(nb-1):bupper;
b = b(:);

%% run solver Matlab native

[q,bp,vp,def,time] = solver(b,y,pdfy,ua) ;

%% run solver Matlab mex

codegen -report solver -args {b,y,pdfy,ua} -o solver_mex1000

[q,bp,vp,def,time] = solver_mex1000(b,y,pdfy,ua) ;

%% run solver Matlab mex parfor

codegen -report solver_mexParfor -args {b,y,pdfy,ua} -o solver_mexParfor1000

[q,bp,vp,def,time] = solver_mexParfor1000(b,y,pdfy,ua);

%% run solver Mex CUDA

reset(gpuDevice)

cfg = coder.gpuConfig('mex');
cfg.GenerateReport = true;

b = gpuArray(b);
y = gpuArray(y);
pdfy = gpuArray(pdfy);
ua = gpuArray(ua);

codegen -config cfg solver_mexGpu -args {b,y,pdfy,ua} -o solver_mexGpu1000
[q,bp,vp,def,time]  = solver_mexGpu1000(b,y,pdfy,ua);

