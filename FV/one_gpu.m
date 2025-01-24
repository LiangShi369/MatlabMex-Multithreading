
clear

reset(gpuDevice);

phi0 = -0.35;
%phi1 = 0.24558;
rstar = 0.01;  %quarterly risk-free interest rate (Chatterjee and Eyigungor, AER 2012). 
theta = 0.0385; %probability of reentyy (USG  and Chatterjee and Eyigungor, AER 2012). 
sigg = 2; %intertemporal elasticity of consumption substitution 
betta = 0.90;%discount factor, from Na, Schmitt-Grohe, Uribe, and Yue (2014)
ny = 35; % number of transitory shock grids
nd = 400; % # of grid points for debt 

rhoy = 0.931654648380119; %mean reversion of log prod  %% 0.86 from data
sdy = 0.036960096814455;  %stdev of log prod shock 
width = 4.2; 
muy = 0; %long run mean of log prod
[ygrid,pdfy] = tauchen(ny,muy,rhoy,sdy,width);

y = exp(ygrid);
ny = numel(ygrid); %number of grid points for log of ouput
%ya = (1-phi)*y;

phi1 = (1-phi0)/2/max(y);
ya =  y - max(0,phi0*y + phi1*y.^2); %output in autarky
%ua = (ya.^(1-sigg)-1) / (1-sigg);

%debt grid
dupper = 2;    dlower = 0;
d = dlower:(dupper-dlower)/(nd-1):dupper;
d = d(:);
[~,nd0] = min(abs(d));  d(nd0) = 0; %force the element closest to zero to be exactly zero

n = ny*nd; %total number of states 

%matrix for  indices of output as a function of the current state (size ny-by-nd)
yix = repmat((1:ny)',1,nd);

%matrix for  indices of debt as a function of the current state (size ny-by-nd)
dix = repmat(1:nd,ny,1);

%Consider a generic n-by-nd matrix Xtry. Each row of Xtry indicates one of the n possible states in the current period  and columns correspond to all (nd) possible values for  debt assumed in the current peirod and due in the next period under continuation. The variable X could be d_t, y_t, d_{t+1}, etc. 
dtry = repmat(d',[ny 1  nd]);
dtry = reshape(dtry,n,nd);

dptryix = repmat(1:nd,n,1);
dptry = d(dptryix);

ytryix = repmat(yix,nd,1);
ytry = y(ytryix);

%AUTARKY
%Consumption under autarky
%ca = repmat(ya, [1 nd]); %consumption of under bad standing
ua = ( ya.^(1-sigg)-1)  / (1-sigg);  %period utility under autarky

%Initialize the Value functions
vgood = gpuArray.zeros(ny,nd);  %continue repaying
vbad = gpuArray.zeros(ny,1); 

vbadgood = vbad;
vgood1 = vgood;  
vbad1 = vbad;

dp = gpuArray.zeros(ny,nd); %debt policy function (expressed in indices)  
dp1 = dp;
q = gpuArray.ones(ny,nd)/(1+rstar); %q is price of debt; it is a function of  (y_t, d_{t+1}) 
ua = gpuArray(ua);
dtry = gpuArray(dtry);
ytry = gpuArray(ytry);
pdfy = gpuArray(pdfy);

diff = 1;
tol = 1e-7;
its = 1;
maxits = 2000;

smctime   = tic;
totaltime = 0;

while diff > tol  &&  its < maxits

qtry = repmat(q,[nd 1]);

ctry =  dptry.*qtry - dtry + ytry;

utry = (ctry.^(1-sigg) -1)  / (1-sigg);

utry(ctry<=0) = -inf;

Evgood = pdfy * vgood;
Evgood = repmat(Evgood,nd,1);

[vgood1(:), dp1(:)] = max(utry + betta*Evgood, [], 2);

vbad1 = ua + betta*pdfy*(theta*vbadgood + (1-theta)*vbad);
%vbad1m = repmat(vbad1,1,nd);

def = vgood1 < repmat(vbad1,1,nd); 

qnew = (1- pdfy*def)/(1+rstar);

vgood_1 = max(repmat(vbad1,1,nd),vgood1);

diff = max(abs(qnew(:) - q(:))) + max(abs(vgood_1(:) - vgood(:)))...
    + max(abs(vbad1(:) - vbad(:)));

vbadgood = vgood_1(:,nd0);
vgood = vgood_1;
vbad = vbad1;
q = qnew;
dp = dp1;

totaltime = totaltime + toc(smctime);
  avgtime   = totaltime/its;

  if mod(its, 40) == 0 || diff<=tol
      fprintf('%d ~%8.8f ~%8.5fs ~%8.5fs \n', its, diff, totaltime, avgtime);
  end
    
  its = its+1;
  smctime = tic; % re-start clock

end %while dist>...


%debt choice under continuation
dp = gather(dp);
dpolicy = d(dp);
q = gather(q);

%[X, Y] = meshgrid(y,d);
%Z = griddata(d,y,q,Y,X,'linear');

%figure
%ss = surf(X,Y,Z,'FaceAlpha',0.5);
%ss.EdgeColor = 'interp';
%axis tight
%xlabel('y states','interpreter','latex')
%ylabel('d policy','interpreter','latex')
%zlabel('Debt price')


fname = sprintf('one_gpu_%dy_%dd.mat', ny,nd);
save(fname, 'betta', 'd','dp', 'def','dlower','dupper','nd','ny','pdfy','q',...
    'rhoy','rstar','sdy','sigg','theta','width','y','ya','its',...
    'totaltime','avgtime')

