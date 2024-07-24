function [q,bp,vp,def,totaltime] = solver_mexGpu(b,y,pdfy,ua)

coder.gpu.kernelfun;

theta = 0.0385; 
betta = 0.90;
sigg = 2;
rstar = 0.01; 

ny = size(y,1);
nb = size(b,1);
[~,nb0] = min(abs(b));

%Initialize the Value functions
vp = coder.nullcopy(zeros(ny,nb));  %continue repaying
vp1 = coder.nullcopy(zeros(ny,nb));  
evp = coder.nullcopy(zeros(ny,nb));

vd = coder.nullcopy(zeros(ny,1)); 
vo = coder.nullcopy(zeros(ny,1));

def = coder.nullcopy(false(ny,nb));
bp = coder.nullcopy(zeros(ny,nb)); %debt policy function (expressed in indices)  
q = ones(ny,nb)/(1+rstar); %q is price of debt; it is a function of  (y_t, d_{t+1}) 

diff = 1;
tol = 1e-7;
its = 1;

timer = tic;                   % <----- Start the timer

while diff > tol && its < 1000

  evp = betta*pdfy*vp;

  coder.gpu.kernel()
  for iy = 1:ny

      coder.gpu.constantMemory(y) 

    for ib = 1:nb

        tmpmax = - Inf ;
        bib = b(ib);

        for i = 1:nb

            coder.gpu.constantMemory(q)
            coder.gpu.constantMemory(evp)

            c1 =  b(i)*q(iy,i) - bib + y(iy);
            if c1 <= 0; c1 = - Inf ; 
                else; c1 = (c1^(1-sigg)-1)/(1-sigg) + evp(iy,i);
            end

            if tmpmax < c1; tmpmax = c1 ; end
        end
        vp1(iy,ib) = tmpmax;
    end
  end

vd1 = ua + betta*pdfy*(theta*vo + (1-theta)*vd);

def = vp1 < repmat(vd1,1,nb); 

qnew = (1- pdfy*def)/(1+rstar);

vp_1 = max(repmat(vd1,1,nb),vp1);

diff = max(max(abs(qnew-q))) + max(max(abs(vp_1-vp))) + max(abs(vd1-vd));

vo = vp_1(:,nb0);
vp = vp_1;
vd = vd1;
q = qnew;

if mod(its, 60) == 0 
  fprintf('%5.0f ~ %8.10f \n', its, diff);
end

its = its + 1;

end

totaltime = toc(timer);
avgtime   = totaltime/(its-1);

fprintf('# its%4.0f ~Time %8.8fs ~Avgtime %8.8fs \n', its-1, totaltime, avgtime);


coder.gpu.kernel()
for iy = 1:ny
    coder.gpu.constantMemory(y) 
  for ib = 1:nb
      tmpmax = - Inf ;
      maxidx = nb0;
      for i = 1:nb
          coder.gpu.constantMemory(q)
          coder.gpu.constantMemory(evp)
          c1 =  b(i)*q(iy,i) - b(ib) + y(iy);
          if c1 <= 0
              c1 = - Inf ; 
          else
              c1 = (c1^(1-sigg)-1)/(1-sigg) + evp(iy,i);
          end
          if tmpmax < c1; tmpmax = c1; maxidx = i ; end
      end
      bp(iy,ib) = maxidx;
  end
end

end

