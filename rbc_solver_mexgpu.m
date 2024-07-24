function [v,totaltime]  = rbc_solver_mexgpu(c0,kpgrid,pdfz)

coder.gpu.kernelfun;

tol = 1e-8;                       % Tolerance for V
diff = 1;
its = 1; 

beta = 0.95;      % Discount rate
eta = 2;

[nz,nk] = size(c0);

v = coder.nullcopy(zeros(nz,nk));     % Value Function
v0 = coder.nullcopy(zeros(nz,nk));    % v at the previous iteration

timer = tic;                   % <----- Start the timer

while diff > tol && its < 1000

ev = beta*pdfz*v;

  coder.gpu.kernel()
  for iz = 1:nz  
    for ik = 1:nk
        
        tmpmax = - Inf ;
        coder.gpu.constantMemory(c0) 

        for i = 1:nk
            
            coder.gpu.constantMemory(kpgrid);
            c1 = c0(iz,ik) - kpgrid(i);

            if c1 < 0; break; end
            coder.gpu.constantMemory(ev);
            c1 = c1^(1-eta)/(1-eta) + ev(iz,i);
            if tmpmax < c1; tmpmax = c1; end
            
        end

        v0(iz,ik) = tmpmax; 
    end
  end

diff = max(max(abs(v-v0)));   % Check convergence:
v = v0;                       % Save the value function

if mod(its, 60) == 0 
  fprintf('%5.0f ~ %8.10f \n', its, diff);
end

its = its + 1;

end

totaltime = toc(timer);
avgtime   = totaltime/(its-1);

fprintf('# its%4.0f ~Time %8.8fs ~Avgtime %8.8fs \n', its-1, totaltime, avgtime);

end



