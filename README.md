# MatlabMex-Multithreading
Simple macroeconomics models solved on CPU and GPU-based multithreading, using Matlab Mex Code generation.
The model discussed are a classic stochastic growth model and a sovereign default model (Arellano, 2008).

I use Matlab language to generate C code that runs on multiple cores of my CPU (Mex parfor), and CUDA code on GPU (Mex CUDA). 
In addition to Matlab Parallel Computation Toolbox, several drivers and compilers are necessary to enable these methods. 
This multitreading method combines the ease of use of Matlab language and IDE with the efficiency of compiled language, i.e. C and CUDA. 
Maltab Coder takes care of the C/CUDA code generation, knowledge of C/CUDA is not necessary. 

On my Laptop (Intel 11800H CPU and NVIDIA RTX 3060 GPU), compared with the speed of serial Matlab method on CPU, Mex parfor brings 20-fold acceleration, while Mex CUDA brings 50-fold acceleration with double-precision (FP64), and 600-fold acceleration with single-precision (FP32). 

Further details can be found in the attached working paper in file `Parallel_Compiled_July2024.pdf'
