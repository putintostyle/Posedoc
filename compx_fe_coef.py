import numpy as np
import matplotlib.pyplot as plt
## Given ODE
## dx/dt = -11 x(t)
J = np.copy(1j)
T = 1 # terminate time
mu = -1+5J
dt = 1e-6
exactTime = np.linspace(0, T, int(T/dt)+1) # time step = 0, dt, 2dt...
def dynamic(input, m): 
    #  input: time
    #  m: mu 
    #  return : e^{\mu t} 
    return np.exp(input*m)
exactSol = dynamic(exactTime, mu) # exact solution x(t) = e^{mu t}

## Approx ODE using only data
###############################
## x_{n+1}-x_{n} = dt_0 * A x_{n}, given x_{n}, find A
## and |1+dt_0*mu|>1
###############################

def findCoef(input, timestep):
    input_re = np.real(input)[1:-1]
    input_im = np.imag(input)[1:-1]

    forwDiff = (input[1:-1]-input[0:-2])/timestep

    forwDiff_re = np.real(forwDiff)
    forwDiff_im = np.imag(forwDiff)

    lamb_re = (np.inner(forwDiff_re, input_re)+ np.inner(forwDiff_im, input_im))/np.linalg.norm(input[1:-1])**2
    lamb_im = -(np.inner(forwDiff_re, input_im)- np.inner(forwDiff_im, input_re))/np.linalg.norm(input[1:-1])**2
    
    return lamb_re+J*lamb_im
aprxmu = []

stab_reg = -2*np.real(mu)/np.linalg.norm(mu)**2
# plt.figure()
for ts in [dt]:
    print(np.linalg.norm(1+mu*ts))
    ap_mu = findCoef(exactSol, ts)
    aprxmu.append(ap_mu)
print(aprxmu)   
# plt.scatter(np.real(aprxmu), np.imag(aprxmu))

# plt.loglog(np.linspace(stab_reg, 1e-3,10000), np.abs(np.array(aprxmu)-mu))
# plt.show()