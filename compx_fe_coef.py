import numpy as np
import matplotlib.pyplot as plt
## Given ODE
## dx/dt = -11 x(t)
J = np.copy(1j)
T = 10 # terminate time
mu = -1+2*J
dt = 1e-3
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
    input_re = np.real(input)[0:-2]
    input_im = np.imag(input)[0:-2]

    forwDiff = (input[1:-1]-input[0:-2])/timestep

    forwDiff_re = np.real(forwDiff)
    forwDiff_im = np.imag(forwDiff)

    lamb_re = (np.inner(forwDiff_re, input_re)+ np.inner(forwDiff_im, input_im))/np.linalg.norm(input[1:-1])**2
    lamb_im = -(np.inner(forwDiff_re, input_im)- np.inner(forwDiff_im, input_re))/np.linalg.norm(input[1:-1])**2
    
    return lamb_re+J*lamb_im
aprxmu = []

stab_reg = -2*np.real(mu)/np.linalg.norm(mu)**2

trailTime = np.linspace(dt/2, 10*stab_reg, 100001)
print(stab_reg)
for ts in trailTime:
    ap_mu = findCoef(exactSol, ts)
    aprxmu.append(ap_mu)

aprxmu = np.array(aprxmu)


##### Plot
## Figure 1
## plot the distribution of the explored dynamic 
## identify the given time step is inside |1+\mu \dt|<1 or not 
idx = np.array(np.abs(1+mu*trailTime))<1

plt.figure(1)
plt.scatter(np.real(mu), np.imag(mu), marker = 'x', label = 'exact', c = 'green') ## exact dynamic
print(np.real(aprxmu[idx]))
plt.scatter(np.real(aprxmu[idx]), np.imag(aprxmu[idx]), color = 'red', marker = '^', label = 'inside stability region', facecolors='none')
plt.scatter(np.real(aprxmu[~idx]), np.imag(aprxmu[~idx]), color = 'navy',marker = 'o', label = 'outside stability region')
plt.suptitle('exact $\mu$ ='+str(mu)+'\n Estimated dynamic $\hat{\mu}(\Delta t)$')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.legend()
plt.grid(which = 'major',ls= 'dashed')
plt.savefig('distr_of_estimate.png', dpi = 300)

## Figure 2
## plot the differnece of |\hat{mu}(\dt_0)-\mu|
plt.figure(2)
plt.loglog(trailTime[idx], np.abs(mu-np.array(aprxmu)[idx]), color = 'red', label = 'inside stability region')
plt.scatter(trailTime[idx], np.abs(mu-np.array(aprxmu)[idx]), color = 'red', marker = 'x')
plt.loglog(trailTime[~idx], np.abs(mu-np.array(aprxmu)[~idx]), color = 'navy', label = 'outside stability region')
plt.scatter(trailTime[~idx], np.abs(mu-np.array(aprxmu)[~idx]), color = 'navy', marker = 'x')
plt.suptitle('exact $\mu$ ='+str(mu))
plt.xlabel('$\Delta t$')
plt.ylabel('$|\hat{\mu}(\Delta t)-\mu|$')
plt.legend()
plt.grid(which = 'major',ls= 'dashed')
plt.savefig('diff_of_estimate.png', dpi = 300)
plt.show()