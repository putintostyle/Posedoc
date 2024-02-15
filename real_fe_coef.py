import numpy as np
import matplotlib.pyplot as plt
## Given ODE
## dx/dt = -11 x(t)
T = 1 # terminate time
mu = -10
dt = 1e-2
exactTime = np.linspace(0, T, int(T/dt)+1) # time step = 0, dt, 2dt...
def dynamic(input, m): 
    #  input: time
    #  m: mu 
    #  return : e^{\mu t} 
    return np.exp(m*input)
exactSol = dynamic(exactTime, mu) # exact solution x(t) = e^{mu t}

## Approx ODE using only data
###############################
## x_{n+1}-x_{n} = dt_0 * A x_{n}, given x_{n}, find A
## and |1+dt_0*mu|>1
###############################
dt_0 = dt*10
print(abs(1+mu*dt_0)) # our of stablility region
def findCoef(input, timestep):
    forwDiff = (input[1:-1]-input[0:-2])/timestep
    return np.inner(forwDiff, input[1:-1])/np.sum(input[0:-2]**2)
aprxmu = findCoef(exactSol, dt_0)
print(aprxmu)
# plot exact
Tmesh = np.linspace(0, T, int(T/dt)+1)
plt.plot(Tmesh, dynamic(Tmesh, mu), 'r', label = 'exact')
# plot approx
plt.scatter(Tmesh, dynamic(Tmesh, aprxmu), marker='x', label = 'explored')
plt.legend()
plt.show()
