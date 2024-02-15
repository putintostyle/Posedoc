import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
## Given ODE
## dx/dt = -10 x(t)
T = 1 # terminate time
mu = -5
# stab_reg = -2/(mu) # given stability region 
#-------------------------------------------#
## find underlying $\hat{\mu}$ by using least square fitting
aprxmu = np.array([])
ratio = np.array([])
## given different time steps
timeTrail = np.linspace(0,1, 1001)
timeTrail = timeTrail[1:-1]
for tr in timeTrail:
    dt = tr
    exactTime = np.linspace(0, T, int(T/dt)+1) # time step = 0, dt, 2dt...
    
    def dynamic(input, m): 
        #  input: time
        #  m: mu 
        #  return : e^{\mu t} 
        return np.exp(m*input)
    exactSol = dynamic(exactTime, mu) # exact solution x(t) = e^{mu t}
    

    ## Approx ODE using only initial data and time step
    ###############################
    ## z_{n+1} = (1+\hat{\mu} \dt) z_0, given z_0 = x(0)
    ###############################
    ## Define cost function
    ## \sum |z_{n+1}-x(t_n)|^2
    # print(1/(int(T/dt)+1)*np.sum((np.array([exactSol[0]*(1+mu*dt)**n for n in range(int(T/dt)+1)])-exactSol))**2)
    
    def costFun(mu0):
        predict = np.array([])
        for n in range(int(T/dt)+1):
            predict = np.append(predict, (1+mu0*dt)**n)
        
        return (predict-exactSol)
    ## least square fitting
    mu_hat = least_squares(costFun, 10)
    # print('estimate\n')
    # print(np.exp(mu_hat.cost))
    
    aprxmu = np.append(aprxmu, mu_hat.x)
    ratio = np.append(ratio, abs(1+dt*mu_hat.x))
### Plot prop1 in stab region
## determine whether given step size is in the stability region
idx = np.ndarray.flatten(ratio)>1

# plot the estimated \hat{\mu}
plt.plot(timeTrail[idx], aprxmu[idx], color = 'r', label = 'outside stability region')
plt.plot(timeTrail[~idx], aprxmu[~idx], 'navy', label = 'inside stability region')
plt.plot(timeTrail, np.abs(aprxmu-mu), label = '$|mu-\hat{\mu}|$', ls = 'dashdot') # exact
plt.xlabel('$\Delta t$ ')
plt.ylabel('$\hat{\mu}(\Delta t)$')
# plt.savefig('rfe_estimated_T1.png')
plt.show()    
### Plot prop2 |1+\mu^*\Delta t|<e^{\mu_0\Delta t}
plt.plot(timeTrail, ratio, label = '$|1+\mu^*\Delta t|$', ls = '-')
plt.plot(timeTrail, np.exp(mu*timeTrail), label = '$e^{\mu_0\Delta t}$', ls = 'dotted')
plt.xlabel('$\Delta t$ ')
plt.suptitle('profiles of discovery dynamic')
plt.legend()
plt.grid(ls = 'dashed')
# plt.savefig('rfe_profile_T1.png')
plt.show()
