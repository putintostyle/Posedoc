import numpy as np
import matplotlib.pyplot as plt
## Given ODE
## dx/dt = -11 x(t)
T = 10 # terminate time
lamb = -10
exactTime = np.linspace(0, T, 101) # time step = 0.01, ...
exactSol = np.exp(lamb*exactTime) # exact solution

dt = .1
x0 = 1
numerSol = [x0]
numerTime = np.linspace(0, T, int(T/dt)+1)

for i in range(len(numerTime)-1):
    x0 = (1+(lamb)*dt)*x0
    numerSol.append(x0)
print(abs(1+dt*lamb))
plt.plot(exactTime, exactSol,  c = 'r', label = 'exact')
plt.scatter(numerTime, numerSol, marker = 'o', c = 'b', label = 'app')

plt.legend()
plt.show()
