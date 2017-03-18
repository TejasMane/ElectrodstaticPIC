# Setting the variables in the maxwell distribution
m = 1
K = 1
T = 1
e =-1

# k for the mode in fourier space
k = 2*np.pi
amp = 0.05


# The maxwell Boltzman function
def f_0(v):
    return np.sqrt(m/(2*np.pi*K*T))*np.exp(-m*v**2/(2*K*T))

# This the function which returns the derivative of the maxwell boltzmann equation
def diff_f_0_v(v):
    return np.sqrt(m/(2*np.pi*K*T))*np.exp(-m*v**2/(2*K*T)) * ( -m * v / (K * T))


# Assign the maxim and minimum velocity for the velocity grid
velocity_max =  +10
velocity_min =  -10

# Set the divisions for the velocity grid
number_of_velocities_points = 501
velocity_x = np.linspace(velocity_min, velocity_max, number_of_velocities_points)
dv = velocity_x[1] - velocity_x[0]

# Function that returns df_i/dt and df_r/dt used for odeint function
# See the latex document for more details on the differential equations
# This has been done to split the imaginary and real part of the ODE
def diff_delta_f(Y,t):
    f_r = Y[0:len(velocity_x)]  # Initial conditions for odeint
    f_i = Y[len(velocity_x): 2 * len(velocity_x)]

    int_Df_i = np.sum(f_i) * (velocity_x[1]-velocity_x[0])
    int_Df_r = np.sum(f_r) * (velocity_x[1]-velocity_x[0])

    # This the derivate for f_r and f_i given in the latex document
    dYdt =np.concatenate([(k * velocity_x * f_i) - e*e*(int_Df_i * diff_f_0_v(velocity_x)/k ), \
                           -(k * velocity_x * f_r) + e*e*(int_Df_r * diff_f_0_v(velocity_x)/k )\
                         ], axis = 0)
    # This returns the derivative for the coupled set of ODE

    return dYdt

def diff_delta_f_Ex(Y,t):

    f_r = Y[0:len(velocity_x)]  # Initial conditions for odeint
    f_i = Y[len(velocity_x): 2 * len(velocity_x)]
    E_x_r = Y[2 * len(velocity_x)]
    E_x_i = Y[2 * len(velocity_x) + 1]

    int_v_delta_f_dv_i = e * np.sum(f_i * velocity_x) * (dv)
    int_v_delta_f_dv_r = e * np.sum(f_r * velocity_x) * (dv)
    int_v_delta_f_dv = np.array([int_v_delta_f_dv_r, int_v_delta_f_dv_i ] )

    # This the derivate for f_r and f_i given in the latex document
    dYdt =np.concatenate([(    k * velocity_x * f_i) - e*(E_x_r * diff_f_0_v(velocity_x) ), \
                            - (k * velocity_x * f_r) - e*(E_x_i * diff_f_0_v(velocity_x) ), \
                                -1 * int_v_delta_f_dv\
                         ], axis = 0\
                        )
    # This returns the derivative for the coupled set of ODE

    return dYdt

# Set the initial conditions for delta f(v,t) here
delta_f_initial = np.zeros((2 * len(velocity_x)), dtype = np.float)
delta_f_initial[0: len(velocity_x)] = amp * f_0(velocity_x)

delta_f_Ex_initial = np.zeros((2 * len(velocity_x)+2), dtype = np.float)
delta_f_Ex_initial[0 : len(velocity_x)] = amp * f_0(velocity_x)
delta_f_Ex_initial[2 * len(velocity_x) + 1] = -1 * e * (1/k) * np.sum(delta_f_Ex_initial[0: len(velocity_x)] ) * dv

# Setting the parameters for time here
final_time = 3
dt = 0.001
time_ana = np.arange(0, final_time, dt)


# Variable for temperorily storing the real and imaginary parts of delta f used for odeint
initial_conditions_delta_f = np.zeros((2 * len(velocity_x)), dtype = np.float)
old_delta_f = np.zeros((2 * len(velocity_x)), dtype = np.float)


initial_conditions_delta_f_Ex = np.zeros((2 * len(velocity_x) + 2), dtype = np.float)
old_delta_f_Ex = np.zeros((2 * len(velocity_x) + 2 ), dtype = np.float)
# Variable for storing delta rho

delta_rho1 = np.zeros(len(time_ana), dtype = np.float)
delta_rho2 = np.zeros(len(time_ana), dtype = np.float)
Ex_amp  = np.zeros(len(time_ana), dtype = np.float)
Ex_amp2 = np.zeros(len(time_ana), dtype = np.float)
Ex_amp_real  = np.zeros(len(time_ana), dtype = np.float)
# Ex_amp3 = np.zeros(len(time_ana), dtype = np.float)
delta_f_temp = np.zeros(2 * len(velocity_x), dtype=np.float)
temperory_delta_f_Ex = np.zeros(2 * len(velocity_x) + 2, dtype=np.float)


for time_index, t0 in enumerate(time_ana):
    if(time_index%1000==0):
        print("Computing for TimeIndex = ", time_index)
    t0 = time_ana[time_index]
    if (time_index == time_ana.size - 1):
        break
    t1 = time_ana[time_index + 1]
    t = [t0, t1]

    # delta f is defined on the velocity grid


    # Initial conditions for the odeint
    if(time_index == 0):
        # Initial conditions for the odeint for the 2 ODE's respectively for the first time step
        # First column for storing the real values of delta f and 2nd column for the imaginary values
        initial_conditions_delta_f                 = delta_f_initial.copy()
        initial_conditions_delta_f_Ex                 = delta_f_Ex_initial.copy()
        # Storing the integral sum of delta f dv used in odeint

    else:
        # Initial conditions for the odeint for the 2 ODE's respectively for all other time steps
        # First column for storing the real values of delta f and 2nd column for the imaginary values
        initial_conditions_delta_f= old_delta_f.copy()
        initial_conditions_delta_f_Ex= old_delta_f_Ex.copy()
        # Storing the integral sum of delta f dv used in odeint

    # Integrating delta f

    temperory_delta_f = odeint(diff_delta_f, initial_conditions_delta_f, t)[1]
    temperory_delta_f_Ex = odeint(diff_delta_f_Ex, initial_conditions_delta_f_Ex, t)[1]

    # Saving delta rho for current time_index
    delta_rho1[time_index] = ((sum(dv * temperory_delta_f[ 0: len(velocity_x)])))
    delta_rho2[time_index] = ((sum(dv * temperory_delta_f_Ex[ 0: len(velocity_x)])))
    Ex_amp[time_index] = (e/k)*sum(  dv * temperory_delta_f[ 0: len(velocity_x)]  )
    Ex_amp2[time_index] = (e/k)*(sum  (  dv * temperory_delta_f_Ex[ 1 * len(velocity_x) : 2 * len(velocity_x)]  ))
    Ex_amp_real[time_index] = np.sqrt( Ex_amp[time_index]**2 + Ex_amp2[time_index]**2  )
    # Saving the solution for to use it for the next time step
    old_delta_f = temperory_delta_f.copy()
    old_delta_f_Ex = temperory_delta_f_Ex.copy()




# print('ExAmp is ', Ex_amp)
# h5f = h5py.File('data_files/LT.h5', 'w')
# h5f.create_dataset('delta_rho1',   data = delta_rho1)
# h5f.create_dataset('delta_rho2',   data = delta_rho2)
# h5f.create_dataset('Ex_amp',   data = Ex_amp)
# h5f.close()
#
#
# h5f           = h5py.File('data_files/LT.h5', 'r')
# delta_rho1     = h5f['delta_rho1'][:]
# delta_rho2     = h5f['delta_rho2'][:]
# Ex_amp     = h5f['Ex_amp'][:]
# h5f.close()

# print('data is ', data)

# Plotting the required quantities here
# pl.plot(time_ana, abs((Ex_amp)),label = '$LT1$')
# pl.plot(time_ana, abs((Ex_amp2)),label = '$LT2$')

# pl.plot(time_ana, (abs(delta_rho1)),label = '$\mathrm{Linear\;Theory\;fields}$')
# pl.plot(time_ana, (abs(delta_rho2)),label = '$\mathrm{Linear\;Theory\;No\;fields}$')
pl.plot(time_ana, (abs(Ex_amp_real)),label = '$\mathrm{Linear\;Theory\;fields}$')
# pl.plot(time_mill,data,label = '$\mathrm{Numerical\;PIC}$')

# pl.plot(time_mill, dataEx, label = '$\mathrm{Numerical\;PIC}$')
# pl.plot(time_mill, data_energy, label = '$\mathrm{Numerical\;PIC\;Energy}$')
pl.xlabel('$\mathrm{time}$')
# pl.ylabel(r'$\delta \hat{\rho}\left(t\right)$')
pl.ylabel('$\delta \hat{Ex}(t)$')

pl.title('$\mathrm{Linear\;Landau\;damping}$')
pl.legend()
# pl.ylim(0, 0.01)
# pl.xlim(0,2)
pl.show()
pl.clf()

# pl.plot(time_ana, np.log(abs(Ex_amp)),label = '$LT1$')
# pl.plot(time_ana, np.log(abs(Ex_amp2)),label = '$LT2$')

pl.semilogy(time_ana, (abs(Ex_amp_real)),label = '$LT1$')

# pl.plot(time_ana, np.log(abs(delta_rho1)),label = '$\mathrm{Linear\;Theory\;with\;fields}$')
# pl.plot(time_ana, np.log(abs(delta_rho2)),label = '$\mathrm{Linear\;Theory\;No\;fields}$')
pl.xlabel('$\mathrm{time}$')
# pl.xlim(0,2)
# pl.ylabel(r'$log(\delta \hat{\rho}\left(t\right))$')
pl.ylabel(r'$\log(\delta \hat{E_{x}}\left(t\right))$')
pl.title('$\mathrm{Linear\;Landau\;damping}$')
pl.legend()
pl.show()

