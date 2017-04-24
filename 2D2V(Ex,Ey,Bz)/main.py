import numpy as np
import pylab as pl
import arrayfire as af
from initialization_functions import set_up_cosine_perturbation
from particle_pusher import Boris
from fdtd import fdtd, fft_poisson, periodic_field
from charge_deposition import cloud_charge_deposition, norm_background_ions
from current_deposition import Umeda_2003
import params

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'


# Weighting factor w_p = number of electrons composing the macro particle
# Doesn't affect the the physics of the system (Coarse graining doesn't affect vlasov maxwell system)
# It should be kept such that the number density in the PIC code corresponds
# to the densities found in real plasmas
w_p             = params.w_p

# Macro Particle parameters
k_boltzmann     = params.k_boltzmann
mass_electron   = params.mass_electron
tempertature    = params.tempertature
charge_electron = params.charge_electron
charge_ion      = params.charge_ion

# Setting the length of the domain
length_domain_x = params.length_domain_x
length_domain_y = params.length_domain_y

# Setting the number of ghost cells
ghost_cells  = params.ghost_cells

# Setting number of particle in the domain
number_of_electrons = params.number_of_electrons

# Initializing the positions and velocities of the particles
positions_x = length_domain_x * np.random.rand(number_of_electrons)
positions_y = length_domain_y * np.random.rand(number_of_electrons)

# setting the mean and standard deviation of the maxwell distribution
# Thermal/mean velocity of macro particles should correspond to
# that of individual electrons in the plasma
mu_x, sigma_x = 0, (k_boltzmann * tempertature / (mass_electron / w_p))
mu_y, sigma_y = 0, (k_boltzmann * tempertature / (mass_electron / w_p))

# Initializing the velocitites according to the maxwell distribution
velocity_x = np.random.normal(mu_x, sigma_x, number_of_electrons)
velocity_y = np.random.normal(mu_y, sigma_y, number_of_electrons)

# Divisions in x grid
divisions_domain_x = params.divisions_domain_x
divisions_domain_y = params.divisions_domain_y

dx = length_domain_x / divisions_domain_x
dy = length_domain_y / divisions_domain_y


# initializing the positions grid

x_grid = np.linspace(    0 - ghost_cells * dx,\
                            length_domain_x + ghost_cells * dx, \
                            divisions_domain_x + 1 + 2 * ghost_cells,\
                            endpoint=True,\
                            dtype = np.double\
                    )


x_right = x_grid + dx/2

# initializing the x grid
y_grid = np.linspace(    0 - ghost_cells * dy,\
                            length_domain_y + ghost_cells * dy, \
                            divisions_domain_y + 1 + 2 * ghost_cells,\
                            endpoint=True,\
                            dtype = np.double\
                    )

# dx, dy is the distance between consecutive grid nodes along x and y



y_top  = y_grid + dy/2



# Setting the amplitude for perturbation
N_divisions_x       = params.divisions_domain_x
N_divisions_y       = params.divisions_domain_y
amplitude_perturbed = params.amplitude_perturbed
k_x                 = params.k_x
k_y                 = params.k_y

# Initializing the perturbation
positions_x,\
positions_y = set_up_cosine_perturbation( positions_x, positions_y,\
                                                                  number_of_electrons,\
                                                                  N_divisions_x,\
                                                                  N_divisions_y,\
                                                                  amplitude_perturbed,\
                                                                  k_x, k_y, length_domain_x,\
                                                                  length_domain_y,\
                                                                  dx, dy\
                                                                )

# For 1D simulation:
positions_x = length_domain_x * np.random.rand(number_of_electrons)
velocity_x  = np.zeros(number_of_electrons)


# Converting to arrayfire arrays
positions_x  = af.to_array(positions_x)
positions_y  = af.to_array(positions_y)
velocity_x   = af.to_array(velocity_x)
velocity_y   = af.to_array(velocity_y)
x_grid       = af.to_array(x_grid)
y_grid       = af.to_array(y_grid)
x_right      = af.to_array(x_right)
y_top        = af.to_array(y_top)


# Time parameters
start_time = params.start_time

end_time   = params.end_time

dt         = params.dt

time       = np.arange(    start_time,\
                            end_time + dt,\
                            dt,\
                            dtype = np.double\
                        )

# Some variables for storing data
Ex_max       = np.zeros(len(time), dtype = np.double)
Ey_max       = np.zeros(len(time), dtype = np.double)


# Charge deposition using linear weighting scheme

rho_electrons  = cloud_charge_deposition( charge_electron,\
                                                            number_of_electrons,\
                                                            positions_x,\
                                                            positions_y,\
                                                            x_grid,\
                                                            y_grid,\
                                                            ghost_cells,\
                                                            length_domain_x,\
                                                            length_domain_y,\
                                                            dx,\
                                                            dy,\
                                                            w_p
                                                           )
rho_initial    = norm_background_ions(rho_electrons, number_of_electrons, w_p)

# # plotting intial rho in the system considering background ions
pl.plot(np.array(rho_initial)[:, 1])
pl.show()
pl.clf()

# Computing the initial electric fields Ex and Ey
Ex_initial_centered = af.data.constant(0, y_grid.elements(), x_grid.elements(), dtype = af.Dtype.f64)
Ey_initial_centered = af.data.constant(0, y_grid.elements(), x_grid.elements(), dtype = af.Dtype.f64)

rho_physical = rho_initial[ghost_cells:-ghost_cells, ghost_cells:-ghost_cells].copy()

Ex_temp, Ey_temp = fft_poisson(rho_physical, dx, dy)

Ex_initial_centered[ghost_cells:-ghost_cells\
                    ,ghost_cells:-ghost_cells\
                    ]                           = Ex_temp.copy()

Ey_initial_centered[ghost_cells:-ghost_cells\
                    ,ghost_cells:-ghost_cells\
                    ]                           = Ey_temp.copy()

Ex_initial_centered = periodic_field(Ex_initial_centered, ghost_cells)
Ey_initial_centered = periodic_field(Ey_initial_centered, ghost_cells)

# Bringing Ex_initial_centered, Ey_initial_centered to staggered grid
Ex_initial_staggered = 0.5 * (Ex_initial_centered + af.shift(Ex_initial_centered, 0, -1))
Ex_initial_staggered = periodic_field(Ex_initial_staggered, ghost_cells)

Ey_initial_staggered = 0.5 * (Ey_initial_centered + af.shift(Ey_initial_centered, -1, 0))
Ey_initial_staggered = periodic_field(Ey_initial_staggered, ghost_cells)


# The following cell block determines $v(\frac{\Delta t}{2})$:
# \begin{align}
# v(\frac{\Delta t}{2}) = v(t = 0) + E_{x}\left(x(\frac{\Delta t}{2})\right)\frac{\Delta t}{2}
# \end{align}

# This cell block is to obtain v at (t = 0.5dt) to implement the verlet algorithm.

positions_x_half = positions_x + velocity_x * dt/2
positions_y_half = positions_y + velocity_y * dt/2

# Periodic Boundary conditions for particles

positions_x_half, positions_y_half = periodic_particles(positions_x_half,\
                                                        positions_y_half,\
                                                        length_domain_x,\
                                                        length_domain_y\
                                                        )

# Finding interpolant fractions for the positions

fracs_Ex_x, fracs_Ex_y = fraction_finder(positions_x_half,\
                                            positions_y_half, \
                                            x_right, y_grid,\
                                            dx, dy\
                                        )

fracs_Ey_x, fracs_Ey_y = fraction_finder(positions_x_half,\
                                            positions_y_half, \
                                            x_grid, y_top,\
                                            dx, dy\
                                        )
# Interpolating the fields at each particle

Ex_particle = af.signal.approx2(Ex_initial_staggered, fracs_Ex_y, fracs_Ex_x)
Ey_particle = af.signal.approx2(Ey_initial_staggered, fracs_Ey_y, fracs_Ey_x)

# Updating the velocity using the interpolated Electric fields to find v at (t = 0.5dt)

velocity_x = velocity_x  + (Ex_particle * charge_electron / mass_electron ) * dt/2
velocity_y = velocity_y  + (Ey_particle * charge_electron / mass_electron ) * dt/2

Ex = Ex_initial_staggered.copy()
Ey = Ey_initial_staggered.copy()
Bz = 0 * Ex_initial_staggered.copy()
pl.plot(np.array(Ey_initial_staggered)[:, 1])
pl.show()
pl.clf()

for time_index in range(len(time)):
    if(time_index%25 ==0):
        print('Computing for time = ', time_index * dt)
        
        
    # Updating the positions of particle using the velocites (Verlet algorithm)
    # velocity at t = (n + 1/2) dt, positions_x at t = (n)dt and positions_x_new
    # at t = (n+1)dt
    positions_x_new = positions_x + velocity_x * dt
    positions_y_new = positions_y + velocity_y * dt

    # Periodic Boundary conditions for particles
    positions_x_new, positions_y_new = periodic_particles(positions_x_new, positions_y_new,\
                                                          length_domain_x, length_domain_y\
                                                         )

    # Computing the current densities on the staggered grid provided by Umeda's scheme

    Jx_staggered, Jy_staggered = Umeda_2003( w_p, charge_electron, number_of_electrons, positions_x ,positions_y, velocity_x,\
                    velocity_y, x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y,\
                    dx, dy, dt\
                  )

    # Evolving electric fields using currents

    Ex_updated, Ey_updated, Bz_updated = fdtd(Bz, Ex, Ey, length_domain_x, length_domain_y, ghost_cells, Jx_staggered, Jy_staggered, dt)


    # calculating the interpolation fraction needed for arrayfire's 1D interpolation

    fracs_Ex_x, fracs_Ex_y = fraction_finder(positions_x_new, positions_y_new, x_right, y_grid, dx, dy)

    fracs_Ey_x, fracs_Ey_y = fraction_finder(positions_x_new, positions_y_new, x_grid, y_top, dx, dy)

    fracs_Bz_x, fracs_Bz_y = fraction_finder(positions_x_new, positions_y_new, x_right, y_top, dx, dy)


    # Interpolating the fields at particle locations

    Ex_particle = af.signal.approx2(Ex_updated, fracs_Ex_y, fracs_Ex_x)

    Ey_particle = af.signal.approx2(Ey_updated, fracs_Ey_y, fracs_Ey_x)

    Bz_particle = af.signal.approx2(Bz_updated, fracs_Bz_y, fracs_Bz_x)


    # Updating the velocity using the interpolated Electric fields

    velocity_x_new, velocity_y_new = Boris( charge_electron, mass_electron,\
                                            velocity_x, velocity_y, dt,\
                                            Ex_particle, Ey_particle, Bz_particle\
                                          )

    # Saving the Electric fields for plotting

    Ex_max[time_index]       = (af.max(af.abs(Ex[ghost_cells:-ghost_cells,ghost_cells:-ghost_cells])))
    Ey_max[time_index]       = (af.max(af.abs(Ey[ghost_cells:-ghost_cells,ghost_cells:-ghost_cells])))

    pl.plot(np.array(Ey)[ghost_cells:-ghost_cells, 2])
    pl.ylim(-1, 1)
    pl.savefig('images/' + str(time_index) + '.png')
    pl.clf()


    # Saving the updated velocites for the next timestep
    positions_x = positions_x_new.copy()
    positions_y = positions_y_new.copy()
    velocity_x  = velocity_x_new.copy()
    velocity_y  = velocity_y_new.copy()
    Ex          = Ex_updated.copy()
    Ey          = Ey_updated.copy()
    Bz          = Bz_updated.copy()



# Reading data generated by the Cheng Knorr code
h5f = h5py.File('CK_256.h5', 'r')
Ex_max_CK = h5f['max_E'][:]
h5f.close()
time_CK = np.linspace(0,time[-1], len(Ex_max_CK))

time_grid = np.linspace(0, time[-1], len(Ex_max))
pl.plot(time_grid, Ey_max , label = r'$\mathrm{PIC}$')
# pl.plot(time_LT, Ex_max_LT,'--',lw = 3,label = '$\mathrm{LT}$')
pl.plot(time_CK, Ex_max_CK, label = '$\mathrm{Cheng\;Knorr}$')
# print('(abs(Ex_amplitude[0])) is ',(abs(Ex_amplitude[0])))
# print('(abs(Ex_max[0])) is ',(abs(Ex_max[0])))
pl.xlabel('$t$')
pl.ylabel('$\mathrm{MAX}(|E_{x}|)$')
pl.legend()
pl.show()
# pl.savefig('MaxE.png')
pl.clf()

pl.semilogy(time_grid, Ey_max ,label = r'$\mathrm{PIC}$')
# pl.semilogy(time_LT, Ex_max_LT,'--',lw = 3,label = '$\mathrm{LT}$')
pl.semilogy(time_CK,Ex_max_CK, label = '$\mathrm{Cheng\;Knorr}$')
pl.legend()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{MAX}(|E_{x}|)$')
pl.show()
# pl.savefig('MaxE_semilogy.png')
pl.clf()

pl.loglog(time_grid, Ey_max ,label = r'$\mathrm{PIC}$')
# pl.loglog(time_LT, Ex_max_LT,'--',lw = 3,label = '$\mathrm{LT}$')
pl.semilogy(time_CK,Ex_max_CK, label = '$\mathrm{Cheng\;Knorr}$')
pl.legend()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{MAX}(|E_{x}|)$')
pl.show()
# pl.savefig('MaxE_loglog.png')
pl.clf()
