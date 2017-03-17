# %%
import numpy as np
import pylab as pl
import scipy.fftpack as ff
import h5py
# %%
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

# $$

# Functions defined here

# FFT solver :

def fft_poisson(rho,dx):

    k = ff.fftfreq(len(rho), d = dx)

    # print('k is ', k)

    rho_fft = (ff.fft(rho))

    rho_fft[1:] = -1 * (1/(k[1:])**2) * rho_fft[1:]

    rho_fft[0] = 0

    # print('rho_fft is ', rho_fft)

    V = np.array(ff.ifft(rho_fft), dtype=np.float)

    return V

def compute_electric_field(V,dx):

    E = -(np.roll(V, -1) - np.roll(V, 1))/dx

    return E



# b1 charge depositor
def cloud_charge_deposition(charge, zone_x, frac_x, x_grid):

    left_corner_charge = (1 - frac_x) * charge
    right_corner_charge = (frac_x) * charge

    left_corners  = zone_x.copy()
    right_corners = left_corners + 1

    corners = np.concatenate([left_corners, right_corners], axis=0)
    charges = np.concatenate([left_corner_charge, right_corner_charge], axis=0)

    rho, temp = np.histogram(corners, bins=len(x_grid), range=(0, len(x_grid)), weights=charges)

    return rho


# %%

k_boltzmann     = 1
mass_electron   = 1
tempertature    = 1
charge_electron = -1
charge_ion      = +1

# %%

length_domain_x = 1

# %%

number_of_electrons = 3

positions_x = length_domain_x * np.random.rand(number_of_electrons)

mu, sigma = 0, (k_boltzmann * tempertature / mass_electron)

velocity_x = np.random.normal(mu, sigma, number_of_electrons)

# %%

divisions_domain_x = 100

x_grid = np.linspace(0, length_domain_x, divisions_domain_x + 1, endpoint=True)

dx = x_grid[1] - x_grid[0]
# %%
start_time = 0

end_time   = 3

dt  = 0.01

time = np.arange(start_time, end_time + dt, dt)

# %%

rho_ions = (charge_ion * number_of_electrons) / (length_domain_x)

# %%

for i in range(len(time)):

    # Updating the positions

    positions_x += velocity_x * dt

    input('check')

    # Boundary conditions
    outside_domain = np.where([positions_x < 0])[0]

    input('check')

    positions_x[outside_domain] += length_domain_x

    input('check')

    outside_domain = np.where([positions_x > length_domain_x])[0]
    positions_x[outside_domain] -= length_domain_x

    input('check')

    # Finding interpolant fractions for the positions

    zone_x = np.floor(((positions_x - x_grid[0]) / dx))
    zone_x = np.array([zone_x], dtype=np.int)
    print('zone_x is ',zone_x)
    frac_x = (positions_x - x_grid[zone_x]) / (dx)

    input('check')

    # Charge deposition using linear weighting scheme

    rho = cloud_charge_deposition(charge_electron, zone_x, frac_x, x_grid)
    rho+= rho_ions

    input('check')

    # Calculating the potential from the charge deposition.

    V = fft_poisson(rho,dx)

    input('check')

    # Computing E from the potential

    Ex = np.array(compute_electric_field(V,dx), dtype=None)

    input('check')

    # Interpolating the fields at each particle
    Ex_particle = Ex[zone_x] + frac_x * Ex[zone_x + 1]

    input('check')

    # Updating the particles using the interpolated field values.
    print('Ex_particle.dtype is ',Ex_particle.dtype)
    print('velocity_x.dtype is ',velocity_x.dtype)

    velocity_x += (Ex_particle * charge_electron / mass_electron ) * dt

    input('check')

    h5f = h5py.File('data/timestepped_data/solution_'+str(time_index)+'.h5', 'w')
    h5f.create_dataset('positions_x',   data = positions_x)
    h5f.create_dataset('velocity_x',   data = velocity_x)
    h5f.create_dataset('Ex',   data = (Ex))
    h5f.close()




# # cloud_charge_deposition test
# charge = 1
# x_grid = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# x = np.array([0.9])
# zone_x = np.array([4])
# frac_x = np.array([0.5])
# print(cloud_charge_deposition(charge, zone_x, frac_x, x_grid))



# # FFT test

# rho_size = 100
# x = np.linspace(0, 1, rho_size )
#
# A = 0.5
# rho =  A * np.sin(2 * np.pi * x)
# dx = x[1] - x[0]
#
# V = fft_poisson(rho, dx)
# E = compute_electric_field(V,dx)
#
# pl.plot(x, V,label = '$V$')
# pl.plot(x, rho, label = r'$\rho$')
# pl.plot(x, E, label = '$E_{x}$')
# pl.legend()
# pl.show()
# pl.clf()
