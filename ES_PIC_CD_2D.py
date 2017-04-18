
# coding: utf-8

# In[1]:

import numpy as np
import h5py
import pylab as pl
import arrayfire as af
from scipy.fftpack import fftfreq
from numpy.fft import fft2, ifft2
# af.set_backend('cpu')


# In[2]:

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


# In[3]:

def fraction_finder(x, y, x_grid, y_grid, dx_frac_finder, dy_frac_finder):

    # print('x_grid[0] is ', x_grid[0])

    x_frac = (x - af.sum(x_grid[0])) / dx_frac_finder
    # print('y_grid[0] is ', y_grid[0])
    # print(' (y - (y_grid[0])) / dy_frac_finder is ', (y - (y_grid[0])) / dy_frac_finder)
    y_frac = (y - af.sum(y_grid[0])) / dy_frac_finder

    af.eval(x_frac, y_frac)

    return x_frac, y_frac


# In[4]:

def periodic_particles(positions_x, positions_y, length_domain_x, length_domain_y):
    '''
    function periodic_particles(positions_x, length_domain_x)
    -----------------------------------------------------------------------
    Input variables: positions_x and length_domain_x

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        length_domain_x: This is the length of the domain. The domain is assumed to be from x = 0
        to x = length_domain_x

    -----------------------------------------------------------------------
    returns: positions_x
        This function returns the modified positions_x such that particle previously gone outside the domain
        through the left boundary enter via the right boundary and vice versa.
        In other words, the function implements periodic boundary conditions for the particles.

    '''

    # Arrayfire implementation
    # Determine indices of particles which have gone outside the domain
    # through right boundary
    outside_domain_right_x       = af.algorithm.where(positions_x >= length_domain_x)
    outside_domain_top_y         = af.algorithm.where(positions_y >= length_domain_y)

    # Determine indices of particles which have gone outside the domain
    # through left boundary
    outside_domain_left_x        = af.algorithm.where(positions_x <  0  )
    outside_domain_bottom_y      = af.algorithm.where(positions_y <  0  )


    if outside_domain_right_x.elements() > 0:

        # Apply periodic boundary conditions

        positions_x[outside_domain_right_x] = positions_x[outside_domain_right_x] - length_domain_x

    if outside_domain_top_y.elements() > 0:

        # Apply periodic boundary conditions

        positions_y[outside_domain_top_y] = positions_x[outside_domain_top_y] - length_domain_y

    if outside_domain_left_x.elements() > 0:

        # Apply periodic boundary conditions

        positions_x[outside_domain_left_x]  = positions_x[outside_domain_left_x] + length_domain_x

    if outside_domain_bottom_y.elements() > 0:

        # Apply periodic boundary conditions

        positions_y[outside_domain_bottom_y]  = positions_x[outside_domain_bottom_y] + length_domain_y


    af.eval(positions_x, positions_y)

    return positions_x, positions_y


# ## cloud in cell charge deposition:
# \begin{align}
# \implies  \rho(\mathbf{x_{i}}, t) &= \frac{1 }{\Delta x}w_{p}q \sum_{p = 1}^{N} b_{1}\left(\frac{x_{p} - \mathbf{x_{i}}}{\Delta x}\right)
# \end{align}

# In[5]:

# b1 charge depositor
def charge_b1_depositor(charge, x, y, x_grid, y_grid, ghost_cells, Lx, Ly):
    # b1 charge depositor
    '''
    function cloud_charge_deposition(charge, zone_x, frac_x, x_grid, dx)
    -----------------------------------------------------------------------
    Input variables: charge, zone_x, frac_x, x_grid, dx

        charge: This is a scalar denoting the charge of the macro particle in the PIC code.

        zone_x: This is an array of size (number of electrons/macro particles) containing the indices
        of the left corners of the grid cells containing the respective particles

        frac_x: This is an array of size (number of electrons/macro particles) containing the fractional values of
        the positions of particles in their respective grid cells. This is used for linear interpolation

        x_grid: This is an array denoting the position grid chosen in the PIC simulation

        dx: This is the distance between any two consecutive grid nodes of the position grid.

    -----------------------------------------------------------------------
    returns: rho

        rho: This is an array containing the charges deposited at the density grid nodes.
    '''




    number_of_particles = x.elements()

    x_charge_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)
    y_charge_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)

    nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
    ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

    dx = Lx/nx
    dy = Ly/ny

    x_zone = (((af.abs(x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    y_zone = (((af.abs(y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

    x_zone_plus = x_zone + 1
    y_zone_plus = y_zone + 1

    dy_by_delta_y = (1/dy) * (y-y_grid[y_zone])
    dy_by_delta_y_complement = 1 - dy_by_delta_y

    dx_by_delta_x = (1/dx) * (x-x_grid[x_zone])
    dx_by_delta_x_complement = 1 - dx_by_delta_x

    weight_corner1 = dy_by_delta_y_complement * dx_by_delta_x_complement
    weight_corner2 = dy_by_delta_y * dx_by_delta_x_complement
    weight_corner3 = dy_by_delta_y * dx_by_delta_x
    weight_corner4 = dy_by_delta_y_complement * dx_by_delta_x

    charge_by_dxdy = ((charge/(dx*dy)))

    corner1_charge   = weight_corner1 * charge_by_dxdy
    corner2_charge   = weight_corner2 * charge_by_dxdy
    corner3_charge   = weight_corner3 * charge_by_dxdy
    corner4_charge   = weight_corner4 * charge_by_dxdy

    all_corners_weighted_charge = af.join(0,corner1_charge, corner2_charge, corner3_charge, corner4_charge)

    x_charge_zone[0 * number_of_particles : 1 * number_of_particles] = x_zone
    x_charge_zone[1 * number_of_particles : 2 * number_of_particles] = x_zone
    x_charge_zone[2 * number_of_particles : 3 * number_of_particles] = x_zone_plus
    x_charge_zone[3 * number_of_particles : 4 * number_of_particles] = x_zone_plus

    y_charge_zone[0 * number_of_particles : 1 * number_of_particles] = y_zone
    y_charge_zone[1 * number_of_particles : 2 * number_of_particles] = y_zone_plus
    y_charge_zone[2 * number_of_particles : 3 * number_of_particles] = y_zone_plus
    y_charge_zone[3 * number_of_particles : 4 * number_of_particles] = y_zone

    af.eval(x_charge_zone, y_charge_zone)
    af.eval(all_corners_weighted_charge)

    return x_charge_zone, y_charge_zone, all_corners_weighted_charge


# In[6]:

def cloud_charge_deposition(charge,                            no_of_particles,                            positions_x,                            positions_y,                            x_center_grid,                            y_center_grid,                            shape_function,                            ghost_cells,                            Lx,                            Ly,                            dx,                            dy                           ):

    elements = x_center_grid.elements()*y_center_grid.elements()

    rho_x_indices,     rho_y_indices,     rho_values_at_these_indices = shape_function(charge,positions_x, positions_y,                                                x_center_grid, y_center_grid,                                               ghost_cells, Lx, Ly                                              )

    input_indices = (rho_x_indices*(y_center_grid.elements())+ rho_y_indices)

    rho, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=rho_values_at_these_indices)
    rho = af.data.moddims(af.to_array(rho), y_center_grid.elements(), x_center_grid.elements())

    # Periodic BC's for charge deposition

    rho[0, :]  = rho[-1, :] + rho[0, :]
    rho[-1, :] = rho[0, :].copy()
    rho[:, 0]  = rho[:, -1] + rho[:, 0]
    rho[:, -1] = rho[:, 0].copy()

    af.eval(rho)

    return rho


# The charge density is computed using the normalization factor $A$ given by :
# \begin{align}
# f_{pic} &= f_{a}  \\
# \implies A &= \frac{\int \int f_{a}\;dv\;dx}{\int \int f_{pic}\;dv\;dx}  \\
# A &= \frac{\int \int f_{a}\;dv\;dx}{\int \int\sum_{p=1}^{N_{m}} w_{p}S(x, x_{p})S(v, v_{x,p})\;dv\;dx} \\
# A &= \frac{\int \int f_{a}\;dv\;dx}{N_{m} * w_{p}}
# \end{align}
# Background ion density is added to the computed charge density array:
# \begin{align}
# \int_{x_{i - \frac{1}{2}}}^{x_{i + \frac{1}{2}}} S(x, x_{p})dx &= b_{0+1}(\frac{\mathbf{x_{i}}- x_{p}}{\Delta x}) \\
# \implies  \rho(\mathbf{x_{i}}, t) &= \frac{1 }{\Delta x}Aw_{p}q \sum_{p = 1}^{N} b_{1}\left(\frac{x_{p} - \mathbf{x_{i}}}{\Delta x}\right) + \rho_{ions}
# \end{align}

# In[7]:

def norm_background_ions(rho_electrons, number_of_electrons, w_p):
    '''
    function norm_background_ions(rho_electrons, number_of_electrons)
    -----------------------------------------------------------------------
    Input variables: rho_electrons, number_of_electrons
        rho_electrons: This is an array containing the charges deposited at the density grid nodes.

        number_of_electrons: A scalar denoting the number of macro particles/electrons taken in the simulation
    -----------------------------------------------------------------------
    returns: rho_normalized
        This function returns a array denoting the normalized charge density throughout the domain containing
        the contribution due background ions

    '''
    A                        = 1/(number_of_electrons * w_p)
    rho_electrons_normalized = A*rho_electrons

    # Adding background ion density, and ensuring charge neutrality
    rho_ions_normalized      = - np.sum(rho_electrons_normalized) / (rho_electrons_normalized.elements())

    rho_normalized           = rho_electrons_normalized + rho_ions_normalized


    af.eval(rho_normalized)

    return rho_normalized




# \begin{align}
# \hat{V}(k) &= \int_{0}^{1} potential(x)e^{-2\pi\;i\;k\;x}dx \\ \\
# potential(x) &= \frac{1}{Npoints}\int_{0}^{1} \hat{potential}(k)e^{+2\pi\;i\;k\;x}dk \\ \\
# \hat{potential}(k) &= \frac{1}{4\pi^{2}\;k^2}\hat{\rho(k)} \\ \\
# \hat{E}(k) &= -i(2\pi\;k)\hat{potential}(k)
# \end{align}

# In[8]:

def fft_poisson(rho, dx, dy = None):
    """
    FFT solver which returns the value of electric field. This will only work
    when the system being solved for has periodic boundary conditions.

    Parameters:
    -----------
    rho : The 1D/2D density array obtained from calculate_density() is passed to this
          function.

    dx  : Step size in the x-grid

    dy  : Step size in the y-grid.Set to None by default to avoid conflict with the 1D case.

    Output:
    -------
    E_x, E_y : Depending on the dimensionality of the system considered, either both E_x, and
               E_y are returned or E_x is returned.
    """

    k_x = af.to_array(fftfreq(rho.shape[1], dx))
    k_x = af.Array.as_type(k_x, af.Dtype.c64)
    k_y = af.to_array(fftfreq(rho.shape[0], dy))
    k_x = af.tile(af.reorder(k_x), rho.shape[0], 1)
    k_y = af.tile(k_y, 1, rho.shape[1])
    k_y = af.Array.as_type(k_y, af.Dtype.c64)

    rho = np.array(rho)
    rho_hat       = fft2(rho)
    rho_hat = af.to_array(rho_hat)
    potential_hat = af.constant(0, rho.shape[0], rho.shape[1], dtype=af.Dtype.c64)

    potential_hat       = (1/(4 * np.pi**2 * (k_x * k_x + k_y * k_y))) * rho_hat

    potential_hat[0, 0] = 0

    potential_hat = np.array(potential_hat)

    E_x_hat = -1j * 2 * np.pi * np.array(k_x) * potential_hat
    E_y_hat = -1j * 2 * np.pi * np.array(k_y) * potential_hat

    E_x = (ifft2(E_x_hat)).real
    E_y = (ifft2(E_y_hat)).real

    E_x = af.to_array(E_x)
    E_y = af.to_array(E_y)

    af.eval(E_x, E_y)


    return(E_x, E_y)


# In[9]:

def set_up_perturbation(positions_x,                        positions_y,                        number_particles,                        N_divisions_x,                        N_divisions_y,                        amplitude,                        k_x,                        k_y,                        length_domain_x,                        length_domain_y,                        dx,                        dy
                       ):
    '''
    function set_up_perturbation(positions_x, number_particles, N_divisions,\
                                 amplitude , k, length_domain_x\
                                ):
    -----------------------------------------------------------------------
    Input variables: positions_x, number_particles, N_divisions, amplitude, k,length_domain_x

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        number_particles: The number of electrons /macro particles

        N_divisions: The number of divisions considered for placing the macro particles

        amplitude: This is the amplitude of the density perturbation

        k_x: The is the wave number of the cosine density pertubation

        length_domain_x: This is the length of the domain in x direction

    -----------------------------------------------------------------------
    returns: positions_x
        This function returns a array positions_x such that there is a cosine density perturbation
        of the given amplitude

    '''
    # There might be a few particles left out during execution of function and the statement
    # below randomizes those positions
    positions_x = length_domain_x * np.random.rand(number_particles)
    positions_y = length_domain_y * np.random.rand(number_particles)

    particles_till_x_i = 0

    # Looping over grid cells in the domain
    for j in range(N_divisions_y):
        for i in range(N_divisions_x):

            # Average number of particles present between two consecutive grid nodes
            average_particles_x_i_to_i_plus_one = (number_particles/\
                                                   ((length_domain_x * length_domain_y)/(dx * dy))\
                                                  )

            # Amplitude in the current grid cell used to compute number of particles in the
            # current grid cell
            temp_amplitude = amplitude * np.cos((k_x * (i + 0.5) * dx / length_domain_x) + \
                                                (k_y * (j + 0.5) * dy / length_domain_y))

            # Computing number of particles in the current grid cell
            number_particles_x_i_to_i_plus_one = int(average_particles_x_i_to_i_plus_one \
                                                     * (1 + temp_amplitude)\
                                                    )

            # Assigining these number of particles their respective positions in the current grid cell
            positions_x[particles_till_x_i\
                        :particles_till_x_i\
                        + number_particles_x_i_to_i_plus_one \
                       ] \
                                = i * dx \
                                  + dx * np.random.rand(number_particles_x_i_to_i_plus_one)

            positions_y[particles_till_x_i\
                        :particles_till_x_i\
                        + number_particles_x_i_to_i_plus_one \
                       ] \
                                = j * dy \
                                  + dy * np.random.rand(number_particles_x_i_to_i_plus_one)
            # keeping track of the number of particles that have been assigned positions
            particles_till_x_i += number_particles_x_i_to_i_plus_one

    return positions_x, positions_y


# In[10]:

# Umeda needs x(n), and v(n+0.5dt) for implementation
def Umeda_b1_deposition( charge_particle, positions_x, positions_y, velocity_x, velocity_y,\
                            x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt  ):

    '''
    A modified Umeda's scheme was implemented to handle a pure one dimensional case.

    function Umeda_b1_deposition( charge, x, velocity_x,\
                                  x_grid, ghost_cells, length_domain_x, dt\
                                )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x, dt

        charge: This is an array containing the charges deposited at the density grid nodes.

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        velocity_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the velocities of particles in x direction.

        x_grid: This is an array denoting the position grid chosen in the PIC simulation

        ghost_cells: This is the number of ghost cells used in the simulation domain.

        length_domain_x: This is the length of the domain in x direction

        dt: this is the dt/time step chosen in the simulation
    -----------------------------------------------------------------------
    returns: Jx_x_indices, Jx_values_at_these_indices

        Jx_x_indices: This returns the indices of the array where the respective currents stored in
        Jx_values_at_these_indices have to be deposited

        Jx_values_at_these_indices: This is an array containing the currents to be deposited.

    For further details on the scheme refer to Umeda's paper provided in the sagemath folder as the
    naming conventions used in the function use the papers naming convention.(x_1, x_2, x_r, F_x, )

    '''

    x_current_zone = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    y_current_zone = af.data.constant(0, positions_y.elements(), dtype=af.Dtype.u32)

    nx = (x_grid.elements() - 1 - 2 * ghost_cells )  # number of zones
    ny = (y_grid.elements() - 1 - 2 * ghost_cells )  # number of zones

    dx = length_domain_x/nx
    dy = length_domain_y/ny

    x_1 = (positions_x).as_type(af.Dtype.f64)
    x_2 = (positions_x + (velocity_x * dt)).as_type(af.Dtype.f64)

    y_1 = (positions_y).as_type(af.Dtype.f64)
    y_2 = (positions_y + (velocity_y * dt)).as_type(af.Dtype.f64)

    i_1 = af.arith.floor( ((af.abs( x_1 - af.sum(x_grid[0])))/dx) - ghost_cells)
    j_1 = af.arith.floor( ((af.abs( y_1 - af.sum(y_grid[0])))/dy) - ghost_cells)

    i_2 = af.arith.floor( ((af.abs( x_2 - af.sum(x_grid[0])))/dx) - ghost_cells)
    j_2 = af.arith.floor( ((af.abs( y_2 - af.sum(y_grid[0])))/dy) - ghost_cells)

    i_dx = dx * af.join(1, i_1, i_2)
    j_dy = dy * af.join(1, j_1, j_2)

    i_dx_x_avg = af.join(1, af.max(i_dx,1), ((x_1+x_2)/2))
    j_dy_y_avg = af.join(1, af.max(j_dy,1), ((y_1+y_2)/2))

    x_r_term_1 = dx + af.min(i_dx, 1)
    x_r_term_2 = af.max(i_dx_x_avg, 1)

    y_r_term_1 = dy + af.min(j_dy, 1)
    y_r_term_2 = af.max(j_dy_y_avg, 1)

    x_r_combined_term = af.join(1, x_r_term_1, x_r_term_2)
    y_r_combined_term = af.join(1, y_r_term_1, y_r_term_2)

    x_r = af.min(x_r_combined_term, 1)
    y_r = af.min(y_r_combined_term, 1)


    F_x_1 = charge_particle * (x_r - x_1)/dt
    F_x_2 = charge_particle * (x_2 - x_r)/dt

    F_y_1 = charge_particle * (y_r - y_1)/dt
    F_y_2 = charge_particle * (y_2 - y_r)/dt

    W_x_1 = (x_1 + x_r)/(2 * dx) - i_1
    W_x_2 = (x_2 + x_r)/(2 * dx) - i_2

    W_y_1 = (y_1 + y_r)/(2 * dy) - j_1
    W_y_2 = (y_2 + y_r)/(2 * dy) - j_2

    J_x_1_1 = (1/(dx * dy)) * (F_x_1 * (1 - W_y_1))
    J_x_1_2 = (1/(dx * dy)) * (F_x_1 * (W_y_1))

    J_x_2_1 = (1/(dx * dy)) * (F_x_2 * (1 - W_y_2))
    J_x_2_2 = (1/(dx * dy)) * (F_x_2 * (W_y_2))

    J_y_1_1 = (1/(dx * dy)) * (F_y_1 * (1 - W_x_1))
    J_y_1_2 = (1/(dx * dy)) * (F_y_1 * (W_x_1))

    J_y_2_1 = (1/(dx * dy)) * (F_y_2 * (1 - W_x_2))
    J_y_2_2 = (1/(dx * dy)) * (F_y_2 * (W_x_2))

    Jx_x_indices = af.join(0,\
                           i_1 + ghost_cells,\
                           i_1 + ghost_cells,\
                           i_2 + ghost_cells,\
                           i_2 + ghost_cells\
                          )

    Jx_y_indices = af.join(0,\
                           j_1 + ghost_cells,\
                           (j_1 + 1 + ghost_cells),\
                            j_2 + ghost_cells,\
                           (j_2 + 1 + ghost_cells)\
                          )

    Jx_values_at_these_indices = af.join(0,\
                                         J_x_1_1,\
                                         J_x_1_2,\
                                         J_x_2_1,\
                                         J_x_2_2\
                                        )

    Jy_x_indices = af.join(0,\
                           i_1 + ghost_cells,\
                           (i_1 + 1 + ghost_cells),\
                            i_2 + ghost_cells,\
                           (i_2 + 1 + ghost_cells)\
                          )

    Jy_y_indices = af.join(0,\
                           j_1 + ghost_cells,\
                           j_1 + ghost_cells,\
                           j_2 + ghost_cells,\
                           j_2 + ghost_cells\
                          )

    Jy_values_at_these_indices = af.join(0,\
                                         J_y_1_1,\
                                         J_y_1_2,\
                                         J_y_2_1,\
                                         J_y_2_2\
                                        )

    af.eval(Jx_x_indices, Jx_y_indices, Jy_x_indices, Jy_y_indices)

    af.eval(Jx_values_at_these_indices, Jy_values_at_these_indices)

    return Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
           Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices


# In[11]:

def Umeda_2003(charge, no_of_particles, positions_x ,positions_y, velocities_x, velocities_y, \
                x_center_grid, y_center_grid, ghost_cells, Lx, Ly, dx, dy, dt              \
                ):

    '''
    function Umeda_b1_deposition( charge, x, velocity_x,\
                                  x_grid, ghost_cells, length_domain_x, dt\
                                )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x, dt

        charge: This is an array containing the charges deposited at the density grid nodes.

        positions_x(t = n*dt): An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        velocity_x(t = (n+1/2)*dt): An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the velocities of particles in x direction.

        x_grid: This is an array denoting the position grid chosen in the PIC simulation.

        ghost_cells: This is the number of ghost cells used in the simulation domain..

        length_domain_x: This is the length of the domain in x direction.

        dt: this is the dt/time step chosen in the simulation.
    -----------------------------------------------------------------------
    returns: Jx_staggered

        Jx_staggered: This returns the array Jx on a staggered yee lattice.


    '''

    elements = x_center_grid.elements() * y_center_grid.elements()

    Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
    Jy_x_indices, Jy_y_indices,\
    Jy_values_at_these_indices = Umeda_b1_deposition( charge, positions_x, positions_y, velocities_x,\
                                                     velocities_y, x_center_grid, y_center_grid,\
                                                     ghost_cells, Lx, Ly, dt\
                                                   )

    input_indices = (Jx_x_indices*(y_center_grid.elements()) + Jx_y_indices)
    Jx, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jx_values_at_these_indices)
    Jx = af.data.moddims(af.to_array(Jx), y_center_grid.elements(), x_center_grid.elements())

    input_indices = (Jy_x_indices*(y_center_grid.elements()) + Jy_y_indices)
    Jy, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jy_values_at_these_indices)
    Jy = af.data.moddims(af.to_array(Jy), y_center_grid.elements(), x_center_grid.elements())

    af.eval(Jx, Jy)

    return Jx, Jy


# The current density is computed using the normalization factor $A$ given by :
# \begin{align}
# f_{pic} &= f_{a}  \\
# \implies A &= \frac{\int \int f_{a}\;dv\;dx}{\int \int f_{pic}\;dv\;dx}  \\
# A &= \frac{\int \int f_{a}\;dv\;dx}{\int \int\sum_{p=1}^{N_{m}} w_{p}S(x, x_{p})S(v, v_{x,p})\;dv\;dx} \\
# A &= \frac{\int \int f_{a}\;dv\;dx}{N_{m} * w_{p}}
# \end{align}
# Background ion density is added to the computed charge density array:
# \begin{align}
# \int_{x_{i - \frac{1}{2}}}^{x_{i + \frac{1}{2}}} S(x, x_{p})dx &= b_{0+1}(\frac{\mathbf{x_{i}}- x_{p}}{\Delta x}) \\
# \implies  J(\mathbf{x_{i}}, t) &= \frac{1}{\Delta x} Aw_{p}\;q \sum_{p=1}^{N_{m}} v_{x,p}\;b_{1}(\mathbf{x_{i}}, x_{p})
# \end{align}

# In[12]:

def current_norm_BC_Jx(Jx_staggered, number_of_electrons, w_p):

    '''
    function current_norm_BC(Jx_staggered, number_of_electrons, w_p)
    -----------------------------------------------------------------------
    Input variables: Jx_staggered, number_of_electrons, w_p

        Jx_staggered: This is an array containing the currents deposited on staggered lattice.

        number_of_electrons: Number of macroparticles taken in the domain.

        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------
    returns: Jx_norm_centered

        Jx_norm_centered: This returns the array Jx on the centered lattice same as the electric field.


    '''


    # Normalizing the currents to be deposited
    A          = 1/(number_of_electrons * w_p)

    Jx_norm    = A * Jx_staggered

    # Storing few of the boundary points for further evaluation

    temp1 = Jx_norm[:,  0].copy()
    temp2 = Jx_norm[:, -1].copy()
    temp3 = Jx_norm[:, -2].copy()


    # Bring the Jx_norm from the staggered grid to centered grid

    Jx_norm_centered     = (0.5) * (Jx_norm + af.shift(Jx_norm, 0, 1))

    # assigning the current density to the boundary points
    Jx_norm_centered[:,  0] = 0.5 * (temp1 + temp2 + temp3)
    Jx_norm_centered[:, -1] = Jx_norm_centered[:,0].copy()

    Jx_norm_centered[:,  1]  = 0.5 * (Jx_norm_centered[:,  0] + Jx_norm_centered[:,  2])
    Jx_norm_centered[:, -2]  = 0.5 * (Jx_norm_centered[:, -1] + Jx_norm_centered[:, -3])

    af.eval(Jx_norm_centered)

    return Jx_norm_centered


# In[13]:

def current_norm_BC_Jy(Jy_staggered, number_of_electrons, w_p):

    '''
    function current_norm_BC(Jx_staggered, number_of_electrons, w_p)
    -----------------------------------------------------------------------
    Input variables: Jx_staggered, number_of_electrons, w_p

        Jx_staggered: This is an array containing the currents deposited on staggered lattice.

        number_of_electrons: Number of macroparticles taken in the domain.

        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------
    returns: Jx_norm_centered

        Jx_norm_centered: This returns the array Jx on the centered lattice same as the electric field.


    '''


    # Normalizing the currents to be deposited
    A          = 1/(number_of_electrons * w_p)

    Jy_norm    = A * Jy_staggered

    # Storing few of the boundary points for further evaluation

    temp1 = Jy_norm[0,  :].copy()
    temp2 = Jy_norm[-1, :].copy()
    temp3 = Jy_norm[-2, :].copy()


    # Bring the Jx_norm from the staggered grid to centered grid

    Jy_norm_centered     = (0.5) * (Jy_norm + af.shift(Jy_norm, 1, 0))

    # assigning the current density to the boundary points
    Jy_norm_centered[0,  :]  = 0.5 * (temp1 + temp2 + temp3)
    Jy_norm_centered[-1, :]  = Jx_norm_centered[0, :].copy()
    # print('Jy_norm_centered is ', Jy_norm_centered.dims())
    # input('check')

    Jy_norm_centered[1, :]  = 0.5 * (Jy_norm_centered[0, :] + Jy_norm_centered[2, :])
    Jy_norm_centered[-2, :] = 0.5 * (Jy_norm_centered[-1, :] + Jy_norm_centered[-3, :])

    af.eval(Jy_norm_centered)

    return Jy_norm_centered


# Using Computational particles
# \begin{align}
# q_{p} = w_{p}q \\
# m_{p} = w_{p}m \\
# \end{align}
# where $q_{p}$ and $m_{p} $is the charge of the macro particle and mass of the macro particle respectively

# In[14]:

# Weighting factor w_p = number of electrons composing the macro particle
# Doesn't affect the the physics of the system (Coarse graining doesn't affect vlasov maxwell system)
# It should be kept such that the number density in the PIC code corresponds
# to the densities found in real plasmas
w_p             = 1


# In[15]:

# Macro Particle parameters
k_boltzmann     = 1
mass_electron   = 1 * w_p
tempertature    = 1
charge_electron = -10 * w_p
charge_ion      = +10 * w_p


# In[16]:

# Setting the length of the domain
length_domain_x = 1
length_domain_y = 1


# In[17]:

# Setting the number of ghost cells
ghost_cells  = 0


# In[18]:

# Setting number of particle in the domain
number_of_electrons = 2000000


# In[19]:

# Initializing the positions and velocities of the particles
positions_x = length_domain_x * np.random.rand(number_of_electrons)
positions_y = length_domain_x * np.random.rand(number_of_electrons)

# setting the mean and standard deviation of the maxwell distribution
# Thermal/mean velocity of macro particles should correspond to
# that of individual electrons in the plasma
mu_x, sigma_x = 0, (k_boltzmann * tempertature / (mass_electron / w_p))

# Initializing the velocitites according to the maxwell distribution

velocity_x = np.random.normal(mu_x, sigma_x, number_of_electrons)
mu_y, sigma_y = 0, 0.000000000000001
velocity_y = np.random.normal(mu_y, sigma_y, number_of_electrons)


# In[20]:

# Divisions in x grid
divisions_domain_x = 100
divisions_domain_y = 2

# initializing the x grid
x_grid  = np.linspace(    0,  length_domain_x,  divisions_domain_x + 1,    endpoint=True,   dtype = np.double   )

dx = x_grid[1] - x_grid[0]

x_right = x_grid + dx/2

# initializing the x grid
y_grid = np.linspace( 0, length_domain_y, divisions_domain_y + 1, endpoint=True, dtype = np.double )

# dx, dy is the distance between consecutive grid nodes along x and y


dy = y_grid[1] - y_grid[0]

y_top  = y_grid + dy/2



# In[21]:

# Setting the amplitude for perturbation
N_divisions_x         = divisions_domain_x
N_divisions_y         = divisions_domain_y
amplitude_perturbed = 0.5
k_x                 = 2 * np.pi
k_y                 = 0 #2 * np.pi

# Initializing the perturbation
positions_x, positions_y   = set_up_perturbation(     positions_x,                                                      positions_y,                                                      number_of_electrons,                                                      N_divisions_x,                                                      N_divisions_y,                                                      amplitude_perturbed,                                                      k_x,                                                      k_y,                                                      length_domain_x,                                                      length_domain_y,                                                      dx,                                                      dy                                                 )


# In[22]:

# Converting to arrayfire arrays
positions_x  = af.to_array(positions_x)
positions_y  = af.to_array(positions_y)
velocity_x   = af.to_array(velocity_x)
velocity_y   = af.to_array(velocity_y)
x_grid       = af.to_array(x_grid)
y_grid       = af.to_array(y_grid)
x_right      = af.to_array(x_right)
y_top        = af.to_array(y_top)

# In[23]:

# Plotting the initial density perturbation

position_grid   = np.linspace(0, 1, divisions_domain_x)
number_electrons_in_bins, b  = np.histogram(np.array(positions_x), bins=(100),range = (0, length_domain_x)   )

number_density     = (number_electrons_in_bins / (number_of_electrons / (divisions_domain_x)))


# In[24]:

# print(number_density)


# In[25]:

# # Plotting the initial density perturbation
# pl.plot(number_density, label = r'$n$')
# pl.title(r'$\mathrm{Initial\;density\;perturbation}$')
# pl.xlabel('$x$')
# pl.ylabel(r'$n_{electrons}(x)$')
# # pl.ylim(0.0,2.0)
# pl.show()
# pl.clf()


# In[26]:

# Time parameters
start_time = 0

end_time   = 3

dt         = 0.002

time       = np.arange(    start_time,   end_time + dt,   dt, dtype = np.double  )


# In[27]:

# Some variables for storing data
Ex_max       = np.zeros(len(time), dtype = np.double)


# In[28]:


# Charge deposition using linear weighting scheme

rho_electrons  = cloud_charge_deposition(charge_electron,                            number_of_electrons,                            positions_x,                            positions_y,                            x_grid,                            y_grid,                            charge_b1_depositor,                            ghost_cells,                            length_domain_x,                            length_domain_y,                            dx,                            dy                           )

rho_initial    = norm_background_ions(rho_electrons, number_of_electrons, w_p)


# In[29]:

# rho_initial = np.array(rho_initial)
# print(rho_initial[0].shape)
# print(dy)


# In[30]:

# # plotting intial rho in the system considering background ions
pl.plot(np.array(rho_initial)[2])
pl.show()
pl.clf()


# In[31]:

# rho_kspace = af.fft2(rho_initial)
# rho_initial = af.ifft2(rho_kspace)
# pl.plot(np.array(rho_initial)[2])
# pl.show()
# pl.clf()


# In[32]:

# Computing initial potential and electric field
Ex_initial, Ey_initial = fft_poisson(rho_initial, dx, dy)

# In[35]:

# Plotting the Electric field in the system for the given initial conditions
pl.plot(np.array(Ex_initial)[0], label = 'Ex numerical')
pl.xlabel('x')
pl.ylabel('$E_{x}$')
pl.title('$\mathrm{Initial\;conditions\;E_{x}}$')
pl.show()
pl.clf()


# The following cell block determines $v(\frac{\Delta t}{2})$:
# \begin{align}
# v(\frac{\Delta t}{2}) = v(t = 0) + E_{x}\left(x(\frac{\Delta t}{2})\right)\frac{\Delta t}{2}
# \end{align}

# In[36]:

# This cell block is to obtain v at (t = 0.5dt) to implement the verlet algorithm.

positions_x_half = positions_x + velocity_x * dt/2
positions_y_half = positions_y + velocity_y * dt/2

# Periodic Boundary conditions for particles

positions_x_half, positions_y_half = periodic_particles(positions_x_half, positions_y_half,   length_domain_x, length_domain_y )

# Finding interpolant fractions for the positions

fracs_Ex_x, fracs_Ex_y = fraction_finder(positions_x_half, positions_y_half,x_right, y_grid, dx, dy)

fracs_Ey_x, fracs_Ey_y = fraction_finder(positions_x_half, positions_y_half,  x_grid, y_top, dx, dy)
# Interpolating the fields at each particle

Ex_particle = af.signal.approx2(Ex_initial, fracs_Ex_y, fracs_Ex_x)

Ey_particle = af.signal.approx2(Ey_initial, fracs_Ey_y, fracs_Ey_x)

# Updating the velocity using the interpolated Electric fields to find v at (t = 0.5dt)

velocity_x = velocity_x  + (Ex_particle * charge_electron / mass_electron ) * dt/2
velocity_y = velocity_y  + (Ey_particle * charge_electron / mass_electron ) * dt/2

Ex = Ex_initial.copy()
Ey = Ey_initial.copy()


# In[ ]:

for time_index in range(len(time)):
    if(time_index%50 ==0):
        print('Computing for time = ', time_index * dt)


    # Updating the positions of particle using the velocites (Verlet algorithm)
    # velocity at t = (n + 1/2) dt, positions_x at t = (n)dt and positions_x_new
    # at t = (n+1)dt
    positions_x_new = positions_x + velocity_x * dt
    positions_y_new = positions_y + velocity_y * dt

    # Periodic Boundary conditions for particles
    positions_x_new, positions_y_new = periodic_particles(   positions_x_new, positions_y_new, length_domain_x, length_domain_y )

    # Computing the current densities on the staggered grid provided by Umeda's scheme

    Jx_staggered, Jy_staggered = Umeda_2003( charge_electron,\
                                             number_of_electrons,\
                                             positions_x, positions_y,\
                                             velocity_x, velocity_y,\
                                             x_grid, y_grid,\
                                             ghost_cells,\
                                             length_domain_x, length_domain_y,\
                                             dx, dy,\
                                             dt \
                                            )

    # Normalizing and evaluation of current densities on the centered grid

    Jx_norm_centered = current_norm_BC_Jx(Jx_staggered, number_of_electrons, w_p)
    Jy_norm_centered = current_norm_BC_Jy(Jy_staggered, number_of_electrons, w_p)

    # Evolving electric fields using currents

    Ex_updated = Ex - Jx_norm_centered * dt
    Ey_updated = Ey - Jy_norm_centered * dt

    # Interpolating the fields at each particle
    # calculating the interpolation fraction needed for arrayfire's 1D interpolation
    fracs_Ex_x, fracs_Ex_y = fraction_finder(positions_x_new, positions_y_new, x_right, y_grid, dx, dy)

    fracs_Ey_x, fracs_Ey_y = fraction_finder(positions_x_new, positions_y_new,  x_grid, y_top, dx, dy )

    Ex_particle = af.signal.approx2(Ex_updated, fracs_Ex_y, fracs_Ex_x)

    Ey_particle = af.signal.approx2(Ey_updated, fracs_Ey_y, fracs_Ey_x)

    # Updating the velocity using the interpolated Electric fields

    velocity_x_new = velocity_x  + (Ex_particle * charge_electron / mass_electron ) * dt
    velocity_y_new = velocity_y  + (Ey_particle * charge_electron / mass_electron ) * dt

    # Saving the Electric fields for plotting

    Ex_max[time_index]       = (af.max(af.abs(Ex)))

    # Saving the updated velocites for the next timestep
    velocity_x  = velocity_x_new.copy()
    positions_x = positions_x_new.copy()
    Ex          = Ex_updated.copy()
    Ey          = Ey_updated.copy()


# ### Comparing the plots from both Linear theory calculations and Electrostatic PIC below

# In[ ]:

# Reading data generated by the Cheng Knorr code
h5f = h5py.File('CK_256.h5', 'r')
Ex_max_CK = h5f['max_E'][:]
Ex_max_CK = np.abs(Ex_max_CK)
h5f.close()
time_CK = np.linspace(0,3, len(Ex_max_CK))


# In[ ]:

# Reading data generated by the Cheng Knorr code
h5f = h5py.File('LT.h5', 'r')
Ex_max_LT = h5f['Ex_amp'][:]
Ex_max_LT = np.abs(Ex_max_LT)
time_LT   = h5f['time_LT'][:]
h5f.close()


# In[ ]:

time_grid = np.linspace(0, time[-1], len(Ex_max))
pl.plot(time_grid, Ex_max , label = r'$\mathrm{PIC}$')
pl.plot(time_LT, Ex_max_LT,'--',lw = 3,label = '$\mathrm{LT}$')
pl.plot(time_CK, Ex_max_CK, label = '$\mathrm{Cheng\;Knorr}$')
# print('(abs(Ex_amplitude[0])) is ',(abs(Ex_amplitude[0])))
# print('(abs(Ex_max[0])) is ',(abs(Ex_max[0])))
pl.xlabel('$t$')
pl.ylabel('$\mathrm{MAX}(|E_{x}|)$')
pl.legend()
pl.show()
pl.savefig('MaxE_LT.png')
pl.clf()


# In[ ]:

pl.semilogy(time_grid, Ex_max ,label = r'$\mathrm{PIC}$')
pl.semilogy(time_LT, Ex_max_LT,'--',lw = 3,label = '$\mathrm{LT}$')
pl.semilogy(time_CK,Ex_max_CK, label = '$\mathrm{Cheng\;Knorr}$')
pl.legend()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{MAX}(|E_{x}|)$')
pl.show()
pl.savefig('MaxE_semilogy_LT.png')
pl.clf()


# In[ ]:

pl.loglog(time_grid, Ex_max ,label = r'$\mathrm{PIC}$')
pl.loglog(time_LT, Ex_max_LT,'--',lw = 3,label = '$\mathrm{LT}$')
pl.semilogy(time_CK,Ex_max_CK, label = '$\mathrm{Cheng\;Knorr}$')
pl.legend()
pl.xlabel('$t$')
pl.ylabel('$\mathrm{MAX}(|E_{x}|)$')
pl.show()
pl.savefig('MaxE_loglog_LT.png')
pl.clf()


# In[ ]:




#
