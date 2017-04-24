import arrayfire as af
import numpy as np

def periodic_current(Jx_staggered, Jy_staggered, number_of_electrons, ghost_cells, w_p):

    Jx_norm_staggered = current_norm_BC_Jx(Jx_staggered, number_of_electrons, ghost_cells, w_p)
    Jy_norm_staggered = current_norm_BC_Jy(Jy_staggered, number_of_electrons, ghost_cells, w_p)

    af.eval(Jx_norm_staggered, Jy_norm_staggered)

    return Jx_norm_staggered, Jy_norm_staggered



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

def periodic_ghost(field, ghost_cells):
    # Applying periodic BC's for charge depostions from last and first zone since
    # first and the last x_grid point are the same point according to periodic BC's
    # Since first and last point are the same, charge being deposited on the last grid point
    # must also be deposited on the first grid point

    len_x = field.dims()[1]
    len_y = field.dims()[0]


    field[ 0 : ghost_cells, :]            = field[len_y -1 - 2 * ghost_cells\
                                                  : len_y -1 - 1 * ghost_cells, :\
                                                 ]

    field[ :, 0 : ghost_cells]            = field[:, len_x -1 - 2 * ghost_cells\
                                                  : len_x -1 - 1 * ghost_cells\
                                                 ]

    field[len_y - ghost_cells : len_y, :] = field[ghost_cells + 1:\
                                                  2 * ghost_cells + 1, :\
                                                 ]

    field[:, len_x - ghost_cells : len_x] = field[: , ghost_cells + 1\
                                                  : 2 * ghost_cells + 1\
                                                 ]

    return field


def current_norm_BC_Jx(Jx_staggered, number_of_electrons, ghost_cells, w_p):

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

    len_x = Jx_staggered.dims()[1]
    len_y = Jx_staggered.dims()[0]


    # Normalizing the currents to be deposited
    A                  = 1/(number_of_electrons * w_p)

    Jx_norm_staggered  = A * Jx_staggered

    # assigning the current density to the boundary points for periodic boundary conditions
    Jx_norm_staggered[:, ghost_cells]  =   Jx_norm_staggered[:, ghost_cells] \
                                         + Jx_norm_staggered[:, -1 - ghost_cells]

    Jx_norm_staggered[:, -2 - ghost_cells] =   Jx_norm_staggered[:, -2 - ghost_cells] \
                                             + Jx_norm_staggered[:, ghost_cells - 1]

    Jx_norm_staggered[:, -1 - ghost_cells] = Jx_norm_staggered[:, ghost_cells].copy()


    # assigning the current density to the boundary points in top and bottom rows along y direction
    Jx_norm_staggered[ghost_cells, :] = Jx_norm_staggered[ghost_cells, :] + Jx_norm_staggered[-1-ghost_cells, :]
    Jx_norm_staggered[-1-ghost_cells, :] = Jx_norm_staggered[ghost_cells, :].copy()

    # Assigning ghost cell values
    Jx_norm_staggered = periodic_ghost(Jx_norm_staggered, ghost_cells)

    af.eval(Jx_norm_staggered)

    return Jx_norm_staggered


def current_norm_BC_Jy(Jy_staggered, number_of_electrons, ghost_cells, w_p):

    '''
    function current_norm_BC(Jy_staggered, number_of_electrons, w_p)
    -----------------------------------------------------------------------
    Input variables: Jx_staggered, number_of_electrons, w_p

        Jy_staggered: This is an array containing the currents deposited on staggered lattice.

        number_of_electrons: Number of macroparticles taken in the domain.

        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------
    returns: Jy_norm_centered

        Jy_norm_centered: This returns the array Jx on the centered lattice same as the electric field.


    '''

    len_x = Jy_staggered.dims()[1]
    len_y = Jy_staggered.dims()[0]

    # Normalizing the currents to be deposited
    A                  = 1/(number_of_electrons * w_p)

    Jy_norm_staggered  = A * Jy_staggered


    # assigning the current density to the boundary points for periodic boundary conditions
    Jy_norm_staggered[ghost_cells, :]  =   Jy_norm_staggered[ghost_cells, :] \
                                         + Jy_norm_staggered[-1 - ghost_cells, :]

    Jy_norm_staggered[-2 - ghost_cells, :] =   Jy_norm_staggered[-2 - ghost_cells, :] \
                                             + Jy_norm_staggered[ghost_cells - 1, :]

    Jy_norm_staggered[-1 - ghost_cells, :] = Jy_norm_staggered[ghost_cells, :].copy()


    # assigning the current density to the boundary points in left and right columns along x direction
    Jy_norm_staggered[:, ghost_cells] = Jy_norm_staggered[:, ghost_cells] + Jy_norm_staggered[:, -1-ghost_cells]
    Jy_norm_staggered[:, -1-ghost_cells] = Jy_norm_staggered[:, ghost_cells].copy()

    # Assigning ghost cell values
    Jy_norm_staggered = periodic_ghost(Jy_norm_staggered, ghost_cells)


    af.eval(Jy_norm_staggered)

    return Jy_norm_staggered


# Umeda needs x(n), and v(n+0.5dt) for implementation
def Umeda_b1_deposition( charge_particle, positions_x, positions_y, velocity_x, velocity_y,                            x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt  ):

    '''
    A modified Umeda's scheme was implemented to handle a pure one dimensional case.

    function Umeda_b1_deposition( charge, x, velocity_x,\
                                  x_grid, ghost_cells, length_domain_x, dt\
                                )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x,
    dt

        charge: This is an array containing the charges deposited at the density
        grid nodes.

        positions_x: An one dimensional array of size equal to number of particles
        taken in the PIC code. It contains the positions of particles in x direction.

        positions_y:  An one dimensional array of size equal to number of particles
        taken in the PIC code. It contains the positions of particles in y direction.

        velocity_x: An one dimensional array of size equal to number of particles
        taken in the PIC code. It contains the velocities of particles in y direction.

        velocity_y: An one dimensional array of size equal to number of particles
        taken in the PIC code. It contains the velocities of particles in x direction.

        x_grid: This is an array denoting the position grid in x direction chosen
        in the PIC simulation.

        y_grid: This is an array denoting the position grid in y direction chosen
        in the PIC simulation

        ghost_cells: This is the number of ghost cells used in the simulation domain.

        length_domain_x: This is the length of the domain in x direction

        dt: this is the dt/time step chosen in the simulation
    -----------------------------------------------------------------------
    returns: Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
           Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices

        Jx_x_indices: This returns the x indices (columns) of the array where the
        respective currents stored in Jx_values_at_these_indices have to be deposited

        Jx_y_indices: This returns the y indices (rows) of the array where the respective
        currents stored in Jx_values_at_these_indices have to be deposited

        Jx_values_at_these_indices: This is an array containing the currents to be
        deposited.

        Jy_x_indices, Jy_y_indices and Jy_values_at_these_indices are similar to
        Jx_x_indices, Jx_y_indices and Jx_values_at_these_indices for Jy

    For further details on the scheme refer to Umeda's paper provided in the sagemath
    folder as thenaming conventions used in the function use the papers naming convention.
    (x_1, x_2, x_r, F_x etc)

    '''

    x_current_zone = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    y_current_zone = af.data.constant(0, positions_y.elements(), dtype=af.Dtype.u32)

    nx = (x_grid.elements() - 1 - 2 * ghost_cells )  # number of zones
    ny = (y_grid.elements() - 1 - 2 * ghost_cells )  # number of zones

    dx = length_domain_x/nx
    dy = length_domain_y/ny

    # Start location x_1, y_1 at t = n * dt
    # Start location x_2, y_2 at t = (n+1) * dt

    x_1 = (positions_x).as_type(af.Dtype.f64)
    x_2 = (positions_x + (velocity_x * dt)).as_type(af.Dtype.f64)

    y_1 = (positions_y).as_type(af.Dtype.f64)
    y_2 = (positions_y + (velocity_y * dt)).as_type(af.Dtype.f64)

    # Calculation i_1 and i_2, indices of left corners of cells containing the particles
    # at x_1 and x_2 respectively and j_1 and j_2: indices of bottoms of cells containing the particles
    # at y_1 and y_2 respectively

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

    # Computing the relay point (x_r, y_r)


    x_r = af.min(x_r_combined_term, 1)
    y_r = af.min(y_r_combined_term, 1)

    # Calculating the fluxes and the weights

    F_x_1 = charge_particle * (x_r - x_1)/dt
    F_x_2 = charge_particle * (x_2 - x_r)/dt

    F_y_1 = charge_particle * (y_r - y_1)/dt
    F_y_2 = charge_particle * (y_2 - y_r)/dt

    W_x_1 = (x_1 + x_r)/(2 * dx) - i_1
    W_x_2 = (x_2 + x_r)/(2 * dx) - i_2

    W_y_1 = (y_1 + y_r)/(2 * dy) - j_1
    W_y_2 = (y_2 + y_r)/(2 * dy) - j_2

    # computing the charge densities at the grid nodes using the
    # fluxes and the weights

    J_x_1_1 = (1/(dx * dy)) * (F_x_1 * (1 - W_y_1))
    J_x_1_2 = (1/(dx * dy)) * (F_x_1 * (W_y_1))

    J_x_2_1 = (1/(dx * dy)) * (F_x_2 * (1 - W_y_2))
    J_x_2_2 = (1/(dx * dy)) * (F_x_2 * (W_y_2))

    J_y_1_1 = (1/(dx * dy)) * (F_y_1 * (1 - W_x_1))
    J_y_1_2 = (1/(dx * dy)) * (F_y_1 * (W_x_1))

    J_y_2_1 = (1/(dx * dy)) * (F_y_2 * (1 - W_x_2))
    J_y_2_2 = (1/(dx * dy)) * (F_y_2 * (W_x_2))

    # concatenating the x, y indices for Jx

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

    # concatenating the currents at x, y indices for Jx

    Jx_values_at_these_indices = af.join(0,\
                                         J_x_1_1,\
                                         J_x_1_2,\
                                         J_x_2_1,\
                                         J_x_2_2\
                                        )

    # concatenating the x, y indices for Jy

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

    # concatenating the currents at x, y indices for Jx

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


def Umeda_2003( w_p, charge_electron, number_of_electrons, positions_x ,positions_y, velocity_x,\
                velocity_y, x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y,\
                dx, dy, dt\
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

        velocity_x(t = (n+1/2)*dt): An one dimensional array of size equal to number of particles
        taken in the PIC code. It contains the velocities of particles in x direction.

        x_grid: This is an array denoting the position grid chosen in the PIC simulation.

        ghost_cells: This is the number of ghost cells used in the simulation domain..

        length_domain_x: This is the length of the domain in x direction.

        dt: this is the dt/time step chosen in the simulation.
    -----------------------------------------------------------------------
    returns: Jx_staggered, Jy_staggered

        Jx_staggered, Jy_staggered: This returns the array Jx and Jy on their respective staggered yee lattice.


    '''

    elements = x_grid.elements() * y_grid.elements()

    Jx_x_indices, Jx_y_indices,\
    Jx_values_at_these_indices,\
    Jy_x_indices, Jy_y_indices,\
    Jy_values_at_these_indices = Umeda_b1_deposition(charge_electron,\
                                                     positions_x, positions_y,\
                                                     velocity_x, velocity_y,\
                                                     x_grid, y_grid,\
                                                     ghost_cells,\
                                                     length_domain_x, length_domain_y,\
                                                     dt\
                                                   )

    # Current deposition using numpy's histogram
    input_indices = (Jx_x_indices*(y_grid.elements()) + Jx_y_indices)

    Jx_staggered, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=Jx_values_at_these_indices\
                                     )

    Jx_staggered = af.data.moddims(af.to_array(Jx_staggered), y_grid.elements(), x_grid.elements())

    input_indices = (Jy_x_indices*(y_grid.elements()) + Jy_y_indices)

    Jy_staggered, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=Jy_values_at_these_indices\
                                     )

    Jy_staggered = af.data.moddims(af.to_array(Jy_staggered), y_grid.elements(), x_grid.elements())

    Jx_staggered, Jy_staggered = periodic_current(Jx_staggered, Jy_staggered, number_of_electrons, ghost_cells, w_p)

    af.eval(Jx_staggered, Jy_staggered)

    return Jx_staggered, Jy_staggered
