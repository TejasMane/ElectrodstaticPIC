import numpy as np
import arrayfire as af

def periodic_rho(rho, ghost_cells):
    # Applying periodic BC's for charge depostions from last and first zone since
    # first and the last x_grid point are the same point according to periodic BC's
    # Since first and last point are the same, charge being deposited on the last grid point
    # must also be deposited on the first grid point

    # Periodic BC's for charge deposition

    rho[ghost_cells, :]  = rho[-1 - ghost_cells, :] + rho[ghost_cells, :]
    rho[-1 - ghost_cells, :] = rho[ghost_cells, :].copy()
    rho[:, ghost_cells]  = rho[:, -1 - ghost_cells] + rho[:, ghost_cells]
    rho[:, -1 - ghost_cells] = rho[:, ghost_cells].copy()


    len_x = rho.dims()[1]
    len_y = rho.dims()[0]


    rho[ 0 : ghost_cells, :]            = rho[len_y -1 - 2 * ghost_cells\
                                                  : len_y -1 - 1 * ghost_cells, :\
                                             ]

    rho[ :, 0 : ghost_cells]            = rho[:, len_x -1 - 2 * ghost_cells\
                                                  : len_x -1 - 1 * ghost_cells\
                                             ]

    rho[len_y - ghost_cells : len_y, :] = rho[ghost_cells + 1:\
                                                  2 * ghost_cells + 1, :\
                                             ]

    rho[:, len_x - ghost_cells : len_x] = rho[: , ghost_cells + 1\
                                                  : 2 * ghost_cells + 1\
                                             ]

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

    return rho_normalized

# b1 charge depositor
def charge_b1_depositor(charge_electron,\
                        positions_x, positions_y,\
                        x_grid, y_grid,\
                        ghost_cells,\
                        length_domain_x, length_domain_y\
                       ):
    # b1 charge depositor
    '''
    function charge_b1_depositor(charge_electron,\
                                 positions_x, positions_y,\
                                 x_grid, y_grid,\
                                 ghost_cells,\
                                 length_domain_x, length_domain_x\
                                 )
    -----------------------------------------------------------------------
    Input variables: charge, zone_x, frac_x, x_grid, dx

        charge_electron: This is a scalar denoting the charge of the macro particle in the PIC code.

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in x direction.

        positions_y:  An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in y direction.

        x_grid, y_grid: This is an array denoting the position grid chosen in the PIC simulation in
        x and y directions.

        ghost_cells: This is the number of ghost cells in the domain

        length_domain_x, length_domain_x: This is the length of the domain in x and y.

    -----------------------------------------------------------------------
    returns: rho

        rho: This is an array containing the charges deposited at the density grid nodes.
    '''


    number_of_particles = positions_x.elements()

    x_charge_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)
    y_charge_zone = af.data.constant(0, 4 * number_of_particles, dtype=af.Dtype.u32)

    # calculating the number of grid cells

    nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
    ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

    dx = length_domain_x/nx
    dy = length_domain_y/ny

    # Determining the left(x) and bottom (y) indices of the left bottom corner grid node of
    # the grid cell containing the particle

    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    y_zone = (((af.abs(positions_y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

    x_zone_plus = x_zone + 1
    y_zone_plus = y_zone + 1

    # Calculating the fractions needed for calculating the weights

    dy_by_delta_y            = (1/dy) * (positions_y-y_grid[y_zone])
    dy_by_delta_y_complement = 1 - dy_by_delta_y

    dx_by_delta_x            = (1/dx) * (positions_x - x_grid[x_zone])
    dx_by_delta_x_complement = 1 - dx_by_delta_x

    # Calculating the weights at all corners
    # Order of corners is available on the main thesis document
    # order -----bottom right --->bottom left---->top left-----> top right

    weight_corner1 = dy_by_delta_y_complement * dx_by_delta_x_complement
    weight_corner2 = dy_by_delta_y * dx_by_delta_x_complement
    weight_corner3 = dy_by_delta_y * dx_by_delta_x
    weight_corner4 = dy_by_delta_y_complement * dx_by_delta_x

    charge_by_dxdy = ((charge_electron/(dx*dy)))

    corner1_charge   = weight_corner1 * charge_by_dxdy
    corner2_charge   = weight_corner2 * charge_by_dxdy
    corner3_charge   = weight_corner3 * charge_by_dxdy
    corner4_charge   = weight_corner4 * charge_by_dxdy

    # Concatenating the all the weights for all 4 corners into one vector all_corners_weighted_charge

    all_corners_weighted_charge = af.join(0,corner1_charge, corner2_charge, corner3_charge, corner4_charge)

    # concatenating the x indices into x_charge_zone

    x_charge_zone[0 * number_of_particles : 1 * number_of_particles] = x_zone
    x_charge_zone[1 * number_of_particles : 2 * number_of_particles] = x_zone
    x_charge_zone[2 * number_of_particles : 3 * number_of_particles] = x_zone_plus
    x_charge_zone[3 * number_of_particles : 4 * number_of_particles] = x_zone_plus

    # concatenating the x indices into x_charge_zone

    y_charge_zone[0 * number_of_particles : 1 * number_of_particles] = y_zone
    y_charge_zone[1 * number_of_particles : 2 * number_of_particles] = y_zone_plus
    y_charge_zone[2 * number_of_particles : 3 * number_of_particles] = y_zone_plus
    y_charge_zone[3 * number_of_particles : 4 * number_of_particles] = y_zone


    af.eval(x_charge_zone, y_charge_zone)
    af.eval(all_corners_weighted_charge)

    return x_charge_zone, y_charge_zone, all_corners_weighted_charge

def cloud_charge_deposition(charge_electron,\
                            no_of_particles,\
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
                          ):

    '''
    function cloud_charge_deposition(   charge,\
                                        no_of_particles,\
                                        positions_x,\
                                        positions_y,\
                                        x_grid,\
                                        y_grid,\
                                        shape_function,\
                                        ghost_cells,\
                                        length_domain_x,\
                                        length_domain_x,\
                                        dx,\
                                        dy,\
                                        w_p\
                                   )
    -----------------------------------------------------------------------
    Input variables: charge, zone_x, frac_x, x_grid, dx

        charge_electron: This is a scalar denoting the charge of the macro particle in the PIC code.

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in x direction.

        positions_y:  An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in y direction.

        x_grid, y_grid: This is an array denoting the position grid chosen in the PIC simulation in
        x and y directions.

        shape_function: The weighting scheme used for the charge deposition.

        ghost_cells: This is the number of ghost cells in the domain

        length_domain_x, length_domain_x: This is the length of the domain in x and y.

    -----------------------------------------------------------------------
    returns: rho

        rho: This is an array containing the charges deposited at the density grid nodes.
    '''

    elements = x_grid.elements()*y_grid.elements()

    rho_x_indices, \
    rho_y_indices, \
    rho_values_at_these_indices = charge_b1_depositor(charge_electron,positions_x, positions_y,\
                                                      x_grid, y_grid,\
                                                      ghost_cells, length_domain_x, length_domain_y\
                                                      )

    input_indices = (rho_x_indices*(y_grid.elements())+ rho_y_indices)

    rho, temp = np.histogram(input_indices,\
                             bins=elements,\
                             range=(0, elements),\
                             weights=rho_values_at_these_indices\
                            )

    rho = af.data.moddims(af.to_array(rho), y_grid.elements(), x_grid.elements())


    rho = periodic_rho(rho, ghost_cells)

    norm_background_ions(rho, no_of_particles, w_p)

    af.eval(rho)

    return rho
