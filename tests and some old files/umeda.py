import arrayfire as af
import numpy as np



# Umeda needs x(n), and v(n+0.5dt) for implementation
def Umeda_b1_deposition( charge, x, velocity_required_x,\
                         x_grid, ghost_cells, Lx, dt\
                       ):

    '''
    function Umeda_b1_deposition( charge, x, velocity_required_x,\
                         x_grid, ghost_cells, Lx, dt\
                       )
    -----------------------------------------------------------------------  
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, Lx, dt

        x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        Lx: This is the length of the domain in x direction

    -----------------------------------------------------------------------      
    returns: Jx_x_indices, Jx_values_at_these_indices
        This function returns a array positions_x such that there is a cosine density perturbation 
        of the given amplitude

    '''  
    
    x_current_zone = af.data.constant(0,x.elements(), dtype=af.Dtype.u32)

    nx = (x_grid.elements() - 1 - 2 * ghost_cells )  # number of zones

    dx = Lx/nx

    x_1 = (x).as_type(af.Dtype.f64)
    x_2 = (x + (velocity_required_x * dt)).as_type(af.Dtype.f64)


    i_1 = af.arith.floor( ((af.abs( x_1 - af.sum(x_grid[0])))/dx) - ghost_cells)


    i_2 = af.arith.floor( ((af.abs( x_2 - af.sum(x_grid[0])))/dx) - ghost_cells)


    i_dx = dx * af.join(1, i_1, i_2)

    i_dx_x_avg = af.join(1, af.max(i_dx,1), ((x_1+x_2)/2))

    x_r_term_1 = dx + af.min(i_dx, 1)
    x_r_term_2 = af.max(i_dx_x_avg, 1)

    x_r_combined_term = af.join(1, x_r_term_1, x_r_term_2)

    x_r = af.min(x_r_combined_term, 1)

    # print('x_r is ', x_r)

    F_x_1 = charge * (x_r - x_1)/dt
    F_x_2 = charge * (x_2 - x_r)/dt

    J_x_1_1 = (1/(dx)) * (F_x_1)

    J_x_2_1 = (1/(dx)) * (F_x_2)

    Jx_x_indices = af.join(0, i_1 + ghost_cells, i_2 + ghost_cells)

    Jx_values_at_these_indices = af.join(0, J_x_1_1, J_x_2_1)

    af.eval(Jx_x_indices)
    af.eval(Jx_values_at_these_indices)

    return Jx_x_indices, Jx_values_at_these_indices
    
    
def Umeda_2003(charge,\
               no_of_particles,\
               positions_x,\
               velocities_x,\
               x_center_grid,\
               ghost_cells,\
               Lx,\
               dx,\
               dt\
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
    returns: Jx
        This function returns a array positions_x such that there is a cosine density perturbation 
        of the given amplitude

    '''
    elements = x_center_grid.elements()

    Jx_x_indices, Jx_values_at_these_indices = Umeda_b1_deposition(charge,\
                                                                 positions_x,\
                                                                 velocities_x,\
                                                                 x_center_grid,\
                                                                 ghost_cells,\
                                                                 Lx,\
                                                                 dt\
                                                                )

    input_indices = (Jx_x_indices)
    Jx, temp = np.histogram(input_indices, bins=elements, range=(0, elements), weights=Jx_values_at_these_indices)

    Jx = af.to_array(Jx)
    af.eval(Jx)

    return Jx
    


charge = 1

# no_of_particles = 2
no_of_particles = 1


# positions_x = af.Array([0.7, 0.3])
# positions_y = af.Array([0.7, 0.3])
# positions_z = af.Array([0.5, 0.5])
#
# velocities_x = af.Array([0.1, 0.1])
# velocities_y = af.Array([0.1, 0.1])
# velocities_z = af.Array([0.0, 0.0])

positions_x = af.Array([0.10])

velocities_x = af.Array([-1.0])

x_center_grid = af.Array([ 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 ])

dt = 0.2

dx = 0.2

ghost_cells = 0

Lx = 1.0

Jx = Umeda_2003( charge, no_of_particles, positions_x,\
                         velocities_x,\
                         x_center_grid, ghost_cells,\
                         Lx, dx, dt\
                       )

print('Jx*dx*dy is ', Jx*dx)



def current_norm_BC(Jx, number_of_electrons, w_p):
    
    A          = 1/(number_of_electrons * w_p)
    
    Jx_norm    = A * Jx
    
    temp1 = Jx_norm[0].copy()
    temp2 = Jx_norm[-1].copy()
    temp3 = Jx_norm[-2].copy()
    
    Jx_norm     = (0.5) * (Jx_norm + af.shift(Jx_norm, 1))
    
    Jx_norm[0]  = 0.5 * (temp1 + temp2 + temp3)
    Jx_norm[-1] = Jx_norm[0].copy()

    Jx_norm[1] = 0.5 * (Jx_norm[0] + Jx_norm[2])
    Jx_norm[-2] = 0.5 * (Jx_norm[-1] + Jx_norm[-3])
    
    
    return Jx_norm
    
