import numpy as np
import arrayfire as af

af.set_backend('cpu')

def periodic_ghost(field, ghost_cells):

    '''
    function periodic_ghost(field, ghost_cells)
    -----------------------------------------------------------------------
    Input variables: field, ghost_cells

    field: An 2 dimensinal array representing an input field.(columns--->x, rows---->y)

    ghost_cells: Number of ghost cells taken in the domain

    -----------------------------------------------------------------------
    returns: field
    This function returns the modified field with appropriate values assigned to the ghost nodes to 
    ensure periodicity in the field.

    '''

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

    af.eval(field)

    return field













# b1 charge depositor
def TSC_charge_deposition_2D(charge_electron,\
                             positions_x, positions_y,\
                             x_grid, y_grid,\
                             ghost_cells,\
                             length_domain_x, length_domain_y\
                            ):
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

    base_indices_x = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    base_indices_y = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)

    dx = af.sum(x_grid[1] - x_grid[0])
    dy = af.sum(y_grid[1] - y_grid[0])


    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    y_zone = (((af.abs(positions_y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))


    temp = af.where(af.abs(positions_x-x_grid[x_zone])<af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices_x[temp] = x_zone[temp]

    temp = af.where(af.abs(positions_x - x_grid[x_zone])>=af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices_x[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)    




    temp = af.where(af.abs(positions_y-y_grid[y_zone])<af.abs(positions_y-y_grid[y_zone + 1]))

    if(temp.elements()>0):
        base_indices_y[temp] = y_zone[temp]

    temp = af.where(af.abs(positions_y - y_grid[y_zone])>=af.abs(positions_y-x_grid[y_zone + 1]))

    if(temp.elements()>0):
        base_indices_y[temp] = (y_zone[temp] + 1).as_type(af.Dtype.u32)  


    base_indices_minus     = (base_indices_x - 1).as_type(af.Dtype.u32)    
    base_indices_plus      = (base_indices_x + 1).as_type(af.Dtype.u32)    


    index_list_x = af.join( 1, base_indices_minus, base_indices_x, base_indices_plus)


    base_indices_minus     = (base_indices_y - 1).as_type(af.Dtype.u32)    
    base_indices_plus      = (base_indices_y + 1).as_type(af.Dtype.u32)    


    index_list_y = af.join( 1, base_indices_minus, base_indices_y, base_indices_plus)



    positions_x_3x        = af.join( 0,positions_x, positions_x, positions_x)

    positions_y_3x        = af.join( 0,positions_y, positions_y, positions_y)



    distance_nodes_x = x_grid[af.flat(index_list_x)]

    distance_nodes_y = y_grid[af.flat(index_list_y)]


    W_x = 0 * distance_nodes_x.copy()
    W_y = 0 * distance_nodes_y.copy()


    # Determining weights in x direction

    temp = af.where(af.abs(distance_nodes_x - positions_x_3x) < (0.5*dx) )

    if(temp.elements()>0):
        W_x[temp] = 0.75 - (af.abs(distance_nodes_x[temp] - positions_x_3x[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_x - positions_x_3x) >= (0.5*dx) )\
                     * (af.abs(distance_nodes_x - positions_x_3x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_x[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_x[temp] - positions_x_3x[temp])/dx))**2



    # Determining weights in y direction

    temp = af.where(af.abs(distance_nodes_y - positions_y_3x) < (0.5*dx) )

    if(temp.elements()>0):
        W_y[temp] = 0.75 - (af.abs(distance_nodes_y[temp] - positions_y_3x[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_y - positions_y_3x) >= (0.5*dx) )\
                     * (af.abs(distance_nodes_y - positions_y_3x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_y[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_y[temp] - positions_y_3x[temp])/dx))**2



    W_x = af.data.moddims(W_x, positions_x.elements(), 3)
    W_y = af.data.moddims(W_y, positions_x.elements(), 3)

    W_x = af.tile(W_x, 1, 1, 3)
    

    W_y = af.tile(W_y, 1, 1, 3)

    W_y = af.reorder(W_y, 0, 2, 1)

    # Determining the final weight matrix

    W = W_x * W_y
    W = af.flat(W)

    # Determining the x indices for charge deposition
    x_charge_zone = af.flat(af.tile(index_list_x, 1, 1, 3))

    # Determining the y indices for charge deposition
    y_charge_zone = af.tile(index_list_y, 1, 1, 3)
    y_charge_zone = af.flat(af.reorder(y_charge_zone, 0, 2, 1))


    # Determining the charges to depositied for the respective x and y indices
    charges = (charge_electron * W)/(dx * dy)

    af.eval(x_charge_zone, y_charge_zone)
    af.eval(charges)


    return x_charge_zone, y_charge_zone, charges





def   charge_deposition(charge_electron,\
                        positions_x,\
                        positions_y,\
                        x_grid,\
                        y_grid,\
                        shape_function,\
                        ghost_cells,\
                        length_domain_x,\
                        length_domain_y,\
                        dx,\
                        dy\
                       ):

    '''
    function cloud_charge_deposition(   charge,\
                                        number_of_electrons,\
                                        positions_x,\
                                        positions_y,\
                                        x_grid,\
                                        y_grid,\
                                        shape_function,\
                                        ghost_cells,\
                                        length_domain_x,\
                                        length_domain_y,\
                                        dx,\
                                        dy\
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
        
        length_domain_x, length_domain_y: This is the length of the domain in x and y.

    -----------------------------------------------------------------------  
    returns: rho
    
        rho: This is an array containing the charges deposited at the density grid nodes.    
    '''
    
    elements = x_grid.elements()*y_grid.elements()

    rho_x_indices, \
    rho_y_indices, \
    rho_values_at_these_indices = shape_function(charge_electron,positions_x, positions_y,\
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

    # Periodic BC's for charge deposition
    # Adding the charge deposited from other side of the grid 

    
    # rho[ghost_cells, :]  = rho[-1 - ghost_cells, :] + rho[ghost_cells, :]
    # rho[-1 - ghost_cells, :] = rho[ghost_cells, :].copy()
    # rho[:, ghost_cells]  = rho[:, -1 - ghost_cells] + rho[:, ghost_cells]
    # rho[:, -1 - ghost_cells] = rho[:, ghost_cells].copy()   
    

    # rho = periodic_ghost(rho, ghost_cells)
    
    af.eval(rho)

    return rho





def indices_and_currents_TSC_2D( charge_electron, positions_x, positions_y, velocity_x, velocity_y,\
                            x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt  ):
    
    positions_x_new     = positions_x + velocity_x * dt
    positions_y_new     = positions_y + velocity_y * dt

    base_indices_x = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    base_indices_y = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)

    dx = af.sum(x_grid[1] - x_grid[0])
    dy = af.sum(y_grid[1] - y_grid[0])


    # S0
    ############################################################################################################
    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    y_zone = (((af.abs(positions_y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))


    temp = af.where(af.abs(positions_x-x_grid[x_zone])<af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices_x[temp] = x_zone[temp]

    temp = af.where(af.abs(positions_x - x_grid[x_zone])>=af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices_x[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)    




    temp = af.where(af.abs(positions_y-y_grid[y_zone])<af.abs(positions_y-y_grid[y_zone + 1]))

    if(temp.elements()>0):
        base_indices_y[temp] = y_zone[temp]

    temp = af.where(af.abs(positions_y - y_grid[y_zone])>=af.abs(positions_y-x_grid[y_zone + 1]))

    if(temp.elements()>0):
        base_indices_y[temp] = (y_zone[temp] + 1).as_type(af.Dtype.u32)  



    base_indices_minus_two = (base_indices_x - 2).as_type(af.Dtype.u32)    
    base_indices_minus     = (base_indices_x - 1).as_type(af.Dtype.u32)    
    base_indices_plus      = (base_indices_x + 1).as_type(af.Dtype.u32)    
    base_indices_plus_two  = (base_indices_x + 2).as_type(af.Dtype.u32)    



    index_list_x = af.join( 1,\
                             af.join(1, base_indices_minus_two, base_indices_minus, base_indices_x),\
                             af.join(1, base_indices_plus, base_indices_plus_two),\
                          )




    base_indices_minus_two = (base_indices_y - 2).as_type(af.Dtype.u32)    
    base_indices_minus     = (base_indices_y - 1).as_type(af.Dtype.u32)    
    base_indices_plus      = (base_indices_y + 1).as_type(af.Dtype.u32)    
    base_indices_plus_two  = (base_indices_y + 2).as_type(af.Dtype.u32)     


    index_list_y = af.join( 1,\
                             af.join(1, base_indices_minus_two, base_indices_minus, base_indices_y),\
                             af.join(1, base_indices_plus, base_indices_plus_two),\
                          )



    positions_x_5x        = af.join( 0,\
                                     af.join(0, positions_x, positions_x, positions_x),\
                                     af.join(0, positions_x, positions_x),\
                                   )

    positions_y_5x        = af.join( 0,\
                                     af.join(0, positions_y, positions_y, positions_y),\
                                     af.join(0, positions_y, positions_y),\
                                   )




    # Determining S0 for positions at t = n * dt


    distance_nodes_x = x_grid[af.flat(index_list_x)]

    distance_nodes_y = y_grid[af.flat(index_list_y)]


    W_x = 0 * distance_nodes_x.copy()
    W_y = 0 * distance_nodes_y.copy()


    # Determining weights in x direction

    temp = af.where(af.abs(distance_nodes_x - positions_x_5x) < (0.5*dx) )

    if(temp.elements()>0):
        W_x[temp] = 0.75 - (af.abs(distance_nodes_x[temp] - positions_x_5x[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_x - positions_x_5x) >= (0.5*dx) )\
                     * (af.abs(distance_nodes_x - positions_x_5x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_x[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_x[temp] - positions_x_5x[temp])/dx))**2



    # Determining weights in y direction

    temp = af.where(af.abs(distance_nodes_y - positions_y_5x) < (0.5*dy) )

    if(temp.elements()>0):
        W_y[temp] = 0.75 - (af.abs(distance_nodes_y[temp] - positions_y_5x[temp])/dy)**2

    temp = af.where((af.abs(distance_nodes_y - positions_y_5x) >= (0.5*dy) )\
                     * (af.abs(distance_nodes_y - positions_y_5x) < (1.5 * dy) )\
                   )

    if(temp.elements()>0):
        W_y[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_y[temp] - positions_y_5x[temp])/dy))**2



    W_x = af.data.moddims(W_x, positions_x.elements(), 5)
    W_y = af.data.moddims(W_y, positions_y.elements(), 5)

    # print('S0 W_y is ', W_y)

    S0_x = af.tile(W_x, 1, 1, 5)
    

    S0_y = af.tile(W_y, 1, 1, 5)


    S0_y = af.reorder(S0_y, 0, 2, 1)



    #S1
    ###################################################################################################

    positions_x_5x_new    = af.join( 0,\
                                     af.join(0, positions_x_new, positions_x_new, positions_x_new),\
                                     af.join(0, positions_x_new, positions_x_new),\
                                   )

    positions_y_5x_new    = af.join( 0,\
                                     af.join(0, positions_y_new, positions_y_new, positions_y_new),\
                                     af.join(0, positions_y_new, positions_y_new),\
                                   )




    # Determining S0 for positions at t = n * dt

    W_x = 0 * distance_nodes_x.copy()
    W_y = 0 * distance_nodes_y.copy()


    # Determining weights in x direction

    temp = af.where(af.abs(distance_nodes_x - positions_x_5x_new) < (0.5*dx) )

    if(temp.elements()>0):
        W_x[temp] = 0.75 - (af.abs(distance_nodes_x[temp] - positions_x_5x_new[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_x - positions_x_5x_new) >= (0.5*dx) )\
                     * (af.abs(distance_nodes_x - positions_x_5x_new) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_x[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_x[temp] - positions_x_5x_new[temp])/dx))**2



    # Determining weights in y direction

    temp = af.where(af.abs(distance_nodes_y - positions_y_5x_new) < (0.5*dy) )

    if(temp.elements()>0):
        W_y[temp] = 0.75 - (af.abs(distance_nodes_y[temp] - positions_y_5x_new[temp])/dy)**2

    temp = af.where((af.abs(distance_nodes_y - positions_y_5x_new) >= (0.5*dy) )\
                     * (af.abs(distance_nodes_y - positions_y_5x_new) < (1.5 * dy) )\
                   )

    if(temp.elements()>0):
        W_y[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_y[temp] - positions_y_5x_new[temp])/dy))**2



    W_x = af.data.moddims(W_x, positions_x.elements(), 5)
    W_y = af.data.moddims(W_y, positions_x.elements(), 5)

    # print('S1 W_y is ', W_y)

    S1_x = af.tile(W_x, 1, 1, 5)
    

    S1_y = af.tile(W_y, 1, 1, 5)

    S1_y = af.reorder(S1_y, 0, 2, 1)

    # print('S1_x is ', S1_x)
    # print('S0_x is ', S0_x)

    # print('S1_y is ', S1_y)
    # print('S0_y is ', S0_y)


    ###############################################################################################

    # Determining the final weight matrix in 3D


    W_x = (S1_x - S0_x) * (S0_y + (0.5 *(S1_y - S0_y)) )


    W_y = (S1_y - S0_y) * (S0_x + (0.5 *(S1_x - S0_x)) )


    ###############################################################################################




    Jx = af.data.constant(0, positions_x.elements(), 5, 5)
    Jy = af.data.constant(0, positions_x.elements(), 5, 5)


    Jx[:, 0, :] = -1 * charge_electron * (dx/dt) * W_x[:, 0, :].copy()
    Jx[:, 1, :] = Jx[:, 0, :] + -1 * charge_electron * (dx/dt) * W_x[:, 1, :].copy()
    Jx[:, 2, :] = Jx[:, 1, :] + -1 * charge_electron * (dx/dt) * W_x[:, 2, :].copy()
    Jx[:, 3, :] = Jx[:, 2, :] + -1 * charge_electron * (dx/dt) * W_x[:, 3, :].copy()
    Jx[:, 4, :] = Jx[:, 3, :] + -1 * charge_electron * (dx/dt) * W_x[:, 4, :].copy()

    Jx = (1/(dx * dy)) * Jx


    Jy[:, :, 0] = -1 * charge_electron * (dy/dt) * W_y[:, :, 0].copy()
    Jy[:, :, 1] = Jy[:, :, 0] + -1 * charge_electron * (dy/dt) * W_y[:, :, 1].copy()
    Jy[:, :, 2] = Jy[:, :, 1] + -1 * charge_electron * (dy/dt) * W_y[:, :, 2].copy()
    Jy[:, :, 3] = Jy[:, :, 2] + -1 * charge_electron * (dy/dt) * W_y[:, :, 3].copy()
    Jy[:, :, 4] = Jy[:, :, 3] + -1 * charge_electron * (dy/dt) * W_y[:, :, 4].copy()

    Jy = (1/(dx * dy)) * Jy



    # Determining the x indices for charge deposition
    index_list_x_Jx = af.flat(af.tile(index_list_x, 1, 1, 5))

    # Determining the y indices for charge deposition
    y_current_zone = af.tile(index_list_y, 1, 1, 5)
    index_list_y_Jx = af.flat(af.reorder(y_current_zone, 0, 2, 1))


    currents_Jx = af.flat(Jx)

    # Determining the x indices for charge deposition
    index_list_x_Jy = af.flat(af.tile(index_list_x, 1, 1, 5))

    # Determining the y indices for charge deposition
    y_current_zone = af.tile(index_list_y, 1, 1, 5)
    index_list_y_Jy = af.flat(af.reorder(y_current_zone, 0, 2, 1))


    currents_Jy = af.flat(Jy)

    af.eval(index_list_x_Jx, index_list_y_Jx)
    af.eval(index_list_x_Jy, index_list_y_Jy)
    af.eval(currents_Jx, currents_Jy)


    return index_list_x_Jx, index_list_y_Jx, currents_Jx,\
           index_list_x_Jy, index_list_y_Jy,\
           currents_Jy














# def indices_and_currents_NGP_2D(charge_electron, positions_x, velocity_x, dt, x_grid, dx, ghost_cells):

#     positions_x_new     = positions_x + velocity_x * dt
#     number_of_electrons = positions_x.elements()


#     base_indices = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
#     Jx           = af.data.constant(0, positions_x.elements(), 3, dtype=af.Dtype.f64)


#     x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))

#     temp = af.where(af.abs(positions_x-x_grid[x_zone])<af.abs(positions_x-x_grid[x_zone + 1]))

#     if(temp.elements()>0):
#         base_indices[temp] = x_zone[temp]

#     temp = af.where(af.abs(positions_x - x_grid[x_zone])>=af.abs(positions_x-x_grid[x_zone + 1]))

#     if(temp.elements()>0):
#         base_indices[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)

#     S0 = af.data.constant(0, 3 * positions_x.elements(), dtype=af.Dtype.f64)
#     S1 = S0.copy()

#     base_indices_plus  = (base_indices + 1 ).as_type(af.Dtype.u32)
#     base_indices_minus = (base_indices - 1 ).as_type(af.Dtype.u32)

#     index_list = af.join(0, base_indices_minus, base_indices, base_indices_plus)


#     # Computing S(base_indices - 1, base_indices, base_indices + 1)

#     positions_x_3x = af.join(0, positions_x, positions_x, positions_x)
#     positions_x_new_3x = af.join(0, positions_x_new, positions_x_new, positions_x_new)

#     temp = af.where(af.abs(positions_x_3x - x_grid[index_list]) < dx/2)

#     if(temp.elements()>0):
#         S0[temp] = 1

#     temp = af.where(af.abs(positions_x_new_3x - x_grid[index_list]) < dx/2)

#     if(temp.elements()>0):
#         S1[temp] = 1


#     # Computing DS
#     DS = S1 - S0

#     DS = af.data.moddims(DS, number_of_electrons, 3)

#     # Computing weights

#     W = DS.copy()

#     Jx[:, 0] = -1 * charge_electron * velocity_x * W[:, 0].copy()

#     Jx[:, 1] = Jx[:, 0] + (-1 * charge_electron * velocity_x * W[:, 1].copy())

#     Jx[:, 2] = Jx[:, 1] + (-1 * charge_electron * velocity_x * W[:, 2].copy())


#     # Flattening the current matrix
#     currents = af.flat(Jx)

#     af.eval(index_list)
#     af.eval(currents)

#     temp = af.where(af.abs(currents) > 0)

#     index_list  = index_list[temp].copy()
#     currents    = currents[temp].copy()

#     af.eval(Jx_x_indices, Jx_y_indices, Jy_x_indices, Jy_y_indices)

#     af.eval(Jx_values_at_these_indices, Jy_values_at_these_indices)

#     return Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
#            Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices



















def Jx_Esirkepov_2D(   charge_electron,\
                       number_of_electrons,\
                       positions_x ,positions_y,\
                       velocities_x, velocities_y,\
                       x_grid, y_grid,\
                       ghost_cells,\
                       length_domain_x, length_domain_y,\
                       dx, dy,\
                       dt\
                   ):
    

    elements = x_grid.elements() * y_grid.elements()

    Jx_x_indices, Jx_y_indices,\
    Jx_values_at_these_indices,\
    Jy_x_indices, Jy_y_indices,\
    Jy_values_at_these_indices = indices_and_currents_TSC_2D(charge_electron,\
                                                     positions_x, positions_y,\
                                                     velocities_x, velocities_y,\
                                                     x_grid, y_grid,\
                                                     ghost_cells,\
                                                     length_domain_x, length_domain_y,\
                                                     dt\
                                                   )
    
    # Current deposition using numpy's histogram
    input_indices = (Jx_x_indices*(y_grid.elements()) + Jx_y_indices)
    
    # Computing Jx_Yee
    
    Jx_Yee, temp = np.histogram(  input_indices,\
                                  bins=elements,\
                                  range=(0, elements),\
                                  weights=Jx_values_at_these_indices\
                                 )
    
    Jx_Yee = af.data.moddims(af.to_array(Jx_Yee), y_grid.elements(), x_grid.elements())
    
    # Computing Jy_Yee
    
    input_indices = (Jy_x_indices*(y_grid.elements()) + Jy_y_indices)
    
    Jy_Yee, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=Jy_values_at_these_indices\
                                     )
    
    Jy_Yee = af.data.moddims(af.to_array(Jy_Yee), y_grid.elements(), x_grid.elements())

    af.eval(Jx_Yee, Jy_Yee)

    return Jx_Yee, Jy_Yee

















no_of_electrons = 1
charge_electron = 1



positions_x     = af.Array([0.5]).as_type(af.Dtype.f64)
positions_y     = af.Array([0.5]).as_type(af.Dtype.f64)


velocity_x      = af.Array([0.0]).as_type(af.Dtype.f64)
velocity_y      = af.Array([-1.0]).as_type(af.Dtype.f64)


length_domain_x = 1.0
length_domain_y = 1.0

ghost_cells     = 2


divisions_domain_x = 5
divisions_domain_y = 5

dx = length_domain_x / divisions_domain_x
dy = length_domain_y / divisions_domain_y


x_grid = np.linspace(    0 - ghost_cells * dx,\
                         length_domain_x + ghost_cells * dx, \
                         divisions_domain_x + 1 + 2 * ghost_cells,\
                         endpoint=True,\
                         dtype = np.double\
                    )

y_grid = np.linspace(    0 - ghost_cells * dx,\
                         length_domain_y + ghost_cells * dx, \
                         divisions_domain_y + 1 + 2 * ghost_cells,\
                         endpoint=True,\
                         dtype = np.double\
                    )

x_grid = af.to_array(x_grid)
y_grid = af.to_array(y_grid)



dt              = 0.2

dx              = af.sum(x_grid[1] - x_grid[0])
dy              = af.sum(y_grid[1] - y_grid[0])


x_right         = x_grid + dx/2
y_top           = y_grid + dy/2



Jx_Yee, Jy_Yee  =   Jx_Esirkepov_2D(   charge_electron,\
                                       no_of_electrons,\
                                       positions_x ,positions_y,\
                                       velocity_x, velocity_y,\
                                       x_grid, y_grid,\
                                       ghost_cells,\
                                       length_domain_x, length_domain_y,\
                                       dx, dy,\
                                       dt\
                                   )



print('Jx_staggered is ', Jx_Yee * dx * dy)
print('Jy_staggered is ', Jy_Yee * dx * dy)
print('total current Jy_Yee is', af.sum(Jy_Yee) * dx * dy)




# rho      = charge_deposition(   charge_electron,\
#                                 positions_x,\
#                                 positions_y,\
#                                 x_grid,\
#                                 y_grid,\
#                                 TSC_charge_deposition_2D,\
#                                 ghost_cells,\
#                                 length_domain_x,\
#                                 length_domain_y,\
#                                 dx,\
#                                 dy\
#                             )


# # print('rho is ', (rho * dx * dy)[2:-2, 2:-2])
# # print('rho is ', af.sum((rho * dx * dy)[2:-2, 2:-2]))
# # print('rho is ', (rho * dx * dy))
# print('rho is ', af.sum((rho * dx * dy)))