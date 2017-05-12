import numpy as np
import arrayfire as af



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

    temp = af.where(af.abs(distance_nodes_x - positions_x_3x) < dx/2 )

    if(temp.elements()>0):
        W_x[temp] = 0.75 - (af.abs(distance_nodes_x[temp] - positions_x_3x[temp])/dx)**2
        print('W_x[temp] is ', W_x)

    temp = af.where((af.abs(distance_nodes_x - positions_x_3x) >= dx/2 )\
                     * (af.abs(distance_nodes_x - positions_x_3x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_x[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_x[temp] - positions_x_3x[temp])/dx))**2



    # Determining weights in y direction

    temp = af.where(af.abs(distance_nodes_y - positions_y_3x) < dx/2 )

    if(temp.elements()>0):
        W_y[temp] = 0.75 - (af.abs(distance_nodes_y[temp] - positions_y_3x[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_y - positions_y_3x) >= dx/2 )\
                     * (af.abs(distance_nodes_y - positions_y_3x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_y[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_y[temp] - positions_y_3x[temp])/dx))**2



    W_x = af.data.moddims(W_x, positions_x.elements(), 3)
    W_y = af.data.moddims(W_y, positions_x.elements(), 3)

    print('positions_x_3x is ', positions_x_3x)
    print('distance_nodes_x is ', distance_nodes_x)

    print('positions_y_3x is ', positions_y_3x)
    print('distance_nodes_y is ', distance_nodes_y)



    W_x = af.tile(W_x, 1, 1, 3)
    

    W_y = af.tile(W_y, 1, 1, 3)

    W_y = af.reorder(W_y, 0, 2, 1)

    print('W_x is ', W_x)
    print('W_y is ', W_y)

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
    
    rho[ghost_cells, :]  = rho[-1 - ghost_cells, :] + rho[ghost_cells, :]
    rho[-1 - ghost_cells, :] = rho[ghost_cells, :].copy()
    rho[:, ghost_cells]  = rho[:, -1 - ghost_cells] + rho[:, ghost_cells]
    rho[:, -1 - ghost_cells] = rho[:, ghost_cells].copy()   
    
    rho = periodic_ghost(rho, ghost_cells)
    
    af.eval(rho)

    return rho





# def indices_and_currents_TSC_2D( charge_electron, positions_x, positions_y, velocity_x, velocity_y,\
#                             x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt  ):
    
#     positions_x_new     = positions_x + velocity_x * dt
#     positions_y_new     = positions_y + velocity_y * dt


#     number_of_electrons = positions_x.elements()





#     # temp = af.where(af.abs(currents) > 0)

#     # if(temp.elements()>0):
# 	   #  index_list  = index_list[temp].copy()
# 	   #  currents    = currents[temp].copy()

#     af.eval(index_list_x_Jx, index_list_y_Jx)
#     af.eval(index_list_x_Jy, index_list_y_Jy)
#     af.eval(currents_Jx, currents_Jy)


#     return index_list_x_Jx, index_list_y_Jx, currents_Jx,\
#            index_list_x_Jy, index_list_y_Jy,\
#            currents_Jy














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



















# def Jx_Esirkepov_2D(   charge_electron,\
#                        number_of_electrons,\
#                        positions_x ,positions_y,\
#                        velocities_x, velocities_y,\
#                        x_grid, y_grid,\
#                        ghost_cells,\
#                        length_domain_x, length_domain_y,\
#                        dx, dy,\
#                        dt\
#                    ):
    

#     elements = x_grid.elements() * y_grid.elements()

#     Jx_x_indices, Jx_y_indices,\
#     Jx_values_at_these_indices,\
#     Jy_x_indices, Jy_y_indices,\
#     Jy_values_at_these_indices = indices_and_currents_TSC_2D(charge_electron,\
#                                                      positions_x, positions_y,\
#                                                      velocities_x, velocities_y,\
#                                                      x_grid, y_grid,\
#                                                      ghost_cells,\
#                                                      length_domain_x, length_domain_y,\
#                                                      dt\
#                                                    )
    
#     # Current deposition using numpy's histogram
#     input_indices = (Jx_x_indices*(y_grid.elements()) + Jx_y_indices)
    
#     # Computing Jx_Yee
    
#     Jx_Yee, temp = np.histogram(  input_indices,\
#                                   bins=elements,\
#                                   range=(0, elements),\
#                                   weights=Jx_values_at_these_indices\
#                                  )
    
#     Jx_Yee = af.data.moddims(af.to_array(Jx_Yee), y_grid.elements(), x_grid.elements())
    
#     # Computing Jy_Yee
    
#     input_indices = (Jy_x_indices*(y_grid.elements()) + Jy_y_indices)
    
#     Jy_Yee, temp = np.histogram(input_indices,\
#                                       bins=elements,\
#                                       range=(0, elements),\
#                                       weights=Jy_values_at_these_indices\
#                                      )
    
#     Jy_Yee = af.data.moddims(af.to_array(Jy_Yee), y_grid.elements(), x_grid.elements())

#     af.eval(Jx_Yee, Jy_Yee)

#     return Jx_Yee, Jy_Yee

















no_of_electrons = 2
charge_electron = 1

positions_x     = af.Array([0.6, 0.0]).as_type(af.Dtype.f64)
positions_y     = af.Array([0.6, 0.6]).as_type(af.Dtype.f64)

velocity_x      = af.Array([-1.0, -1.0]).as_type(af.Dtype.f64)
velocity_y      = af.Array([-1.0, -1.0]).as_type(af.Dtype.f64)

x_grid          = af.Array([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]).as_type(af.Dtype.f64)
y_grid          = af.Array([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]).as_type(af.Dtype.f64)


dt              = 0.2

dx              = af.sum(x_grid[1] - x_grid[0])
dy              = af.sum(y_grid[1] - y_grid[0])


x_right         = x_grid + dx/2
y_top           = y_grid + dy/2

ghost_cells     = 1

length_domain_x = 1.0
length_domain_y = 1.0


# Jx_staggered  = Jx_Esirkepov_2D(  charge_electron, no_of_electrons, positions_x,\
#                                   velocity_x, x_grid, ghost_cells,\
#                                   length_domain_x, dx, dt\
#                                )



# print('Jx_staggered is ', Jx_staggered)



rho      = charge_deposition(   charge_electron,\
                                positions_x,\
                                positions_y,\
                                x_grid,\
                                y_grid,\
                                TSC_charge_deposition_2D,\
                                ghost_cells,\
                                length_domain_x,\
                                length_domain_y,\
                                dx,\
                                dy\
                            )


print('rho is ', rho * dx * dy)
