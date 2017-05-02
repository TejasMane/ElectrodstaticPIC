import numpy as np
import arrayfire as af



# b1 charge depositor
def TSC_charge_deposition(charge_electron, no_of_electrons, positions_x,\
                  x_grid, ghost_cells,length_domain_x, dx):
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

    base_indices = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)


    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))

    temp = af.where(af.abs(positions_x-x_grid[x_zone])<af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices[temp] = x_zone[temp]

    temp = af.where(af.abs(positions_x - x_grid[x_zone])>=af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)    

    base_indices_minus = (base_indices - 1).as_type(af.Dtype.u32)    
    base_indices_plus  = (base_indices + 1).as_type(af.Dtype.u32)    


    index_list = af.join(0, base_indices_minus, base_indices, base_indices_plus)

    positions_x_3x        = af.join(0, positions_x, positions_x, positions_x)

    distance_nodes = x_grid[index_list]


    W = 0 * positions_x_3x.copy()

    temp = af.where(af.abs(distance_nodes - positions_x_3x) < dx/2 )

    if(temp.elements()>0):
        W[temp] = 0.75 - (af.abs(distance_nodes[temp] - positions_x_3x[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes - positions_x_3x) >= dx/2 )\
                     * (af.abs(distance_nodes - positions_x_3x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W[temp] = 0.5 * (1.5 - (af.abs(distance_nodes[temp] - positions_x_3x[temp])/dx))**2

    charges = (charge_electron * W)/dx

    # Depositing currents using numpy histogram
    input_indices      = index_list
    elements           = x_grid.elements()

    rho, temp = np.histogram(input_indices, bins=elements, range=(0, elements),weights=charges)

    rho = af.to_array(rho)

    af.eval(rho)


    return rho


def indices_and_currents_TSC(charge_electron, positions_x, dt, x_grid, dx, ghost_cells):
    
    positions_x_new     = positions_x + velocity_x * dt
    number_of_electrons = positions_x.elements()


    base_indices = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    Jx           = af.data.constant(0, positions_x.elements(), 5, dtype=af.Dtype.f64)


    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    
    temp   = af.where(af.abs(positions_x-x_grid[x_zone])<af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices[temp] = x_zone[temp]

    temp = af.where(af.abs(positions_x - x_grid[x_zone])>=af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)

    S0 = af.data.constant(0, 5 * positions_x.elements(), dtype=af.Dtype.f64)
    S1 = S0.copy()

    base_indices_plus      = (base_indices + 1 ).as_type(af.Dtype.u32)
    base_indices_plus_two  = (base_indices + 2 ).as_type(af.Dtype.u32)

    base_indices_minus     = (base_indices - 1 ).as_type(af.Dtype.u32)
    base_indices_minus_two = (base_indices - 2 ).as_type(af.Dtype.u32)


    index_list_1 = af.join( 0, base_indices_minus_two, base_indices_minus,\
                          )
    index_list_2 = af.join( 0, base_indices, base_indices_plus,base_indices_plus_two\
                          )

    index_list   = af.join(0, index_list_1, index_list_2)


    # Computing S(base_indices - 1, base_indices, base_indices + 1)

    positions_x_5x = af.join(0, positions_x, positions_x, positions_x)

    positions_x_5x = af.join(0, positions_x_5x, positions_x, positions_x)


    positions_x_new_5x = af.join(0, positions_x_new, positions_x_new, positions_x_new)

    positions_x_new_5x = af.join(0, positions_x_new_5x, positions_x_new, positions_x_new)


    # Computing S0


    temp = af.where(af.abs(positions_x_5x - x_grid[index_list]) < dx/2)

    if(temp.elements()>0):

        S0[temp] = (0.75) - (af.abs(positions_x_5x[temp] - x_grid[index_list[temp]])/dx)**2

    temp = af.where((af.abs(positions_x_5x - x_grid[index_list]) >= 0.5 * dx)\
                     * (af.abs(positions_x_5x - x_grid[index_list]) < 1.5 * dx)\
                   )


    if(temp.elements()>0):

        S0[temp] = 0.5 * (3/2 - af.abs((positions_x_5x[temp] - x_grid[index_list[temp]])/dx) )**2


    # Computing S1


    temp = af.where(af.abs(positions_x_new_5x - x_grid[index_list]) < dx/2)

    if(temp.elements()>0):

        S1[temp] = (3/4) - (af.abs(positions_x_new_5x[temp] - x_grid[index_list[temp]])/dx)**2

    temp = af.where((af.abs(positions_x_new_5x - x_grid[index_list]) >= 0.5 * dx)\
                     * (af.abs(positions_x_new_5x - x_grid[index_list]) < 1.5 * dx)\
                   )


    if(temp.elements()>0):

        S1[temp] = 0.5 * (3/2 - af.abs((positions_x_new_5x[temp] - x_grid[index_list[temp]])/dx) )**2


    # Computing DS


    DS = S1 - S0

    W = af.data.moddims(DS, number_of_electrons, 5)


    # Computing weights

    Jx[:, 0] = -1 * charge_electron * (dx/dt) * W[:, 0].copy()


    Jx[:, 1] = Jx[:, 0] + (-1 * charge_electron * (dx/dt) * W[:, 1].copy()) 

    Jx[:, 2] = Jx[:, 1] + (-1 * charge_electron * (dx/dt) * W[:, 2].copy()) 

    Jx[:, 3] = Jx[:, 2] + (-1 * charge_electron * (dx/dt) * W[:, 3].copy()) 

    Jx[:, 4] = Jx[:, 3] + (-1 * charge_electron * (dx/dt) * W[:, 4].copy()) 

    Jx = (1/dx) * Jx


    # Flattening the current matrix
    currents = af.flat(Jx)



    # temp = af.where(af.abs(currents) > 0)

    # if(temp.elements()>0):
	   #  index_list  = index_list[temp].copy()
	   #  currents    = currents[temp].copy()

    af.eval(index_list)
    af.eval(currents)


    return index_list, currents


def indices_and_currents_NGP(charge_electron, positions_x, velocity_x, dt, x_grid, dx, ghost_cells):

    positions_x_new     = positions_x + velocity_x * dt
    number_of_electrons = positions_x.elements()


    base_indices = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    Jx           = af.data.constant(0, positions_x.elements(), 3, dtype=af.Dtype.f64)


    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))

    temp = af.where(af.abs(positions_x-x_grid[x_zone])<af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices[temp] = x_zone[temp]

    temp = af.where(af.abs(positions_x - x_grid[x_zone])>=af.abs(positions_x-x_grid[x_zone + 1]))

    if(temp.elements()>0):
        base_indices[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)

    S0 = af.data.constant(0, 3 * positions_x.elements(), dtype=af.Dtype.f64)
    S1 = S0.copy()

    base_indices_plus  = (base_indices + 1 ).as_type(af.Dtype.u32)
    base_indices_minus = (base_indices - 1 ).as_type(af.Dtype.u32)

    index_list = af.join(0, base_indices_minus, base_indices, base_indices_plus)


    # Computing S(base_indices - 1, base_indices, base_indices + 1)

    positions_x_3x = af.join(0, positions_x, positions_x, positions_x)
    positions_x_new_3x = af.join(0, positions_x_new, positions_x_new, positions_x_new)

    temp = af.where(af.abs(positions_x_3x - x_grid[index_list]) < dx/2)

    if(temp.elements()>0):
        S0[temp] = 1

    temp = af.where(af.abs(positions_x_new_3x - x_grid[index_list]) < dx/2)

    if(temp.elements()>0):
        S1[temp] = 1


    # Computing DS
    DS = S1 - S0

    DS = af.data.moddims(DS, number_of_electrons, 3)

    # Computing weights

    W = DS.copy()

    Jx[:, 0] = -1 * charge_electron * velocity_x * W[:, 0].copy()

    Jx[:, 1] = Jx[:, 0] + (-1 * charge_electron * velocity_x * W[:, 1].copy())

    Jx[:, 2] = Jx[:, 1] + (-1 * charge_electron * velocity_x * W[:, 2].copy())


    # Flattening the current matrix
    currents = af.flat(Jx)

    af.eval(index_list)
    af.eval(currents)

    temp = af.where(af.abs(currents) > 0)

    index_list  = index_list[temp].copy()
    currents    = currents[temp].copy()

    af.eval(index_list)
    af.eval(currents)


    return index_list, currents



def Jx_Esirkepov( charge_electron, no_of_electrons, positions_x, velocity_x,\
                  x_grid, ghost_cells,length_domain_x, dx, dt\
                ):
    

    # storing the number of elements in x_center_grid for current deposition
    elements = x_grid.elements()

    index_list, currents = indices_and_currents_TSC( charge_electron, positions_x, dt, x_grid, dx, ghost_cells)




    # Depositing currents using numpy histogram
    input_indices      = (index_list)
    Jx_staggered, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=currents\
                                     )



    Jx_staggered = af.to_array(Jx_staggered)
    
    af.eval(Jx_staggered)

    return Jx_staggered




no_of_electrons = 1
charge_electron = 1

positions_x     = af.Array([0.45]).as_type(af.Dtype.f64)

velocity_x      = af.Array([-1.0]).as_type(af.Dtype.f64)

x_grid          = af.Array([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]).as_type(af.Dtype.f64)


dt              = 0.2

dx              = af.sum(x_grid[1] - x_grid[0])

x_right         = x_grid + dx/2

ghost_cells     = 1

length_domain_x = 1.0


Jx_staggered  = Jx_Esirkepov(  charge_electron, no_of_electrons, positions_x,\
                                  velocity_x, x_grid, ghost_cells,\
                                  length_domain_x, dx, dt\
                            )
print('Jx_staggered is ', Jx_staggered)

rho_plus = TSC_charge_deposition(charge_electron, no_of_electrons, af.Array([0.25]).as_type(af.Dtype.f64),\
                                x_grid, ghost_cells,length_domain_x, dx)

rho_minus = TSC_charge_deposition(charge_electron, no_of_electrons, af.Array([0.45]).as_type(af.Dtype.f64),\
                                x_grid, ghost_cells,length_domain_x, dx)
print('rho is ', dx * (rho_plus - rho_minus))