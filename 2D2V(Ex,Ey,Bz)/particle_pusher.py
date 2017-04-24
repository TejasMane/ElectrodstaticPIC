
def periodic_particles(positions_x, positions_y, length_domain_x, length_domain_y):
    '''
    function periodic_particles(positions_x, length_domain_x)
    -----------------------------------------------------------------------
    Input variables: positions_x and length_domain_x

        positions_x, positions_y: One dimensional arrays of size equal to number of particles
        taken in the PIC code. It contains the positions of particles in x and y directions
        respectively

        length_domain_x, length_domain_y: This is the length of the domain in x and y directions respectively.
        The domain is assumed to be from x = 0 to x = length_domain_x

    -----------------------------------------------------------------------
    returns: positions_x, positions_y
        This function returns the modified positions_x and positions_y such that particle previously
        gone outside the domain through the left boundary enter via the right boundary and vice versa.
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

        positions_y[outside_domain_top_y] = positions_y[outside_domain_top_y] - length_domain_y

    if outside_domain_left_x.elements() > 0:

        # Apply periodic boundary conditions

        positions_x[outside_domain_left_x]  = positions_x[outside_domain_left_x] + length_domain_x

    if outside_domain_bottom_y.elements() > 0:

        # Apply periodic boundary conditions

        positions_y[outside_domain_bottom_y]  = positions_y[outside_domain_bottom_y] + length_domain_y


    af.eval(positions_x, positions_y)

    return positions_x, positions_y


def fraction_finder(positions_x, positions_y, x_grid, y_grid, dx, dy):
    '''
    function fraction_finder(positions_x, positions_y, x_grid, y_grid, dx, dy)
    -----------------------------------------------------------------------
    Input variables: positions_x and length_domain_x

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in x direction.

        positions_y: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in y direction.

        x_grid, y_grid: This is an array denoting the position grid chosen in the PIC simulation in
        x and y directions respectively


        dx, dy: This is the distance between any two consecutive grid nodes of the position grid
        in x and y directions respectively

    -----------------------------------------------------------------------
    returns: x_frac, y_frac
        This function returns the fractions of grid cells needed to perform the 2D charge deposition

    '''
    x_frac = (positions_x - af.sum(x_grid[0])) / dx
    y_frac = (positions_y - af.sum(y_grid[0])) / dy

    af.eval(x_frac, y_frac)

    return x_frac, y_frac


"""
This is the integrator that implements the Boris Algorithm.
Add details here

Input positions  :  x(n)
Input Velocities :  v(n+0.5dt)
Input fields : E(x((n + 1)*dt)), B(x((n+1)*dt))

"""

def Boris( charge_particle, mass_particle, velocity_x,\
           velocity_y, dt, Ex_particle, Ey_particle, Bz_particle\
         ):

  # print('velocities are',af.sum(velocity_x)/(velocity_x.elements()) )
  vel_x_minus = velocity_x + (charge_particle * Ex_particle * dt) / (2 * mass_particle)
  vel_y_minus = velocity_y + (charge_particle * Ey_particle * dt) / (2 * mass_particle)

  t_magz    = (charge_particle * Bz_particle * dt) / (2 * mass_particle)

  vminus_cross_t_x =  (vel_y_minus * t_magz)
  vminus_cross_t_y = -(vel_x_minus * t_magz)

  vel_dashx = vel_x_minus + vminus_cross_t_x
  vel_dashy = vel_y_minus + vminus_cross_t_y

  t_mag = af.arith.sqrt(t_magz ** 2)

  s_z = (2 * t_magz) / (1 + af.arith.abs(t_mag ** 2))

  vel_x_plus = vel_x_minus + ((vel_dashy * s_z))
  vel_y_plus = vel_y_minus - ((vel_dashx * s_z))

  velocity_x_new  = vel_x_plus + (charge_particle * Ex_particle * dt) / (2 * mass_particle)
  velocity_y_new  = vel_y_plus + (charge_particle * Ey_particle * dt) / (2 * mass_particle)

  # Using v at (n+0.5) dt to push x at (n)dt

  af.eval(velocity_x_new, velocity_y_new)

  return (velocity_x_new , velocity_y_new)
