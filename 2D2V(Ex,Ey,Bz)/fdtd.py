import arrayfire as af
import numpy as np

from scipy.fftpack import fftfreq
from numpy.fft import fft2, ifft2

def periodic_field(field, ghost_cells):
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



""" Equations for FDTD"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B  = dBz/dz

"""
Notes for periodic boundary conditions:
for [0, Lx] domain use periodic BC's such that last point in the physical domain coincides with the first point
for [0, Lx) domain use periodic BC's such that the ghost point after the last physical point coincides with the first
physical point
"""


""" Alignment of the spatial grids for the fields(Convention chosen)

# This is the convention which will be used in the matrix representation

positive y axis -------------> going down
positive x axis -------------> going right

Let the domain be [0,1]
Sample grid with one ghost cell at each end and the physical domain containing only 2 points
Here dx = 1, dx/2 = 0.5

Let the grids for the example case be denoted be:

x_center = [-1, 0, 1, 2]
y_center = [-1, 0, 1, 2]

x_center[0] and x_center[3] are the ghost points and x_center[1] and x_center[2] are the physical points
y_center[0] and y_center[3] are the ghost points and y_center[1] and y_center[2] are the physical points


x_right  = [-0.5, 0.5, 1.5, 2.5]
y_top    = [-0.5, 0.5, 1.5, 2.5]

x_right[0] and x_right[3] are the ghost points and x_right[1] and x_right[2] are the physical points
y_top[0] and y_top[3] are the ghost points and y_top[1] and y_top[2] are the physical points

This can be seen visually with the below presented schematic

where pij are the points located on the fused spatial grids for whole numbers i an j

p11, p12, p13, p14, p15, p16, p17, p18, p28, p38, p48, p58, p68, p78, p88, p87, p86, p85, p84, p83, p82,
p81, p71, p61, p51, p41, p31 and p21 are all ghost points while all other points are the physical points for this
example taken.

+++++++++p11--------p12--------p13--------p14--------p15--------p16--------p17--------p18+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p11 = (x_center[0], y_center[0]), p13 = (x_center[1], y_center[0])       |
          |   p15 = (x_center[2], y_center[0]),p17 = (x_center[3], y_center[0])        |
          |   p12 = (x_right[0], y_center[0]), p14 = (x_right[1], y_center[0])         |
          |   p16 = (x_right[2], y_center[0]), p18 = (x_right[3], y_center[0])         |
          |                                                                            |
+++++++++p21--------p22--------p23--------p24--------p25--------p26--------p27--------p28+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p21 = (x_center[0], y_top[0]), p23 = (x_center[1], y_top[0])             |
          |   p25 = (x_center[2], y_top[0]), p27 = (x_center[3], y_top[0])             |
          |   p22 = (x_right[0], y_top[0]), p24 = (x_right[1], y_top[0])               |
          |   p26 = (x_right[2], y_top[0]), p28 = (x_right[3], y_top[0])               |
          |                                                                            |
+++++++++p31--------p32--------p33--------p34--------p35--------p36--------p37--------p38+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p31 = (x_center[0], y_center[1]), p33 = (x_center[1], y_center[1])       |
          |   p35 = (x_center[2], y_center[1]), p37 = (x_center[3], y_center[1])       |
          |   p32 = (x_right[0], y_center[1]), p34 = (x_right[1], y_center[1])         |
          |   p36 = (x_right[2], y_center[1]), p38 = (x_right[3], y_center[1])         |
          |                                                                            |
+++++++++p41--------p42--------p43--------p44--------p45--------p46--------p47--------p48+++++++++++++++++++++++++++++++
          |                                                                            |
          |   p41 = (x_center[0], y_top[1]), p43 = (x_center[1], y_top[1])             |
          |   p45 = (x_center[2], y_top[1]), p47 = (x_center[3], y_top[1])             |
          |   p42 = (x_right[0], y_top[1]), p44 = (x_right[1], y_top[1])               |
          |   p46 = (x_right[2], y_top[1]), p48 = (x_right[3], y_top[1])               |
          |                                                                            |
+++++++++p51--------p52--------p53--------p54--------p55--------p56--------p57--------p58+++++++++++++++++++++++++++++++
          |                                                                            |
          |                                                                            |
          |                                                                            |
          | And So on ................                                                 |
          |                                                                            |
          |                                                                            |
+++++++++p61--------p62--------p63--------p64--------p65--------p66--------p67--------p68+++++++++++++++++++++++++++++++
          |                                                                            |
          |                                                                            |
          | And So on ................                                                 |
          |                                                                            |
          |                                                                            |
          |                                                                            |
+++++++++p71--------p72--------p73--------p74--------p75--------p76--------p77--------p78+++++++++++++++++++++++++++++++
          |                                                                            |
          |                                                                            |
          |                                                                            |
          | And So on ................                                                 |
          |                                                                            |
          |                                                                            |
+++++++++p81--------p82--------p83--------p84--------p85--------p86--------p87--------p88+++++++++++++++++++++++++++++++

Now the fields aligned in x and y direction along with the following grids:

Ex  = (x_right, y_center  ) 0, dt, 2dt, 3dt...
Ey  = (x_center, y_top    ) 0, dt, 2dt, 3dt...
Bz  = (x_right, y_top     ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...

rho = (x_center, y_center )

Jx  = (x_right, y_center  ) 0.5dt, 1.5dt, 2.5dt...
Jy  = (x_center, y_top    ) 0.5dt, 1.5dt, 2.5dt...

"""

""" Equations for fdtd (variation along x and y)"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B = dBz/dz


def fdtd( Bz, Ex, Ey, Lx, Ly, ghost_cells, Jx, Jy, dt):


    forward_row     = af.Array([1, -1, 0])
    forward_column  = af.Array([1, -1, 0])
    backward_row    = af.Array([0, 1, -1])
    backward_column = af.Array([0, 1, -1])
    identity        = af.Array([0, 1, 0] )

    """ Number of grid points in the field's domain """

    (x_number_of_points,  y_number_of_points) = Bz.dims()

    """ number of grid zones calculated from the input fields """

    Nx = x_number_of_points - 2*ghost_cells-1
    Ny = y_number_of_points - 2*ghost_cells-1

    """ local variables for storing the input fields """

    Bz_local = Bz.copy()
    Ex_local = Ex.copy()
    Ey_local = Ey.copy()

    """Enforcing periodic BC's"""

    Bz_local = periodic_field(Bz_local, ghost_cells)

    Ex_local = periodic_field(Ex_local, ghost_cells)

    Ey_local = periodic_field(Ey_local, ghost_cells)

    """ Setting division size and time steps"""

    dx = np.float(Lx / (Nx))
    dy = np.float(Ly / (Ny))

    """ defining variable for convenience """

    dt_by_dx = dt / (dx)
    dt_by_dy = dt / (dy)


    """  Updating the Electric fields using the current too   """

    Ex_local += dt_by_dy * (af.signal.convolve2_separable(backward_row, identity, Bz_local)) - (Jx) * dt

    # dEx/dt = + dBz/dy

    Ey_local += -dt_by_dx * (af.signal.convolve2_separable(identity, backward_column, Bz_local)) - (Jy) * dt

    # dEy/dt = - dBz/dx

    """  Implementing periodic boundary conditions using ghost cells  """

    Ex_local = periodic_field(Ex_local, ghost_cells)

    Ey_local = periodic_field(Ey_local, ghost_cells)

    """  Updating the Magnetic field  """

    Bz_local += - dt_by_dx * (af.signal.convolve2_separable(identity, forward_column, Ey_local))  + dt_by_dy * (af.signal.convolve2_separable(forward_row, identity, Ex_local))

    # dBz/dt = - ( dEy/dx - dEx/dy )

    #Implementing periodic boundary conditions using ghost cells

    Bz_local = periodic_field(Bz_local, ghost_cells)

    af.eval(Bz_local, Ex_local, Ey_local)

    return Bz_local, Ex_local, Ey_local




# \begin{align}
# \hat{V}(k) &= \int_{0}^{1} potential(x)e^{-2\pi\;i\;k\;x}dx \\ \\
# potential(x) &= \frac{1}{Npoints}\int_{0}^{1} \hat{potential}(k)e^{+2\pi\;i\;k\;x}dk \\ \\
# \hat{potential}(k) &= \frac{1}{4\pi^{2}\;k^2}\hat{\rho(k)} \\ \\
# \hat{E}(k) &= -i(2\pi\;k)\hat{potential}(k)
# \end{align}

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
    rho_temp = rho[0: -1, 0: -1]

    k_x = af.to_array(fftfreq(rho_temp.shape[1], dx))
    k_x = af.Array.as_type(k_x, af.Dtype.c64)
    k_y = af.to_array(fftfreq(rho_temp.shape[0], dy))
    k_x = af.tile(af.reorder(k_x), rho_temp.shape[0], 1)
    k_y = af.tile(k_y, 1, rho_temp.shape[1])
    k_y = af.Array.as_type(k_y, af.Dtype.c64)

    rho_hat       = fft2(rho_temp)
    rho_hat = af.to_array(rho_hat)
    potential_hat = af.constant(0, rho_temp.shape[0], rho_temp.shape[1], dtype=af.Dtype.c64)

    potential_hat       = (1/(4 * np.pi**2 * (k_x * k_x + k_y * k_y))) * rho_hat
    potential_hat[0, 0] = 0

    potential_hat = np.array(potential_hat)

    E_x_hat = -1j * 2 * np.pi * np.array(k_x) * potential_hat
    E_y_hat = -1j * 2 * np.pi * np.array(k_y) * potential_hat

    E_x = (ifft2(E_x_hat)).real
    E_y = (ifft2(E_y_hat)).real

    E_x = af.to_array(E_x)
    E_y = af.to_array(E_y)

    # Applying periodic boundary conditions

    E_x = af.join(0, E_x, E_x[0, :])
    E_x = af.join(1, E_x, E_x[:, 0])
    E_y = af.join(0, E_y, E_y[0, :])
    E_y = af.join(1, E_y, E_y[:, 0])

    E_x[-1, -1] = E_x[0, 0].copy()
    E_y[-1, -1] = E_y[0, 0].copy()

    af.eval(E_x, E_y)
    return(E_x, E_y)

