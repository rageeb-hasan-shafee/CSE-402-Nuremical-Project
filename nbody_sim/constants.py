"""Physical constants and units for the simulation.

Units adopted throughout this package (standard in solar-system dynamics):
    length : AU (astronomical unit)
    time   : day
    mass   : solar mass (Msun)

Using these units, the gravitational constant equals the square of the
Gaussian gravitational constant k, which avoids the extreme magnitudes of
SI units and matches classic astrodynamics practice.
"""

# Gaussian gravitational constant k [AU^(3/2) day^-1 Msun^-1/2]
GAUSSIAN_K = 0.01720209895

# G in AU^3 Msun^-1 day^-2
G = GAUSSIAN_K ** 2

# Conversion helpers
KG_PER_MSUN = 1.98847e30
AU_IN_KM = 1.495978707e8
