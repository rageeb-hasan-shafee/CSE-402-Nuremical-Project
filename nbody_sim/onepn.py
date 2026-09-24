"""Limited first-order post-Newtonian (1PN) correction for a system
dominated by one massive body (a star or black hole), following:

    T. Tatekawa, "Accelerating N-body simulation of self-gravitating
    systems with limited first-order post-Newtonian approximation",
    (2018), arXiv:1801.07986.

Key idea of that paper: computing the full 1PN three-body (EIH)
equations of motion for every triple of bodies costs O(N^3). If one
body (index `central_index`, e.g. the Sun or a black hole) is far more
massive than the rest, the 1PN correction is only significant for
interactions *with that body* -- so only those terms need to be kept,
which brings the cost back down to O(N^2). All other bodies still
interact through plain Newtonian gravity (see `simulation.py`).

This module implements that body-to-central-mass correction term (the
paper's Eq. 2.5, equivalent to the standard test-particle/EIH 1PN
formula used to derive Mercury's perihelion precession). The much
smaller three-body "Cross" terms (the paper's Eqs. 2.3/2.6) and the
correction to the central body's own motion (Eq. 2.2) are dropped --
both are negligible next to this term for a Sun-planet system, and
omitting them keeps the O(N^2) cost and the code simple.
"""
import numpy as np

# Speed of light in AU/day, matching this project's AU / day / Msun units.
_C_KM_S = 299792.458
_AU_KM = 1.495978707e8
_DAY_S = 86400.0
C_AU_PER_DAY = _C_KM_S * _DAY_S / _AU_KM  # ~173.1446 AU/day


def onepn_correction(positions, velocities, masses, G, central_index=0, c=C_AU_PER_DAY):
    """1PN acceleration correction [AU/day^2] for every body relative to
    the dominant mass at `central_index`; zero for the central body
    itself (its own correction is neglected, see module docstring).
    """
    n = len(masses)
    corr = np.zeros((n, 3))
    x1 = positions[central_index]
    GM = G * masses[central_index]

    for i in range(n):
        if i == central_index:
            continue
        r_vec = positions[i] - x1          # points from central body to body i
        r = np.linalg.norm(r_vec)
        v = velocities[i]
        v2 = v @ v
        rv = r_vec @ v
        corr[i] = (GM / (c ** 2 * r ** 3)) * ((4 * GM / r - v2) * r_vec + 4 * rv * v)

    return corr
