"""Build initial state vectors (position, velocity) for the Sun and the
major planets at epoch J2000.0 from mean Keplerian orbital elements.

Element table and rates are the low-precision formulae of Standish (1992),
"Keplerian Elements for Approximate Positions of the Major Planets", as
published by JPL Solar System Dynamics (valid 1800 AD - 2050 AD). Angles
are in degrees, distances in AU, time in Julian centuries from J2000.0.

Solving Kepler's equation for the eccentric anomaly is done with a
Newton-Raphson root find, which is itself one of the core numerical
methods this project is built around.
"""
import numpy as np

from constants import G, KG_PER_MSUN

# name: (a, adot, e, edot, i, idot, L, Ldot, long_peri, long_peri_dot, long_node, long_node_dot)
# a [AU, AU/cy], e [1, 1/cy], i, L, w_bar (long. of perihelion), Omega (long. of asc. node) [deg, deg/cy]
_ELEMENTS = {
    "Mercury": (0.38709927, 0.00000037, 0.20563593, 0.00001906, 7.00497902, -0.00594749,
                252.25032350, 149472.67411175, 77.45779628, 0.16047689, 48.33076593, -0.12534081),
    "Venus":   (0.72333566, 0.00000390, 0.00677672, -0.00004107, 3.39467605, -0.00078890,
                181.97909950, 58517.81538729, 131.60246718, 0.00268329, 76.67984255, -0.27769418),
    "Earth":   (1.00000261, 0.00000562, 0.01671123, -0.00004392, -0.00001531, -0.01294668,
                100.46457166, 35999.37244981, 102.93768193, 0.32327364, 0.0, 0.0),
    "Mars":    (1.52371034, 0.00001847, 0.09339410, 0.00007882, 1.84969142, -0.00813131,
                -4.55343205, 19140.30268499, -23.94362959, 0.44441088, 49.55953891, -0.29257343),
    "Jupiter": (5.20288700, -0.00011607, 0.04838624, -0.00013253, 1.30439695, -0.00183714,
                34.39644051, 3034.74612775, 14.72847983, 0.21252668, 100.47390909, 0.20469106),
    "Saturn":  (9.53667594, -0.00125060, 0.05386179, -0.00050991, 2.48599187, 0.00193609,
                49.95424423, 1222.49362201, 92.59887831, -0.41897216, 113.66242448, -0.28867794),
    "Uranus":  (19.18916464, -0.00196176, 0.04725744, -0.00004397, 0.77263783, -0.00242939,
                313.23810451, 428.48202785, 170.95427630, 0.40805281, 74.01692503, 0.04240589),
    "Neptune": (30.06992276, 0.00026291, 0.00859048, 0.00005105, 1.77004347, 0.00035372,
                -55.12002969, 218.45945325, 44.96476227, -0.32241464, 131.78422574, -0.00508664),
}

# Masses in kg (IAU / JPL reference values). Earth includes the Earth-Moon
# system mass; the Moon is added separately as its own body with a small
# fraction of that mass split off, see build_solar_system().
_MASSES_KG = {
    "Sun": 1.98847e30,
    "Mercury": 3.3011e23,
    "Venus": 4.8675e24,
    "Earth": 5.97237e24,
    "Mars": 6.4171e23,
    "Jupiter": 1.8982e27,
    "Saturn": 5.6834e26,
    "Uranus": 8.6810e25,
    "Neptune": 1.02413e26,
    "Moon": 7.342e22,
}

_COLORS = {
    "Sun": "#FDB813",
    "Mercury": "#8C8C94",
    "Venus": "#E8C27A",
    "Earth": "#3B82C4",
    "Moon": "#AAAAAA",
    "Mars": "#C1440E",
    "Jupiter": "#D9A066",
    "Saturn": "#E3C16F",
    "Uranus": "#9FE3E3",
    "Neptune": "#5B76E8",
}


def _wrap_deg(angle):
    return angle % 360.0


def _solve_kepler(M, e, tol=1e-12, max_iter=100):
    """Solve Kepler's equation E - e*sin(E) = M for E via Newton-Raphson."""
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    E = M if e < 0.8 else np.pi * np.sign(M)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return E


def _rotate_to_ecliptic(x_orb, y_orb, vx_orb, vy_orb, om_node, inc, arg_peri):
    """Rotate perifocal-plane coordinates into the ecliptic J2000 frame
    using R_z(Omega) * R_x(i) * R_z(omega)."""
    cO, sO = np.cos(om_node), np.sin(om_node)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(arg_peri), np.sin(arg_peri)

    # Rotation matrix combining argument of periapsis, inclination, node
    r11 = cO * cw - sO * sw * ci
    r12 = -cO * sw - sO * cw * ci
    r21 = sO * cw + cO * sw * ci
    r22 = -sO * sw + cO * cw * ci
    r31 = sw * si
    r32 = cw * si

    x = r11 * x_orb + r12 * y_orb
    y = r21 * x_orb + r22 * y_orb
    z = r31 * x_orb + r32 * y_orb
    vx = r11 * vx_orb + r12 * vy_orb
    vy = r21 * vx_orb + r22 * vy_orb
    vz = r31 * vx_orb + r32 * vy_orb
    return np.array([x, y, z]), np.array([vx, vy, vz])


def planet_state_vector(name, T_centuries=0.0, mu_sun=None):
    """Heliocentric ecliptic position [AU] and velocity [AU/day] of a major
    planet at T_centuries Julian centuries past J2000.0."""
    a0, ad, e0, ed, i0, idd, L0, Ld, wb0, wbd, om0, omd = _ELEMENTS[name]
    T = T_centuries

    a = a0 + ad * T
    e = e0 + ed * T
    i = np.radians(i0 + idd * T)
    L = L0 + Ld * T
    w_bar = wb0 + wbd * T
    om_node = om0 + omd * T

    M_deg = _wrap_deg(L - w_bar)
    M = np.radians(((M_deg + 180) % 360) - 180)
    arg_peri = np.radians(w_bar - om_node)
    om_node = np.radians(om_node)

    E = _solve_kepler(M, e)

    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E)

    if mu_sun is None:
        mu_sun = G  # GM with M in solar masses (planet mass negligible)
    n = np.sqrt(mu_sun / a ** 3)  # mean motion, rad/day
    E_dot = n / (1 - e * np.cos(E))
    vx_orb = -a * np.sin(E) * E_dot
    vy_orb = a * np.sqrt(1 - e ** 2) * np.cos(E) * E_dot

    pos, vel = _rotate_to_ecliptic(x_orb, y_orb, vx_orb, vy_orb, om_node, i, arg_peri)
    return pos, vel


def moon_state_vector_geocentric():
    """Approximate geocentric position/velocity of the Moon at J2000.0.

    Uses fixed mean orbital elements (no secular node/perigee precession),
    which is adequate for short (few-year) demonstration runs but is not a
    precision lunar ephemeris.
    """
    a = 384400.0 / 1.495978707e8  # km -> AU
    e = 0.0549
    i = np.radians(5.145)
    om_node = np.radians(125.08)
    arg_peri = np.radians(318.15)
    M = np.radians(135.27)

    mu_earth = G * (_MASSES_KG["Earth"] / KG_PER_MSUN)
    E = _solve_kepler(M, e)
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E)
    n = np.sqrt(mu_earth / a ** 3)
    E_dot = n / (1 - e * np.cos(E))
    vx_orb = -a * np.sin(E) * E_dot
    vy_orb = a * np.sqrt(1 - e ** 2) * np.cos(E) * E_dot

    return _rotate_to_ecliptic(x_orb, y_orb, vx_orb, vy_orb, om_node, i, arg_peri)


PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]


def build_solar_system(include_moon=True, bodies=None):
    """Return (names, masses[Msun], positions[AU] (N,3), velocities[AU/day] (N,3), colors).

    Barycentric-ish setup: the Sun starts at the origin at rest, then all
    bodies are shifted so the system's centre of mass is stationary at the
    origin (removes overall drift, matching a real N-body integration).
    """
    if bodies is None:
        bodies = PLANETS

    names = ["Sun"] + list(bodies)
    masses_kg = [_MASSES_KG["Sun"]] + [_MASSES_KG[b] for b in bodies]
    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]

    for name in bodies:
        pos, vel = planet_state_vector(name)
        positions.append(pos)
        velocities.append(vel)

    if include_moon and "Earth" in bodies:
        moon_pos_geo, moon_vel_geo = moon_state_vector_geocentric()
        earth_idx = names.index("Earth")
        names.append("Moon")
        masses_kg.append(_MASSES_KG["Moon"])
        positions.append(positions[earth_idx] + moon_pos_geo)
        velocities.append(velocities[earth_idx] + moon_vel_geo)

    masses = np.array(masses_kg) / KG_PER_MSUN
    positions = np.array(positions)
    velocities = np.array(velocities)

    # Shift to the barycentric frame (zero total momentum) for a stable,
    # non-drifting simulation.
    total_mass = masses.sum()
    com_pos = (masses[:, None] * positions).sum(axis=0) / total_mass
    com_vel = (masses[:, None] * velocities).sum(axis=0) / total_mass
    positions = positions - com_pos
    velocities = velocities - com_vel

    colors = [_COLORS[n] for n in names]
    return names, masses, positions, velocities, colors
