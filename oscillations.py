"""
oscillations.py — Neutrino oscillation probabilities in matter (NuFast algorithm).

All nine P(ν_α → ν_β) are computed analytically, including Earth-matter effects,
following the NuFast method of Denton & Parke (arXiv:2405.02400;
Phys. Rev. D 110, 073005).
Precision is controlled by N_Newton (Newton–Raphson iterations on the eigenvalue).

Conventions
-----------
* Δm²₃₁ > 0 → Normal Ordering (NO); Δm²₃₁ < 0 → Inverted Ordering (IO).
* E > 0 → neutrinos; E < 0 → anti-neutrinos.
* Probability matrix index order: [α][β] = P(ν_α → ν_β),
  with 0=e, 1=μ, 2=τ.  E.g. probs[1][0] = P(ν_μ → ν_e).
"""

import numpy as np

# Unit-conversion constants
_eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
_YerhoE2a = 1.52588e-4


def Probability_Matter_LBL(
    s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton
):
    """
    Compute all nine oscillation probabilities including matter effects.

    Parameters
    ----------
    s12sq, s13sq, s23sq : float
        sin²(θ₁₂), sin²(θ₁₃), sin²(θ₂₃).
    delta : float
        CP-violating phase δ_CP (radians).
    Dmsq21 : float
        Δm²₂₁ = m₂² − m₁²  (eV²).
    Dmsq31 : float
        Δm²₃₁ = m₃² − m₁²  (eV²).  Positive for NO, negative for IO.
    L : float
        Baseline (km).
    E : float
        Neutrino energy (GeV).  Positive for neutrinos, negative for anti-neutrinos.
    rho : float
        Matter density along the baseline (g/cm³).
    Ye : float
        Electron fraction (≈ 0.5 for Earth's crust/mantle).
    N_Newton : int
        Number of Newton–Raphson iterations (0 is usually sufficient; use 1 for
        very long baselines or high statistics).

    Returns
    -------
    probs : ndarray, shape (3, 3)
        probs[α][β] = P(ν_α → ν_β), with 0=e, 1=μ, 2=τ.
    """
    # ------------------------------------------------------------------ #
    # Simple functions of the mixing parameters
    # ------------------------------------------------------------------ #
    c13sq = 1 - s13sq

    Ue2sq = c13sq * s12sq
    Ue3sq = s13sq
    Um3sq = c13sq * s23sq
    Ut2sq = s13sq * s12sq * s23sq
    Um2sq = (1 - s12sq) * (1 - s23sq)

    Jrr   = np.sqrt(Um2sq * Ut2sq)
    sind  = np.sin(delta)
    cosd  = np.cos(delta)
    Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
    Jmatter = 8 * Jrr * c13sq * sind
    Amatter = Ye * rho * E * _YerhoE2a
    Dmsqee  = Dmsq31 - s12sq * Dmsq21

    # A, B, C, See, Tee (partial Tmm)
    A   = Dmsq21 + Dmsq31
    See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq
    Tmm = Dmsq21 * Dmsq31
    Tee = Tmm * (1 - Ue3sq - Ue2sq)
    C   = Amatter * Tee
    A   = A + Amatter

    # ------------------------------------------------------------------ #
    # Leading-order λ₃ (eigenvalue) via the MP/DMP approximation
    # ------------------------------------------------------------------ #
    xmat    = Amatter / Dmsqee
    tmp     = 1 - xmat
    lambda3 = Dmsq31 + 0.5 * Dmsqee * (xmat - 1 + np.sqrt(tmp * tmp + 4 * s13sq * xmat))

    # Optional Newton–Raphson refinement
    B = Tmm + Amatter * See
    for _ in range(N_Newton):
        lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (
            lambda3 * (2 * (lambda3 - A) + lambda3) + B
        )

    # ------------------------------------------------------------------ #
    # Δλ splittings
    # ------------------------------------------------------------------ #
    tmp        = A - lambda3
    Dlambda21  = np.sqrt(tmp * tmp - 4 * C / lambda3)
    lambda2    = 0.5 * (A - lambda3 + Dlambda21)
    Dlambda32  = lambda3 - lambda2
    Dlambda31  = Dlambda32 + Dlambda21

    # ------------------------------------------------------------------ #
    # Effective mixing elements via the Rosetta identity
    # ------------------------------------------------------------------ #
    PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21)
    Xp3 = PiDlambdaInv * Dlambda21
    Xp2 = -PiDlambdaInv * Dlambda31

    Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
    Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

    Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq
    Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A)

    Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
    Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

    # Effective Jarlskog invariant (NHS formula)
    Jmatter = Jmatter * Dmsq21 * Dmsq31 * (Dmsq31 - Dmsq21) * PiDlambdaInv

    Ue1sq = 1 - Ue3sq - Ue2sq
    Um1sq = 1 - Um3sq - Um2sq
    Ut3sq = 1 - Um3sq - Ue3sq
    Ut2sq = 1 - Um2sq - Ue2sq
    Ut1sq = 1 - Um1sq - Ue1sq

    # ------------------------------------------------------------------ #
    # Kinematic phase factors
    # ------------------------------------------------------------------ #
    Lover4E = _eVsqkm_to_GeV_over4 * L / E
    D21     = Dlambda21 * Lover4E
    D32     = Dlambda32 * Lover4E

    sinD21 = np.sin(D21)
    sinD31 = np.sin(D32 + D21)
    sinD32 = np.sin(D32)

    triple_sin   = sinD21 * sinD31 * sinD32
    sinsqD21_2   = 2 * sinD21 * sinD21
    sinsqD31_2   = 2 * sinD31 * sinD31
    sinsqD32_2   = 2 * sinD32 * sinD32

    # ------------------------------------------------------------------ #
    # Three independent probabilities (CP-conserving + CP-violating parts)
    # ------------------------------------------------------------------ #
    Pme_CPC = (
        (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2
        + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2
        + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
    )
    Pme_CPV = -Jmatter * triple_sin

    Pmm = 1 - 2 * (
        Um2sq * Um1sq * sinsqD21_2
        + Um3sq * Um1sq * sinsqD31_2
        + Um3sq * Um2sq * sinsqD32_2
    )
    Pee = 1 - 2 * (
        Ue2sq * Ue1sq * sinsqD21_2
        + Ue3sq * Ue1sq * sinsqD31_2
        + Ue3sq * Ue2sq * sinsqD32_2
    )

    # ------------------------------------------------------------------ #
    # Assemble the full 3×3 probability matrix
    # ------------------------------------------------------------------ #
    probs = np.empty((3, 3))
    probs[0][0] = Pee
    probs[0][1] = Pme_CPC - Pme_CPV
    probs[0][2] = 1 - Pee - probs[0][1]
    probs[1][0] = Pme_CPC + Pme_CPV
    probs[1][1] = Pmm
    probs[1][2] = 1 - probs[1][0] - Pmm
    probs[2][0] = 1 - Pee - probs[1][0]
    probs[2][1] = 1 - probs[0][1] - Pmm
    probs[2][2] = 1 - probs[0][2] - probs[1][2]
    return probs


def Probability_Vacuum_LBL(s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31, L, E):
    """
    Compute all nine oscillation probabilities in vacuum.

    Parameters are the same as Probability_Matter_LBL, except there is no
    rho, Ye, or N_Newton argument.
    """
    c13sq = 1 - s13sq
    Ue3sq = s13sq
    Ue2sq = c13sq * s12sq
    Um3sq = c13sq * s23sq
    Ut2sq = s13sq * s12sq * s23sq
    Um2sq = (1 - s12sq) * (1 - s23sq)

    Jrr   = np.sqrt(Um2sq * Ut2sq)
    sind  = np.sin(delta)
    cosd  = np.cos(delta)
    Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
    Jvac  = 8 * Jrr * c13sq * sind

    Ue1sq = 1 - Ue3sq - Ue2sq
    Um1sq = 1 - Um3sq - Um2sq
    Ut3sq = 1 - Um3sq - Ue3sq
    Ut2sq = 1 - Um2sq - Ue2sq
    Ut1sq = 1 - Um1sq - Ue1sq

    Lover4E = _eVsqkm_to_GeV_over4 * L / E
    D21 = Dmsq21 * Lover4E
    D31 = Dmsq31 * Lover4E

    sinD21 = np.sin(D21)
    sinD31 = np.sin(D31)
    sinD32 = np.sin(D31 - D21)

    triple_sin = sinD21 * sinD31 * sinD32
    sinsqD21_2 = 2 * sinD21 * sinD21
    sinsqD31_2 = 2 * sinD31 * sinD31
    sinsqD32_2 = 2 * sinD32 * sinD32

    Pme_CPC = (
        (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2
        + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2
        + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
    )
    Pme_CPV = -Jvac * triple_sin

    Pmm = 1 - 2 * (
        Um2sq * Um1sq * sinsqD21_2
        + Um3sq * Um1sq * sinsqD31_2
        + Um3sq * Um2sq * sinsqD32_2
    )
    Pee = 1 - 2 * (
        Ue2sq * Ue1sq * sinsqD21_2
        + Ue3sq * Ue1sq * sinsqD31_2
        + Ue3sq * Ue2sq * sinsqD32_2
    )

    probs = np.empty((3, 3))
    probs[0][0] = Pee
    probs[0][1] = Pme_CPC - Pme_CPV
    probs[0][2] = 1 - Pee - probs[0][1]
    probs[1][0] = Pme_CPC + Pme_CPV
    probs[1][1] = Pmm
    probs[1][2] = 1 - probs[1][0] - Pmm
    probs[2][0] = 1 - Pee - probs[1][0]
    probs[2][1] = 1 - probs[0][1] - Pmm
    probs[2][2] = 1 - probs[0][2] - probs[1][2]
    return probs


if __name__ == "__main__":
    # Quick sanity-check: NOvA-like setup
    L       = 810      # km
    E       = -2.0     # GeV (negative → anti-neutrino)
    rho     = 2.84     # g/cm³
    Ye      = 0.5
    N_Newton = 0

    s12sq  = 0.31
    s13sq  = 0.02172
    s23sq  = 0.50
    delta  = np.pi
    Dmsq21 = 7.5e-5    # eV²
    Dmsq31 = 2.5e-3    # eV²  (NO)

    probs = Probability_Matter_LBL(
        s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31, L, E, rho, Ye, N_Newton
    )
    print(f"L={L} km, E={E} GeV, rho={rho} g/cm³")
    print(f"P(ν̄_μ → ν̄_e) = {probs[1][0]:.4g}")
