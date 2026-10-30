import unittest

import numpy as np
from mpi4py import MPI
from pyNastran.bdf.bdf import BDF

from tacs import TACS, constitutive, elements, functions, pytacs

"""
Analytic verification of temperature-dependent thermoelasticity in pyTACS.

1. Clamped plate (2D, CQUAD4) at uniform temperature change theta: the exact
   solution has u = 0, uniform mechanical strain -alpha(T)*theta, and the
   compliance (integral of e . s over the in-plane area) is

       C = A * 2 * t * E * alpha^2 * theta^2 / (1 - nu)

   evaluated with E, nu, alpha at the temperature change theta.
2. Clamped cube (3D, CHEXA) at uniform temperature change theta:

       C = V * 3 * E * alpha^2 * theta^2 / (1 - 2 nu)

3. 1D steady nonlinear conduction (kappa(theta) linear in theta) in a strip
   with mechanically fixed nodes. The exact solution follows from the Kirchhoff
   transform and is compared nodally and through the AverageTemperature
   function.

Constitutive tables are functions of the temperature change theta = T - Tref,
which is the element temperature DOF. Tables are given directly in theta.
"""

COMM = MPI.COMM_WORLD

TACS_DTYPE = TACS.dtype

STEADY = elements.STEADY_STATE_MECHANICAL | elements.STEADY_STATE_THERMAL

# Tables in the temperature change theta = T - Tref [K]
TABLE_THETA = np.array([-50.0, 50.0, 200.0, 350.0])
TABLES = {
    "E": np.array([72e9, 70e9, 64e9, 55e9]),
    "nu": np.array([0.30, 0.31, 0.33, 0.35]),
    "alpha": np.array([22e-6, 24e-6, 27e-6, 31e-6]),
    "kappa": np.array([240.0, 230.0, 215.0, 195.0]),
    "specific_heat": np.array([850.0, 900.0, 975.0, 1050.0]),
}

# Constructor values (used by the constant-material controls)
RHO = 2700.0
E0 = 70e9
NU0 = 0.3
YS = 270e6
ALPHA0 = 24e-6
KAPPA0 = 230.0
CP0 = 900.0

# Solver tolerances
LIN_TOL = 1e-14
NL_ABS_TOL = 1e-9
NL_REL_TOL = 1e-12

# Relative tolerance on the clamped-plate/cube compliance checks
CLAMPED_RTOL = 1e-8


def make_table(values):
    """Return the (2, n) table [theta; values] for the constructor"""
    return np.array([TABLE_THETA, values], dtype=TACS_DTYPE)


def make_material(tables=True):
    """Isotropic material; optionally with all (2, n) theta tables."""
    kwargs = dict(
        rho=RHO,
        E=E0,
        nu=NU0,
        ys=YS,
        alpha=ALPHA0,
        kappa=KAPPA0,
        specific_heat=CP0,
    )
    if tables:
        for name, values in TABLES.items():
            kwargs[name] = make_table(values)
    return constitutive.MaterialProperties(**kwargs)


def table_props_at_theta(theta):
    """Expected E, nu, alpha at the temperature change theta from the tables."""
    return {
        name: float(np.interp(theta, TABLE_THETA, TABLES[name]))
        for name in ["E", "nu", "alpha"]
    }


def solve_problem(bdf, mat, kind, t=1.0, avg_volume=None):
    """
    Build the pyTACS assembler from a pyNastran BDF object, solve the single
    steady static problem and return (problem, funcs, assembler).

    kind is '2d' (CQUAD4 plane stress) or '3d' (CHEXA solid). If avg_volume is
    given, the AverageTemperature function is added with that normalization.
    """
    fea = pytacs.pyTACS(bdf, COMM)

    def elem_call_back(
        dv_num, comp_id, comp_descript, elem_descripts, special_dvs, **kwargs
    ):
        if kind == "2d":
            con = constitutive.PlaneStressConstitutive(mat, t=t, tNum=-1)
            model = elements.LinearThermoelasticity2D(con, STEADY)
            basis = elements.LinearQuadBasis()
            elem = elements.Element2D(model, basis)
        else:
            con = constitutive.SolidConstitutive(mat, t=1.0, tNum=-1)
            model = elements.LinearThermoelasticity3D(con, STEADY)
            basis = elements.LinearHexaBasis()
            elem = elements.Element3D(model, basis)
        return [elem]

    fea.initialize(elem_call_back)

    prob = fea.createStaticProblem(name="thermo")
    prob.addFunction("compliance", functions.Compliance)
    if avg_volume is not None:
        prob.addFunction(
            "avg_temp", functions.AverageTemperature, volume=avg_volume
        )

    prob.setOption("writeSolution", False)
    prob.setOption("L2Convergence", LIN_TOL)
    prob.setOption("L2ConvergenceRel", LIN_TOL)
    if prob.nonlinearSolver is not None:
        prob.nonlinearSolver.setConvergenceTolerance(
            absTol=NL_ABS_TOL, relTol=NL_REL_TOL
        )

    prob.solve()
    funcs = {}
    prob.evalFunctions(funcs)
    return prob, funcs, fea


def quad_plate_bdf(nx, ny, Lx, Ly, t):
    """Rectangular CQUAD4 mesh in the z=0 plane, Nastran node numbering."""
    bdf = BDF()
    bdf.add_mat1(1, E0, None, NU0, RHO)
    bdf.add_pshell(1, mid1=1, t=t)

    def nid(i, j):
        return i * (ny + 1) + j + 1

    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    for i in range(nx + 1):
        for j in range(ny + 1):
            bdf.add_grid(nid(i, j), [xs[i], ys[j], 0.0])

    eid = 1
    for i in range(nx):
        for j in range(ny):
            bdf.add_cquad4(
                eid,
                1,
                [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)],
            )
            eid += 1

    node_xy = {nid(i, j): (xs[i], ys[j]) for i in range(nx + 1) for j in range(ny + 1)}
    return bdf, nid, node_xy


def hex_cube_bdf(n, L):
    """n x n x n CHEXA mesh of the cube [0, L]^3, Nastran node numbering."""
    bdf = BDF()
    bdf.add_mat1(1, E0, None, NU0, RHO)
    bdf.add_psolid(1, 1)

    def nid(i, j, k):
        return i * (n + 1) ** 2 + j * (n + 1) + k + 1

    xs = np.linspace(0.0, L, n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                bdf.add_grid(nid(i, j, k), [xs[i], xs[j], xs[k]])

    eid = 1
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Nastran CHEXA ordering: bottom face CCW, then top face CCW
                conn = [
                    nid(i, j, k),
                    nid(i + 1, j, k),
                    nid(i + 1, j + 1, k),
                    nid(i, j + 1, k),
                    nid(i, j, k + 1),
                    nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i, j + 1, k + 1),
                ]
                bdf.add_chexa(eid, 1, conn)
                eid += 1

    boundary = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                if i in (0, n) or j in (0, n) or k in (0, n):
                    boundary.append(nid(i, j, k))
    return bdf, nid, xs, boundary


def add_spc_set(bdf, set_id, nodes, components, enforced):
    """
    Add one single-node SPC card per node.

    A multi-node SPC card is not used on purpose: copy.deepcopy(bdf) inside
    pyTACS (pymeshloader.scanBdfFile) truncates a multi-node SPC card to its
    first two nodes with the installed pyNastran, so the remaining nodes would
    silently lose their boundary conditions.
    """
    for nid_ in nodes:
        bdf.add_spc(set_id, [nid_], [components], [enforced])


def expected_clamped_compliance_2d(mat_values, theta, t, area):
    E, nu, alpha = mat_values["E"], mat_values["nu"], mat_values["alpha"]
    return area * 2.0 * t * E * alpha**2 * theta**2 / (1.0 - nu)


def expected_clamped_compliance_3d(mat_values, theta, volume):
    E, nu, alpha = mat_values["E"], mat_values["nu"], mat_values["alpha"]
    return volume * 3.0 * E * alpha**2 * theta**2 / (1.0 - 2.0 * nu)


class TestClampedUniformTemperature2D(unittest.TestCase):
    """Plate clamped on all boundary nodes, uniform prescribed theta on all nodes."""

    L = 1.0
    t = 0.01
    n = 4
    theta = 150.0

    def run_case(self, mat):
        bdf, nid, node_xy = quad_plate_bdf(self.n, self.n, self.L, self.L, self.t)
        all_nodes = list(node_xy.keys())
        boundary = [
            nid(i, j)
            for i in range(self.n + 1)
            for j in range(self.n + 1)
            if i in (0, self.n) or j in (0, self.n)
        ]
        # Mechanical clamp on the boundary: u = v = 0
        add_spc_set(bdf, 2, boundary, "12", 0.0)
        # Uniform temperature change on every node
        add_spc_set(bdf, 1, all_nodes, "3", self.theta)

        prob, funcs, fea = solve_problem(bdf, mat, "2d", t=self.t)

        U = np.real(prob.getVariables()).reshape(-1, 3)
        return funcs["thermo_compliance"], U

    def test_clamped_uniform_temperature_2d(self):
        mat = make_material(tables=True)
        vals = table_props_at_theta(self.theta)
        expected = expected_clamped_compliance_2d(
            vals, self.theta, self.t, self.L * self.L
        )
        computed, U = self.run_case(mat)

        # Exact solution is u = v = 0 (uniform thermal strain, clamped boundary)
        self.assertLess(np.max(np.abs(np.real(U[:, :2]))), 1e-12)
        rel = abs(np.real(computed) - expected) / abs(expected)
        print(
            f"\n[2D tables] E={vals['E']:.6e} nu={vals['nu']:.6f} "
            f"alpha={vals['alpha']:.6e}  expected={expected:.10e} "
            f"computed={np.real(computed):.10e} rel={rel:.3e}"
        )
        self.assertLess(rel, CLAMPED_RTOL)

    def test_clamped_uniform_temperature_2d_constant_control(self):
        mat = make_material(tables=False)
        vals = {"E": E0, "nu": NU0, "alpha": ALPHA0}
        expected = expected_clamped_compliance_2d(
            vals, self.theta, self.t, self.L * self.L
        )
        computed, U = self.run_case(mat)

        self.assertLess(np.max(np.abs(np.real(U[:, :2]))), 1e-12)
        rel = abs(np.real(computed) - expected) / abs(expected)
        print(
            f"\n[2D constant] expected={expected:.10e} "
            f"computed={np.real(computed):.10e} rel={rel:.3e}"
        )
        self.assertLess(rel, CLAMPED_RTOL)


class TestClampedUniformTemperature3D(unittest.TestCase):
    """Cube clamped on all boundary nodes, uniform prescribed theta on all nodes."""

    L = 1.0
    n = 3
    theta = 150.0

    def run_case(self, mat):
        bdf, nid, xs, boundary_ids = hex_cube_bdf(self.n, self.L)
        all_nodes = [
            nid(i, j, k)
            for i in range(self.n + 1)
            for j in range(self.n + 1)
            for k in range(self.n + 1)
        ]
        # Mechanical clamp on the boundary: u = v = w = 0
        add_spc_set(bdf, 2, boundary_ids, "123", 0.0)
        # Uniform temperature change on every node
        add_spc_set(bdf, 1, all_nodes, "4", self.theta)

        prob, funcs, fea = solve_problem(bdf, mat, "3d")

        U = np.real(prob.getVariables()).reshape(-1, 4)
        return funcs["thermo_compliance"], U

    def test_clamped_uniform_temperature_3d(self):
        mat = make_material(tables=True)
        vals = table_props_at_theta(self.theta)
        expected = expected_clamped_compliance_3d(vals, self.theta, self.L**3)
        computed, U = self.run_case(mat)

        self.assertLess(np.max(np.abs(np.real(U[:, :3]))), 1e-12)
        rel = abs(np.real(computed) - expected) / abs(expected)
        print(
            f"\n[3D tables] E={vals['E']:.6e} nu={vals['nu']:.6f} "
            f"alpha={vals['alpha']:.6e}  expected={expected:.10e} "
            f"computed={np.real(computed):.10e} rel={rel:.3e}"
        )
        self.assertLess(rel, CLAMPED_RTOL)

    def test_clamped_uniform_temperature_3d_constant_control(self):
        mat = make_material(tables=False)
        vals = {"E": E0, "nu": NU0, "alpha": ALPHA0}
        expected = expected_clamped_compliance_3d(vals, self.theta, self.L**3)
        computed, U = self.run_case(mat)

        self.assertLess(np.max(np.abs(np.real(U[:, :3]))), 1e-12)
        rel = abs(np.real(computed) - expected) / abs(expected)
        print(
            f"\n[3D constant] expected={expected:.10e} "
            f"computed={np.real(computed):.10e} rel={rel:.3e}"
        )
        self.assertLess(rel, CLAMPED_RTOL)


@unittest.skipIf(COMM.size > 1, "1D conduction check runs serially only")
class TestNonlinearConduction1D(unittest.TestCase):
    """
    Steady 1D conduction with kappa(theta) linear in theta, mechanically fixed strip.

    theta = T - Tref is the temperature change. With kappa(theta) =
    300 - 0.4 (theta + 100), the Kirchhoff transform F(theta) =
    int_{-100}^{theta} kappa(s) ds is linear in x. With u = theta + 100 (that is,
    u = T - 200 for Tref = 300), F = 300 u - 0.2 u^2 (taking the lower limit at
    theta = -100, which only shifts F by a constant). Solving the quadratic for
    u gives
        u(x) = (300 - sqrt(90000 - 0.8 F(x))) / 0.4
    and theta(x) = u(x) - 100.
    """

    Lx = 1.0
    Ly = 0.1
    nx = 64
    ny = 2
    t = 0.01

    theta_left = 0.0
    theta_right = 300.0
    kap_theta = np.array([-100.0, 400.0])
    kap_v = np.array([300.0, 100.0])

    @classmethod
    def F(cls, theta):
        u = theta + 100.0
        return 300.0 * u - 0.2 * u**2

    @classmethod
    def theta_analytic(cls, x):
        F_left = cls.F(cls.theta_left)
        F_right = cls.F(cls.theta_right)
        F = F_left + (F_right - F_left) * np.asarray(x) / cls.Lx
        u = (300.0 - np.sqrt(90000.0 - 0.8 * F)) / 0.4
        return u - 100.0

    def make_kappa_material(self):
        return constitutive.MaterialProperties(
            rho=RHO,
            E=E0,
            nu=NU0,
            ys=YS,
            alpha=ALPHA0,
            kappa=np.array([self.kap_theta, self.kap_v], dtype=TACS_DTYPE),
            specific_heat=CP0,
        )

    def test_nonlinear_conduction_1d(self):
        bdf, nid, node_xy = quad_plate_bdf(self.nx, self.ny, self.Lx, self.Ly, self.t)
        all_nodes = list(node_xy.keys())
        left = [nid(0, j) for j in range(self.ny + 1)]
        right = [nid(self.nx, j) for j in range(self.ny + 1)]

        # Mechanically fixed everywhere, so only conduction matters
        add_spc_set(bdf, 2, all_nodes, "12", 0.0)
        # Dirichlet temperature changes theta on the left and right edges
        add_spc_set(bdf, 1, left, "3", self.theta_left)
        add_spc_set(bdf, 1, right, "3", self.theta_right)
        # Top and bottom edges are insulated (natural BC, nothing to apply)

        mat = self.make_kappa_material()
        prob, funcs, fea = solve_problem(
            bdf, mat, "2d", t=self.t, avg_volume=self.Lx * self.Ly
        )

        # Nodal state and coordinates (same local ordering in serial)
        U = np.real(prob.getVariables()).reshape(-1, 3)
        X = np.real(prob.getNodes()).reshape(-1, 3)
        theta_fe = U[:, 2]
        x = X[:, 0]
        theta_an = self.theta_analytic(x)

        # Midpoint temperature (linear constant-kappa profile would give 150 K)
        mid_nodes = np.where(np.isclose(x, 0.5 * self.Lx))[0]
        mid_theta = np.mean(theta_fe[mid_nodes])

        # (a) nodal temperature vs exact, normalized by the max exact theta
        err = np.abs(theta_fe - theta_an)
        scale = np.max(np.abs(theta_an))
        max_err_rel = np.max(err) / scale
        interior = theta_an > 0.5 * scale
        pointwise_rel = np.max(err[interior] / np.abs(theta_an[interior]))
        print(
            f"\n[1D nonlinear] max |theta_fe - theta_an| = {np.max(err):.6e} K, "
            f"normalized by max theta ({scale:.1f} K) = {max_err_rel:.3e}, "
            f"max pointwise rel (theta>0.5*max) = {pointwise_rel:.3e}, "
            f"midpoint theta_fe = {mid_theta:.6f}, exact = {self.theta_analytic(0.5 * self.Lx):.6f}"
        )
        self.assertLess(max_err_rel, 2e-3)

        # (b) average temperature vs the integral of the exact profile
        n_fine = 200001
        xf = np.linspace(0.0, self.Lx, n_fine)
        thf = self.theta_analytic(xf)
        dx = xf[1] - xf[0]
        avg_exact = np.sum(0.5 * (thf[1:] + thf[:-1])) * dx / self.Lx
        avg_fe = np.real(funcs["thermo_avg_temp"])
        rel_avg = abs(avg_fe - avg_exact) / abs(avg_exact)
        print(
            f"[1D nonlinear] avg theta expected = {avg_exact:.8f}, "
            f"computed = {avg_fe:.8f}, rel = {rel_avg:.3e}"
        )
        self.assertLess(rel_avg, 2e-3)

        # The state must differ from the constant-kappa (linear) profile
        self.assertGreater(abs(mid_theta - 150.0), 5.0)
