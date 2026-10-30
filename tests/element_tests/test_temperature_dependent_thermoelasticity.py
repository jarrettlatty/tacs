import unittest

import numpy as np

from tacs import TACS, constitutive, elements

# Temperature changes theta = T - Tref [K] at which the tables are given. The
# element temperature state is theta and the random states used by the tests
# lie in [-1, 1], so theta stays strictly inside the [-20, 20] segment, where
# the piecewise-linear tables are differentiable
THETA_TABLE = [-100.0, -20.0, 20.0, 100.0]

TABLES = {
    "E": [75e3, 72e3, 66e3, 50e3],
    "nu": [0.28, 0.29, 0.31, 0.34],
    "alpha": [20e-6, 22e-6, 30e-6, 40e-6],
    "kappa": [240.0, 235.0, 225.0, 200.0],
    "specific_heat": [850.0, 900.0, 950.0, 1050.0],
}


def make_table(values):
    """Return the (2, n) table [theta; values] for the constructor"""
    return np.array([THETA_TABLE, values], dtype=TACS.dtype)


def make_props(temperature_dependent):
    """Create the isotropic material, with (2, n) theta tables if requested"""
    kwargs = dict(
        rho=2700.0,
        specific_heat=921.096,
        E=70e3,
        nu=0.3,
        ys=270.0,
        alpha=24.0e-6,
        kappa=230.0,
    )
    if temperature_dependent:
        for name, values in TABLES.items():
            kwargs[name] = make_table(values)
    return constitutive.MaterialProperties(**kwargs)


class ThermoelasticityTestMixin:
    """Shared tests for the linear thermoelasticity models.

    Subclasses set temperature_dependent and elem_rtol_complex, and implement
    make_constitutive, make_model, make_element and make_bases.
    """

    temperature_dependent = False
    elem_rtol_complex = None

    def setUp(self):
        max_nodes = 64
        max_vars_per_nodes = 8
        max_vars = max_nodes * max_vars_per_nodes

        # fd/cs step size and tolerances, following the model-level tests
        # (elem_rtol follows the element-level tests of the same dimension)
        if TACS.dtype is complex:
            self.dh = 1e-200
            self.rtol = 1e-11
            self.elem_rtol = self.elem_rtol_complex
        else:
            self.dh = 1e-6
            self.rtol = 1e-2
            self.elem_rtol = 1e-2
        self.dtype = TACS.dtype

        self.atol = np.clip(1e-5 * self.rtol, 1e-14, 1e-8)
        self.elem_atol = np.clip(1e-5 * self.elem_rtol, 1e-14, 1e-8)
        self.print_level = 0

        # Set element index
        self.elem_index = 0
        # Set the simulation time
        self.time = 0.0

        # Set the variable arrays
        np.random.seed(30)  # Seed random numbers for deterministic/repeatable tests
        self.xpts = np.random.rand(3 * max_nodes).astype(self.dtype)
        np.random.seed(30)  # Seed random numbers for deterministic/repeatable tests
        self.vars = np.random.rand(max_vars).astype(self.dtype)
        self.dvars = self.vars.copy()
        self.ddvars = self.vars.copy()

        # Keep the material, constitutive object, model and bases on self so
        # they outlive the models and elements that refer to them
        self.props = make_props(self.temperature_dependent)
        self.con = self.make_constitutive(self.props)
        self.model = self.make_model(self.con)
        self.bases = self.make_bases()

        # Seed random number generator in tacs for consistent test results
        elements.SeedRandomGenerator(0)

    def test_element_model_jacobian(self):
        fail = elements.TestElementModelJacobian(
            self.model,
            self.elem_index,
            self.time,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_element_model_adj_xpt_sens_product(self):
        fail = elements.TestElementModelAdjXptSensProduct(
            self.model,
            self.elem_index,
            self.time,
            self.dh,
            self.print_level,
            self.atol,
            self.rtol,
        )
        self.assertFalse(fail)

    def test_element_model_quantity_sens(self):
        quantities = {
            "FAILURE_INDEX": elements.FAILURE_INDEX,
            "HEAT_FLUX": elements.HEAT_FLUX,
            "TEMPERATURE": elements.TEMPERATURE,
            "STRAIN_ENERGY_DENSITY": elements.STRAIN_ENERGY_DENSITY,
            "TOTAL_STRAIN_ENERGY_DENSITY": elements.TOTAL_STRAIN_ENERGY_DENSITY,
            "ELEMENT_DENSITY": elements.ELEMENT_DENSITY,
            "ELEMENT_DISPLACEMENT": elements.ELEMENT_DISPLACEMENT,
        }
        for name, quantity in quantities.items():
            with self.subTest(quantity=name):
                fail = elements.TestElementModelQuantitySens(
                    self.model,
                    self.elem_index,
                    quantity,
                    self.time,
                    self.dh,
                    self.print_level,
                    self.atol,
                    self.rtol,
                )
                self.assertFalse(fail)

    def test_element_jacobian(self):
        # Loop through every basis class and test Jacobian
        for basis in self.bases:
            with self.subTest(basis=type(basis).__name__):
                element = self.make_element(self.model, basis)
                fail = elements.TestElementJacobian(
                    element,
                    self.elem_index,
                    self.time,
                    self.xpts,
                    self.vars,
                    self.dvars,
                    self.ddvars,
                    -1,
                    self.dh,
                    self.print_level,
                    self.elem_atol,
                    self.elem_rtol,
                )
                self.assertFalse(fail)

    def test_adj_res_product(self):
        # Loop through every basis class and test adjoint residual-dvsens product
        for basis in self.bases:
            with self.subTest(basis=type(basis).__name__):
                element = self.make_element(self.model, basis)
                dvs = element.getDesignVars(self.elem_index)
                fail = elements.TestAdjResProduct(
                    element,
                    self.elem_index,
                    self.time,
                    self.xpts,
                    self.vars,
                    self.dvars,
                    self.ddvars,
                    dvs,
                    self.dh,
                    self.print_level,
                    self.elem_atol,
                    self.elem_rtol,
                )
                self.assertFalse(fail)

    def test_adj_res_xpt_product(self):
        # Loop through every basis class and test adjoint residual-xptsens product
        for basis in self.bases:
            with self.subTest(basis=type(basis).__name__):
                element = self.make_element(self.model, basis)
                fail = elements.TestAdjResXptProduct(
                    element,
                    self.elem_index,
                    self.time,
                    self.xpts,
                    self.vars,
                    self.dvars,
                    self.ddvars,
                    self.dh,
                    self.print_level,
                    self.elem_atol,
                    self.elem_rtol,
                )
                self.assertFalse(fail)


class Linear2DMixin(ThermoelasticityTestMixin):
    """Plane stress thermoelasticity hooks"""

    elem_rtol_complex = 1e-10

    def make_constitutive(self, props):
        return constitutive.PlaneStressConstitutive(props, t=1.0, tNum=0)

    def make_model(self, con):
        return elements.LinearThermoelasticity2D(con)

    def make_element(self, model, basis):
        return elements.Element2D(model, basis)

    def make_bases(self):
        return [elements.LinearQuadBasis(), elements.QuadraticTriangleBasis()]


class Linear3DMixin(ThermoelasticityTestMixin):
    """Solid thermoelasticity hooks"""

    elem_rtol_complex = 1e-6

    def make_constitutive(self, props):
        return constitutive.SolidConstitutive(props, t=1.0, tNum=0)

    def make_model(self, con):
        return elements.LinearThermoelasticity3D(con)

    def make_element(self, model, basis):
        return elements.Element3D(model, basis)

    def make_bases(self):
        return [elements.LinearHexaBasis(), elements.LinearTetrahedralBasis()]


class Linear2DConstantTest(Linear2DMixin, unittest.TestCase):
    temperature_dependent = False


class Linear2DTemperatureDependentTest(Linear2DMixin, unittest.TestCase):
    temperature_dependent = True


class Linear3DConstantTest(Linear3DMixin, unittest.TestCase):
    temperature_dependent = False


class Linear3DTemperatureDependentTest(Linear3DMixin, unittest.TestCase):
    temperature_dependent = True


if __name__ == "__main__":
    unittest.main()
