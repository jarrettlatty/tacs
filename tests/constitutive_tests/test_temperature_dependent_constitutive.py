import unittest

import numpy as np

from tacs import TACS, constitutive, elements


def make_material(dtype, temperature_dependent):
    """Isotropic material, optionally with property tables attached"""
    mat = constitutive.MaterialProperties(
        rho=2700.0,
        E=70e9,
        nu=0.3,
        ys=270e6,
        alpha=24e-6,
        kappa=230.0,
        specific_heat=921.0,
    )
    if temperature_dependent:
        # Temperature changes theta = T - Tref [K] shared by all tables, for absolute
        # temperatures [200, 400, 600, 800] K and Tref = 300 K
        theta = [-100.0, 100.0, 300.0, 500.0]
        tables = {
            "E": [72e9, 70e9, 64e9, 50e9],
            "nu": [0.30, 0.31, 0.33, 0.36],
            "alpha": [21e-6, 23e-6, 26e-6, 30e-6],
            "kappa": [220.0, 230.0, 235.0, 238.0],
            "specific_heat": [800.0, 921.0, 1000.0, 1050.0],
        }
        for name, values in tables.items():
            mat.setTemperatureDependentProperty(
                name, np.array([theta, values], dtype=dtype)
            )
    return mat


class TemperatureDependentConstitutiveMixin:
    """Temperature-dependent material tests shared by the constitutive classes.

    Subclasses set constitutive_class to the constitutive type under test.
    """

    constitutive_class = None

    def setUp(self):
        # fd/cs step size
        if TACS.dtype is complex:
            self.dh = 1e-200
            self.rtol = 1e-11
        else:
            self.dh = 1e-8
            self.rtol = 1e-4
        self.dtype = TACS.dtype

        self.atol = np.clip(1e-5 * self.rtol, 1e-14, 1e-8)
        self.print_level = 0

        # Step for the temperature sensitivity check. In real mode a tiny step
        # (such as the 1e-8 used for design variables) loses digits on large
        # values like E ~ 7e10 when differentiating with respect to
        # temperature. The tables are piecewise linear, so a 1e-3 step has
        # negligible truncation error.
        self.temp_dh = self.dh if self.dtype is complex else 1e-3

        # Set element index
        self.elem_index = 0

        # Set the variable arrays
        self.x = np.ones(3, dtype=self.dtype)
        self.pt = np.zeros(3)
        self.dvs = np.array([0.1], dtype=self.dtype)

        # Keep references to the materials so the constitutive objects never
        # outlive them
        self.const_mat = make_material(self.dtype, False)
        self.temp_mat = make_material(self.dtype, True)
        self.const_con = self.constitutive_class(self.const_mat, t=0.1, tNum=0)
        self.temp_con = self.constitutive_class(self.temp_mat, t=0.1, tNum=0)

        # Seed random number generator in tacs for consistent test results
        elements.SeedRandomGenerator(0)

    def test_temp_sens(self):
        # Theta values give T = 250, 337, 510 and 665 K, each at least 10 K
        # away from the table breakpoints (200, 400, 600 and 800 K)
        for theta in [-50.0, 37.0, 210.0, 365.0]:
            with self.subTest(theta=theta):
                fail = constitutive.TestConstitutiveTempSens(
                    self.temp_con,
                    self.elem_index,
                    self.pt,
                    self.x,
                    self.dvs,
                    theta,
                    self.temp_dh,
                    self.print_level,
                    self.atol,
                    self.rtol,
                )
                self.assertFalse(fail)

    def test_constant_material_consistency(self):
        # Without tables, the temperature-dependent evaluations must exactly
        # reproduce the original ones
        for theta in [-50.0, 0.0, 120.0]:
            with self.subTest(theta=theta):
                fail = constitutive.TestConstitutiveTempSens(
                    self.const_con,
                    self.elem_index,
                    self.pt,
                    self.x,
                    self.dvs,
                    theta,
                    self.temp_dh,
                    self.print_level,
                    self.atol,
                    self.rtol,
                )
                self.assertFalse(fail)

    def test_existing_interfaces_with_temperature_dependent_material(self):
        # The temperature-independent code paths must still work when tables
        # are attached to the material
        self.assertTrue(self.temp_mat.isTemperatureDependent())

        dv_tests = [
            constitutive.TestConstitutiveDensity,
            constitutive.TestConstitutiveSpecificHeat,
            constitutive.TestConstitutiveHeatFlux,
            constitutive.TestConstitutiveStress,
            constitutive.TestConstitutiveThermalStrain,
            constitutive.TestConstitutiveFailure,
        ]
        for test_func in dv_tests:
            with self.subTest(test=test_func.__name__):
                fail = test_func(
                    self.temp_con,
                    self.elem_index,
                    self.pt,
                    self.x,
                    self.dvs,
                    self.dh,
                    self.print_level,
                    self.atol,
                    self.rtol,
                )
                self.assertFalse(fail)

        with self.subTest(test="TestConstitutiveFailureStrainSens"):
            fail = constitutive.TestConstitutiveFailureStrainSens(
                self.temp_con,
                self.elem_index,
                self.pt,
                self.x,
                self.dh,
                self.print_level,
                self.atol,
                self.rtol,
            )
            self.assertFalse(fail)


class PlaneStressTemperatureDependentTest(
    TemperatureDependentConstitutiveMixin, unittest.TestCase
):
    constitutive_class = constitutive.PlaneStressConstitutive


class SolidTemperatureDependentTest(
    TemperatureDependentConstitutiveMixin, unittest.TestCase
):
    constitutive_class = constitutive.SolidConstitutive


if __name__ == "__main__":
    unittest.main()
