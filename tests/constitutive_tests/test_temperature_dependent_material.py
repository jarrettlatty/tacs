import unittest

import numpy as np

from tacs import TACS, constitutive, elements


class TemperatureDependentMaterialTest(unittest.TestCase):
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

        # Constant material properties (used when no table is supplied)
        self.rho = 2700.0
        self.E = 70e9
        self.nu = 0.3
        self.alpha = 24e-6
        self.kappa = 230.0
        self.cp = 921.0

        self.constant_values = {
            "E": self.E,
            "nu": self.nu,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "specific_heat": self.cp,
        }

        # Tables are keyed on the temperature change theta = T - Tref, the
        # temperature state variable used by TACS. The absolute temperatures
        # T = [200, 400, 600, 800] K with Tref = 300 K give
        # theta = [-100, 100, 300, 500]. Each table has a different slope on
        # every segment.
        self.Tref = 300.0
        self.theta = np.array([200.0, 400.0, 600.0, 800.0]) - self.Tref
        self.tables = {
            "E": np.array([self.theta, [72e9, 70e9, 64e9, 50e9]]),
            "nu": np.array([self.theta, [0.30, 0.31, 0.33, 0.36]]),
            "alpha": np.array([self.theta, [21e-6, 23e-6, 26e-6, 30e-6]]),
            "kappa": np.array([self.theta, [220.0, 230.0, 235.0, 238.0]]),
            "specific_heat": np.array([self.theta, [800.0, 921.0, 1000.0, 1050.0]]),
        }

        # Temperature changes used to compare evaluations. Includes breakpoints
        # and points outside the table range.
        self.eval_thetas = [-150.0, -100.0, -20.0, 37.0, 210.0, 365.0, 500.0, 650.0]

        # Material with no tables, and material with all tables set by the setter
        self.const_mat = self.make_material()
        self.temp_mat = self.make_material(self.tables)

        # Seed random number generator in tacs for consistent test results
        elements.SeedRandomGenerator(0)

    def make_material(self, tables=None):
        """Create the isotropic material, optionally adding tables with the setter"""
        mat = constitutive.MaterialProperties(
            rho=self.rho,
            E=self.E,
            nu=self.nu,
            alpha=self.alpha,
            kappa=self.kappa,
            specific_heat=self.cp,
        )
        if tables is not None:
            for name, table in tables.items():
                mat.setTemperatureDependentProperty(
                    name, np.array(table, dtype=self.dtype)
                )
        return mat

    def make_constructor_material(self, **overrides):
        """Create the isotropic material, replacing properties with constructor args"""
        kwargs = {
            "rho": self.rho,
            "E": self.E,
            "nu": self.nu,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "specific_heat": self.cp,
        }
        kwargs.update(overrides)
        return constitutive.MaterialProperties(**kwargs)

    def expected_value(self, name, theta):
        """Piecewise-linear table value at the temperature change theta"""
        return np.interp(theta, *self.tables[name])

    def segment_slope(self, name, theta):
        """Slope of the table segment containing the interior change theta"""
        theta_t, vals = self.tables[name]
        i = np.searchsorted(theta_t, theta, side="right") - 1
        return (vals[i + 1] - vals[i]) / (theta_t[i + 1] - theta_t[i])

    def assert_same_evaluations(self, mat, ref):
        """Check that two materials give the same values and derivatives"""
        for name in self.tables:
            for theta in self.eval_thetas:
                with self.subTest(name=name, theta=theta):
                    value, dvalue = mat.evalTemperatureDependentProperty(name, theta)
                    ref_value, ref_dvalue = ref.evalTemperatureDependentProperty(
                        name, theta
                    )
                    np.testing.assert_allclose(
                        np.real(value), np.real(ref_value), rtol=1e-12
                    )
                    np.testing.assert_allclose(
                        np.real(dvalue), np.real(ref_dvalue), rtol=1e-12
                    )

    def test_constant_material_unchanged(self):
        mat = self.const_mat
        self.assertFalse(mat.isTemperatureDependent())
        for name, const in self.constant_values.items():
            for theta in [-150.0, 0.0, 75.0, 400.0]:
                with self.subTest(name=name, theta=theta):
                    value, dvalue = mat.evalTemperatureDependentProperty(name, theta)
                    np.testing.assert_allclose(np.real(value), const, rtol=1e-12)
                    self.assertEqual(np.real(dvalue), 0.0)

    def test_is_temperature_dependent(self):
        mat = self.temp_mat
        self.assertTrue(mat.isTemperatureDependent())
        for name in self.tables:
            with self.subTest(name=name):
                self.assertTrue(mat.isTemperatureDependent(name))

        # Material with only an E table
        mat = self.make_material({"E": self.tables["E"]})
        self.assertTrue(mat.isTemperatureDependent())
        self.assertTrue(mat.isTemperatureDependent("E"))
        self.assertFalse(mat.isTemperatureDependent("nu"))

        self.assertFalse(self.const_mat.isTemperatureDependent("E"))

    def test_interpolation_values(self):
        mat = self.temp_mat

        # Breakpoints and midpoints inside the table range, given as absolute
        # temperatures T and converted to theta = T - Tref
        T_values = [200.0, 250.0, 400.0, 455.0, 600.0, 799.0]
        for name in self.tables:
            for T in T_values:
                theta = T - self.Tref
                with self.subTest(name=name, T=T):
                    value, _ = mat.evalTemperatureDependentProperty(name, theta)
                    expected = self.expected_value(name, theta)
                    np.testing.assert_allclose(np.real(value), expected, rtol=1e-12)

        # Derivatives are only checked well inside segments (>= 10 K from
        # any breakpoint), since breakpoints are not differentiable
        T_interior = [250.0, 455.0, 700.0]
        for name in self.tables:
            for T in T_interior:
                theta = T - self.Tref
                with self.subTest(name=name, T=T):
                    _, dvalue = mat.evalTemperatureDependentProperty(name, theta)
                    expected = self.segment_slope(name, theta)
                    np.testing.assert_allclose(np.real(dvalue), expected, rtol=1e-12)

    def test_constant_extrapolation(self):
        mat = self.temp_mat
        for name in self.tables:
            _, vals = self.tables[name]
            # Below the first and above the last table temperature
            for T, end_value in [(100.0, vals[0]), (1000.0, vals[-1])]:
                theta = T - self.Tref
                with self.subTest(name=name, T=T):
                    value, dvalue = mat.evalTemperatureDependentProperty(name, theta)
                    np.testing.assert_allclose(np.real(value), end_value, rtol=1e-12)
                    self.assertEqual(np.real(dvalue), 0.0)

    def test_partial_tables(self):
        mat = self.make_material({"alpha": self.tables["alpha"]})
        self.assertTrue(mat.isTemperatureDependent())
        self.assertTrue(mat.isTemperatureDependent("alpha"))

        # Properties without a table keep their constructor values
        for name in ["E", "nu", "kappa", "specific_heat"]:
            with self.subTest(name=name):
                self.assertFalse(mat.isTemperatureDependent(name))
                value, dvalue = mat.evalTemperatureDependentProperty(name, 75.0)
                const = self.constant_values[name]
                np.testing.assert_allclose(np.real(value), const, rtol=1e-12)
                self.assertEqual(np.real(dvalue), 0.0)

        # alpha follows its table
        for T in [250.0, 455.0, 700.0]:
            theta = T - self.Tref
            with self.subTest(name="alpha", T=T):
                value, dvalue = mat.evalTemperatureDependentProperty("alpha", theta)
                expected = self.expected_value("alpha", theta)
                np.testing.assert_allclose(np.real(value), expected, rtol=1e-12)
                expected_slope = self.segment_slope("alpha", theta)
                np.testing.assert_allclose(np.real(dvalue), expected_slope, rtol=1e-12)

    def test_get_table_roundtrip(self):
        for name, table_in in self.tables.items():
            with self.subTest(name=name):
                table_out = self.temp_mat.getTemperatureDependentProperty(name)
                self.assertIsInstance(table_out, np.ndarray)
                self.assertEqual(table_out.shape, (2, 4))
                np.testing.assert_allclose(np.real(table_out), table_in, rtol=1e-12)

                # The returned array is a copy, so editing it leaves the material as is
                table_out[1, 0] = 0.0
                table_again = self.temp_mat.getTemperatureDependentProperty(name)
                np.testing.assert_allclose(np.real(table_again), table_in, rtol=1e-12)

        # Properties without a table return None
        for name in self.constant_values:
            with self.subTest(name=name):
                result = self.const_mat.getTemperatureDependentProperty(name)
                self.assertIsNone(result)

        # Partial material: only the alpha table exists
        mat = self.make_material({"alpha": self.tables["alpha"]})
        self.assertIsNone(mat.getTemperatureDependentProperty("E"))

    def test_temp_sens(self):
        # The properties are piecewise linear in T, so a large real-mode step
        # has negligible truncation error and avoids round-off error on large
        # values such as E ~ 1e11
        dh = self.dh if self.dtype is complex else 1e-3

        # Theta values give T = 250, 337, 510 and 665 K, all inside segments
        for theta in [-50.0, 37.0, 210.0, 365.0]:
            with self.subTest(theta=theta):
                fail = constitutive.TestMaterialPropertiesTempSens(
                    self.temp_mat,
                    theta,
                    dh,
                    self.print_level,
                    self.atol,
                    self.rtol,
                )
                self.assertFalse(fail)

    def test_constructor_tables_match_setter(self):
        # All five properties given as tables in the constructor
        mat = self.make_constructor_material(**self.tables)
        self.assertTrue(mat.isTemperatureDependent())
        for name in self.tables:
            with self.subTest(name=name):
                self.assertTrue(mat.isTemperatureDependent(name))
                table_out = mat.getTemperatureDependentProperty(name)
                np.testing.assert_allclose(
                    np.real(table_out), self.tables[name], rtol=1e-12
                )

        # Same evaluations as the material built with the setter
        self.assert_same_evaluations(mat, self.temp_mat)

    def test_constructor_mixed_scalars_and_tables(self):
        # E, alpha and specific_heat are tables, nu and kappa are fixed scalars
        table_names = ["E", "alpha", "specific_heat"]
        mat = self.make_constructor_material(
            E=self.tables["E"],
            alpha=self.tables["alpha"],
            specific_heat=self.tables["specific_heat"],
        )
        self.assertTrue(mat.isTemperatureDependent())
        for name in self.constant_values:
            is_table = name in table_names
            with self.subTest(name=name):
                self.assertEqual(mat.isTemperatureDependent(name), is_table)
                for theta in self.eval_thetas:
                    value, dvalue = mat.evalTemperatureDependentProperty(name, theta)
                    if is_table:
                        expected = self.expected_value(name, theta)
                        np.testing.assert_allclose(
                            np.real(value), expected, rtol=1e-12
                        )
                    else:
                        np.testing.assert_allclose(
                            np.real(value), self.constant_values[name], rtol=1e-12
                        )
                        self.assertEqual(np.real(dvalue), 0.0)

    def test_constructor_table_constant_value(self):
        # The constant value reported by getMaterialProperties() is the table
        # value at theta = 0 for tables given in the constructor
        mat = self.make_constructor_material(**self.tables)
        props = mat.getMaterialProperties()
        constant_keys = {"E": "E1", "nu": "nu12", "alpha": "alpha1", "kappa": "kappa1"}
        for name, key in constant_keys.items():
            with self.subTest(name=name, key=key):
                expected = self.expected_value(name, 0.0)
                np.testing.assert_allclose(np.real(props[key]), expected, rtol=1e-12)

                # The same value is returned by the evaluation at theta = 0
                value, _ = mat.evalTemperatureDependentProperty(name, 0.0)
                np.testing.assert_allclose(np.real(value), expected, rtol=1e-12)

        # A later setter call changes the table but not the constant value
        new_E = np.array([[-100.0, 100.0], [60e9, 40e9]])
        new_alpha = np.array([[-100.0, 100.0], [10e-6, 30e-6]])
        before = mat.getMaterialProperties()
        mat.setTemperatureDependentProperty("E", new_E)
        mat.setTemperatureDependentProperty("alpha", new_alpha)
        after = mat.getMaterialProperties()
        for key in constant_keys.values():
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    np.real(after[key]), np.real(before[key]), rtol=1e-12
                )

        # The E table did change, so its value at theta = 0 is now 50 GPa
        value, _ = mat.evalTemperatureDependentProperty("E", 0.0)
        np.testing.assert_allclose(np.real(value), 50e9, rtol=1e-12)

    def test_single_point_constructor_table(self):
        # A (2, 1) table given in the constructor is a fixed value
        mat = self.make_constructor_material(E=[[0.0], [65e9]])
        self.assertFalse(mat.isTemperatureDependent())
        self.assertFalse(mat.isTemperatureDependent("E"))
        self.assertIsNone(mat.getTemperatureDependentProperty("E"))
        props = mat.getMaterialProperties()
        np.testing.assert_allclose(np.real(props["E1"]), 65e9, rtol=1e-12)
        for theta in [-100.0, 0.0, 250.0]:
            with self.subTest(theta=theta):
                value, dvalue = mat.evalTemperatureDependentProperty("E", theta)
                np.testing.assert_allclose(np.real(value), 65e9, rtol=1e-12)
                self.assertEqual(np.real(dvalue), 0.0)

        # The setter stores a single-point table, which makes the property
        # temperature dependent even though its value is constant
        mat = self.make_material()
        mat.setTemperatureDependentProperty("E", np.array([[0.0], [65e9]]))
        self.assertTrue(mat.isTemperatureDependent("E"))
        table = mat.getTemperatureDependentProperty("E")
        self.assertEqual(table.shape, (2, 1))

    def test_list_and_array_inputs(self):
        # Tables given as lists of lists are equivalent to numpy arrays, both in
        # the constructor and in the setter
        as_lists = {name: table.tolist() for name, table in self.tables.items()}
        mat_arrays = self.make_constructor_material(**self.tables)
        mat_lists = self.make_constructor_material(**as_lists)
        mat_setter_lists = self.make_material()
        for name, table in as_lists.items():
            mat_setter_lists.setTemperatureDependentProperty(name, table)

        self.assert_same_evaluations(mat_lists, mat_arrays)
        self.assert_same_evaluations(mat_setter_lists, mat_arrays)
        for name in self.tables:
            with self.subTest(name=name):
                table_out = mat_lists.getTemperatureDependentProperty(name)
                np.testing.assert_allclose(
                    np.real(table_out), self.tables[name], rtol=1e-12
                )

    def test_invalid_inputs(self):
        table = self.tables["E"]
        mat = self.make_material()

        # Unknown property name
        with self.assertRaises(ValueError):
            mat.setTemperatureDependentProperty("youngs", table)
        with self.assertRaises(ValueError):
            mat.getTemperatureDependentProperty("youngs")
        with self.assertRaises(ValueError):
            mat.evalTemperatureDependentProperty("youngs", 0.0)

        # A scalar is not a table for the setter
        with self.assertRaises(ValueError):
            mat.setTemperatureDependentProperty("E", self.E)

        # Malformed tables must be (2, n) arrays with theta strictly increasing
        bad_tables = {
            "(n, 2) array": table.T,
            "1-D array": table[1],
            "3-row array": np.vstack([table, np.ones(4)]),
            "empty (2, 0) array": np.zeros((2, 0)),
            "repeated theta": np.array([[-100.0, -100.0, 100.0], [72e9, 70e9, 64e9]]),
            "decreasing theta": np.array([[100.0, 0.0, -100.0], [72e9, 70e9, 64e9]]),
        }
        for label, bad in bad_tables.items():
            bad = np.array(bad, dtype=self.dtype)
            with self.subTest(table=label):
                with self.assertRaises(ValueError):
                    mat.setTemperatureDependentProperty("E", bad)
                with self.assertRaises(ValueError):
                    self.make_constructor_material(E=bad)

        # Temperature-dependent tables are only supported for isotropic materials
        ortho = constitutive.MaterialProperties(
            rho=1550.0, E1=54e9, E2=18e9, nu12=0.25, G12=9e9, G13=9e9
        )
        with self.assertRaises(ValueError):
            ortho.setTemperatureDependentProperty("E", table)
        with self.assertRaises(ValueError):
            constitutive.MaterialProperties(
                rho=1550.0, E1=54e9, E2=18e9, nu12=0.25, G12=9e9, G13=9e9, E=table
            )

        # Only the temperature-dependent properties may be given as arrays
        for name in ["rho", "ys"]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.make_constructor_material(**{name: table})


if __name__ == "__main__":
    unittest.main()
