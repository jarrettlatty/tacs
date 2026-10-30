"""
Temperature-dependent thermoelasticity on a BDF mesh read from a file.

The plate of test_thermoelast_tempdep_quad_2d is written to a BDF file in a
temporary directory. For one of the decks, raw temperature-dependent material
cards (MATT1 and TABLEM1) are appended to the file. pyTACS does not read these
cards: it prints a warning that they are ignored, and the temperature-dependent
properties have to be set in the elemCallBack. There they are given as (2, n)
tables of the temperature change theta = T - Tref (Tref = 300 K).

Tests:
    - constructing pyTACS from a deck with MATT1/TABLEM1 prints the warning
    - constructing pyTACS from a deck without them prints no warning
    - an elemCallBack with (2, n) tables on the file mesh reproduces the
      function values of test_thermoelast_tempdep_quad_2d, which shows that the
      ignored cards do not affect the result
"""

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
from mpi4py import MPI

from tacs import pytacs, elements, constitutive, functions

import test_thermoelast_tempdep_quad_2d as quad

# MAT1 for MID 1, written to every deck
MAT1_CARD = "MAT1,1,70.+9,,.3,2700.,24.-6,300.\n"

# Temperature-dependent material cards. pyTACS ignores these. They are appended
# as raw free-field text because pyNastran's add_matt1/add_matt4 are buggy here.
TEMP_DEP_CARDS = (
    "MATT1,1,10\n"
    "TABLEM1,10,,,,,,,,+T10\n"
    "+T10,250.,72.+9,350.,70.+9,450.,66.+9,ENDT\n"
)

# Substring printed by the pyTACS warning for these cards
WARNING_KEY = "temperature-dependent"


def write_deck(path, with_temp_dep_cards):
    """Write the quad plate mesh to path, optionally with temperature-dependent cards."""
    bdf = quad.create_bdf_object()
    bdf.write_bdf(path, enddata=True)

    with open(path) as f:
        text = f.read()

    cards = MAT1_CARD
    if with_temp_dep_cards:
        cards += TEMP_DEP_CARDS

    # Insert the cards before the ENDDATA line
    idx = text.rfind("ENDDATA")
    if idx < 0:
        raise RuntimeError(f"No ENDDATA line found in written BDF file '{path}'.")
    text = text[:idx] + cards + text[idx:]

    with open(path, "w") as f:
        f.write(text)


def elem_call_back(
    dv_num, comp_id, comp_descript, elem_descripts, special_dvs, **kwargs
):
    # Temperature-dependent properties as (2, n) tables in the temperature change
    # theta; scalar inputs are fixed values
    props = constitutive.MaterialProperties(
        rho=2700.0,
        specific_heat=quad.specific_heat_table,
        E=quad.E_table,
        nu=quad.nu_table,
        ys=270e6,
        alpha=quad.alpha_table,
        kappa=quad.kappa_table,
    )

    # One thickness design variable per component
    con = constitutive.PlaneStressConstitutive(props, t=quad.thickness, tNum=dv_num)

    # Steady state for both the mechanical and thermal parts
    steady_flag = elements.STEADY_STATE_MECHANICAL | elements.STEADY_STATE_THERMAL

    elem_list = []
    for elem_descript in elem_descripts:
        if elem_descript in ["CQUAD4", "CQUADR"]:
            basis = elements.LinearQuadBasis()
        else:
            raise ValueError(
                f"Element type '{elem_descript}' is not supported by this test"
            )
        model = elements.LinearThermoelasticity2D(con, steady_flag)
        elem = elements.Element2D(model, basis)
        elem_list.append(elem)

    # Scale for the thickness design variable
    scale = [1.0]
    return elem_list, scale


class TempDepBDFFileTest(unittest.TestCase):
    N_PROCS = 1  # this is how many MPI processes to use for this TestCase.

    def setUp(self):
        self.comm = MPI.COMM_WORLD

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        self.temp_deck = os.path.join(tmp.name, "tempdep_cards.bdf")
        self.plain_deck = os.path.join(tmp.name, "plain.bdf")
        write_deck(self.temp_deck, with_temp_dep_cards=True)
        write_deck(self.plain_deck, with_temp_dep_cards=False)

    def test_warns_for_temperature_cards(self):
        """A deck with MATT1/TABLEM1 cards prints the pyTACS warning on stdout."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pytacs.pyTACS(self.temp_deck, self.comm)
        out = buf.getvalue()

        self.assertIn(WARNING_KEY, out)
        self.assertIn("MATT1", out)
        self.assertIn("TABLEM1", out)

    def test_no_warning_without_temperature_cards(self):
        """A deck without temperature-dependent material cards prints no such warning."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pytacs.pyTACS(self.plain_deck, self.comm)
        out = buf.getvalue()

        self.assertNotIn(WARNING_KEY, out)

    def test_callback_tables_on_file_mesh(self):
        """(2, n) callback tables on the file mesh reproduce the reference functions."""
        fea_assembler = pytacs.pyTACS(self.temp_deck, self.comm)
        fea_assembler.initialize(elem_call_back)

        # Same options and nonlinear tolerances as test_thermoelast_tempdep_quad_2d
        options = {
            "L2Convergence": 1e-14,
            "L2ConvergenceRel": 1e-14,
            "resetBeforeSolve": True,
        }
        problem = fea_assembler.createStaticProblem(name="tempdep", options=options)
        problem.nonlinearSolver.setConvergenceTolerance(absTol=1e-9, relTol=1e-11)

        problem.addFunction("ks_failure", functions.KSFailure, ksWeight=quad.ksweight)
        problem.addFunction("compliance", functions.Compliance)
        problem.addFunction(
            "avg_temp", functions.AverageTemperature, volume=quad.Lx * quad.Ly
        )
        problem.addFunction("ks_temp", functions.KSTemperature, ksWeight=quad.ksweight)
        problem.addFunction("mass", functions.StructuralMass)

        problem.solve()
        funcs = {}
        problem.evalFunctions(funcs)

        refs = quad.ProblemTest.FUNC_REFS
        self.assertEqual(set(funcs), set(refs))
        for name, ref in refs.items():
            with self.subTest(function=name):
                np.testing.assert_allclose(funcs[name], ref, rtol=1e-10, atol=0.0)
