Temperature-dependent material properties
*****************************************

Overview
--------

TACS supports temperature-dependent isotropic material properties for coupled thermoelastic analysis.
The properties :math:`E`, :math:`\nu`, :math:`\alpha`, :math:`\kappa` and the specific heat :math:`c`
may each be given as a piecewise-linear table versus the temperature change :math:`\theta = T - T_{\mathrm{ref}}`,
which is the temperature state variable used by TACS. The tables are attached to a
:class:`~tacs.constitutive.MaterialProperties` object and are evaluated inside the constitutive and element
models at every quadrature point. Any property without a table keeps its constant value, so existing
models are unaffected.

The feature is available for the :class:`~tacs.constitutive.PlaneStressConstitutive` and
:class:`~tacs.constitutive.SolidConstitutive` classes, used with the
:class:`~tacs.elements.LinearThermoelasticity2D` and :class:`~tacs.elements.LinearThermoelasticity3D`
element models. Once any table is present the coupled problem is nonlinear. See `Python usage`_ below.

Temperature convention
----------------------

The temperature state variable :math:`\theta` used by TACS is a temperature **change** relative to the
stress-free reference state:

.. math::

    \theta = T - T_{\mathrm{ref}}

Here :math:`T` is the absolute temperature and :math:`T_{\mathrm{ref}}` is the absolute temperature at which
the structure is stress free. The variable :math:`\theta` is the quantity that multiplies :math:`\alpha` in
the thermal strain. TACS does not store :math:`T_{\mathrm{ref}}`. Property tables are keyed directly on
:math:`\theta`, so the user normalizes temperatures by subtracting :math:`T_{\mathrm{ref}}` before passing them
to TACS. For example, with :math:`T_{\mathrm{ref}} = 300\,\mathrm{K}`, the table entries :math:`\theta = 0`,
:math:`200\,\mathrm{K}` and :math:`400\,\mathrm{K}` describe absolute temperatures of 300, 500 and 700 K.

A temperature-dependent property :math:`p` is evaluated from its table as

.. math::

    p(\theta) = \hat{p}(\theta), \qquad
    \frac{\partial p}{\partial \theta} = \frac{\partial \hat{p}}{\partial \theta}

where :math:`\hat{p}(\theta)` is the piecewise-linear interpolant described below.

Thermal strain
--------------

The thermal strain uses the secant definition, which is the convention used by the Nastran ``MATT1``
and ``TABLEM1`` cards:

.. math::

    \varepsilon_{th}(\theta) = \alpha(\theta)\,\theta = \alpha(T)\,(T - T_{\mathrm{ref}})

Its derivative with respect to the temperature state is

.. math::

    \frac{d\varepsilon_{th}}{d\theta} = \alpha'(\theta)\,\theta + \alpha(\theta)

For a constant :math:`\alpha` the expression reduces to :math:`\varepsilon_{th} = \alpha\,\theta`, with
derivative :math:`\alpha`, which is the original TACS thermoelastic model.

The stress is obtained from the tangent stiffness :math:`\mathbf{C}(\theta)` built from :math:`E(\theta)` and
:math:`\nu(\theta)`, applied to the mechanical strain (total strain less thermal strain):

.. math::

    \boldsymbol{\sigma} = \mathbf{C}(\theta)\left(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{th}(\theta)\right)

The heat flux follows Fourier's law with the temperature-dependent conductivity, :math:`\mathbf{q} = -\kappa(\theta)\nabla\theta`,
and the specific heat :math:`c(\theta)` multiplies the time-derivative term of the heat equation.

Property tables and interpolation
---------------------------------

A table consists of strictly increasing temperature changes :math:`\theta_0 < \theta_1 < \dots < \theta_{n-1}` and the
corresponding property values :math:`\hat{p}_0, \dots, \hat{p}_{n-1}`. Values are interpolated piecewise
linearly in :math:`\theta`. For :math:`\theta_i \le \theta < \theta_{i+1}` the segment :math:`i` is used:

.. math::

    \hat{p}(\theta) = \begin{cases}
    \hat{p}_0, & \theta < \theta_0 \\
    \hat{p}_i + s_i \left(\theta - \theta_i\right), \quad s_i = \dfrac{\hat{p}_{i+1} - \hat{p}_i}{\theta_{i+1} - \theta_i}, & \theta_i \le \theta < \theta_{i+1} \\
    \hat{p}_{n-1}, & \theta \ge \theta_{n-1}
    \end{cases}

Outside the table range the end values are held constant, so the derivative is zero there. Inside the
table the derivative is the segment slope :math:`s_i`. The derivative is discontinuous at the interior
breakpoints :math:`\theta_i`.

.. note::

    **Future work.** The derivative discontinuity at the breakpoints is a known limitation of the current
    evaluator (``TACSMaterialProperties::evalTemperatureDependentProperty``). If Newton or adjoint
    convergence suffers near breakpoints, a smoothed interpolant with a continuous derivative may be needed.

Input format
------------

Each of ``E``, ``nu``, ``alpha``, ``kappa`` and ``specific_heat`` may be given either as:

* a scalar, which is a fixed value, or
* a (2, n) array, where row 0 holds the strictly increasing temperature changes :math:`\theta = T - T_{\mathrm{ref}}`
  and row 1 holds the property values.

Tables are passed directly to the :class:`~tacs.constitutive.MaterialProperties` constructor. This is the
recommended way to set them inside an ``elemCallBack``. The same (2, n) table format is used by
``setTemperatureDependentProperty(name, table)``, and ``getTemperatureDependentProperty(name)`` returns the
table as a (2, n) array, or ``None`` if no table is set.

The following rules apply to constructor tables:

* The constant value used by functions that do not take a temperature (for example, the functions used by
  shell elements) is the table value at :math:`\theta = 0`. The end values are held constant outside the
  table range.
* A single-point table is treated as a fixed value.
* Passing an array for any property other than ``E``, ``nu``, ``alpha``, ``kappa`` or ``specific_heat``, or
  setting a table on an orthotropic material, raises a ``ValueError``. Passing an unknown name to
  ``setTemperatureDependentProperty`` also raises a ``ValueError``.

Python usage
------------

Temperature-dependent tables are most easily given in the ``elemCallBack``, where the material is created.
The following example shows a 2D plane stress thermoelastic model in a callback passed to
:meth:`pyTACS.initialize <tacs.pytacs.pyTACS.initialize>`. It follows the pattern of the
:doc:`plate example </examples/Example-Plate>`. The mesh is read from a BDF file.

.. code-block:: python

  from tacs import constitutive, elements, pyTACS

  def elemCallBack(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
      # Tables are (2, n) arrays. Row 0 holds the temperature change theta = T - Tref,
      # strictly increasing, and row 1 the property values. Here Tref = 300 K, so
      # theta = 0, 200 and 400 K correspond to T = 300, 500 and 700 K.
      E = [[0.0, 200.0, 400.0], [70e9, 65e9, 55e9]]
      alpha = [[0.0, 200.0, 400.0], [24e-6, 26e-6, 29e-6]]

      # Isotropic material. Any property without a table keeps its constant value.
      prop = constitutive.MaterialProperties(
          rho=2700.0, specific_heat=921.0, E=E, nu=0.3, alpha=alpha, kappa=230.0,
      )

      # Plane stress constitutive model and 2D thermoelastic element model
      con = constitutive.PlaneStressConstitutive(prop, t=0.01, tNum=dvNum)
      model = elements.LinearThermoelasticity2D(con)
      basis = elements.LinearQuadBasis()

      elemList = []
      for descript in elemDescripts:
          if descript == "CQUAD4":
              elemList.append(elements.Element2D(model, basis))
          else:
              raise ValueError(f"Unexpected element of type {descript}.")
      return elemList

  bdfFile = './model.bdf'
  FEAAssembler = pyTACS(bdfFile)
  FEAAssembler.initialize(elemCallBack)
  staticProb = FEAAssembler.createStaticProblem("thermoelastic")

The material methods used for tables are:

* ``setTemperatureDependentProperty(name, table)``, which sets a (2, n) table, replacing any existing table
  for that property.
* ``getTemperatureDependentProperty(name)``, which returns the (2, n) table or ``None`` if no table is set.
* ``isTemperatureDependent(name=None)``, which checks one property or, with no argument, any property.
* ``evalTemperatureDependentProperty(name, theta)``, which returns ``(value, dvalue_dT)`` at the
  temperature change :math:`\theta`.

The supported names are ``"E"``, ``"nu"``, ``"alpha"``, ``"kappa"`` and ``"specific_heat"``. Passing any other name,
or setting a table on an orthotropic material, raises a ``ValueError``.

Once any table is present the thermoelastic residual is nonlinear in the states. pyTACS detects this
automatically when the problem is initialized. It checks the residual for linearity using three residual
evaluations (``_checkNonlinearity`` in ``tacs/pytacs.py``). For a nonlinear problem, ``FEAAssembler.isNonlinear``
is True, and the static problem uses the ``Continuation`` nonlinear solver, which is the default
``nonlinearSolver`` option. No extra options are needed. The check is generic and is not specific to temperature tables.

Supported classes
-----------------

The following classes support temperature-dependent properties in this release:

* Constitutive classes: :class:`~tacs.constitutive.PlaneStressConstitutive` and
  :class:`~tacs.constitutive.SolidConstitutive`.
* Element models: :class:`~tacs.elements.LinearThermoelasticity2D` and
  :class:`~tacs.elements.LinearThermoelasticity3D`. These provide the residual, a consistent Jacobian including the
  temperature-coupling terms, adjoint and design-variable sensitivities, and the point quantities failure index,
  heat flux, strain energy density and total strain energy density.

Limitations
-----------

The following are not yet supported. Where a limitation is listed, the affected model keeps its constant properties.

* Anisotropic and orthotropic materials, and composite plies. Tables are isotropic only.
* Shells, plates, beams, and the pure heat-conduction models :class:`~tacs.elements.HeatConduction2D` and
  :class:`~tacs.elements.HeatConduction3D`.
* Temperature-dependent density and yield stress. The failure index depends on temperature only through the
  stiffness used to compute stress.
* The temperature-coupling Jacobian terms in the mechanical rows are implemented for linear strain only. This matches
  the rest of the existing thermoelastic Jacobian.
* BDF input. The temperature-dependent material cards in the BDF (``MATT1``, ``MATT2``, ``MATT3``, ``MATT4``,
  ``MATT5``, ``MATT8``, ``MATT9``, ``MATT11`` and ``TABLEM1`` to ``TABLEM4``) are not read by pyTACS. pyTACS prints
  a warning that lists the cards it found and ignores them, so the properties must be set in the ``elemCallBack``.
  ``writeBDF`` writes only the constant values, which are the table values at :math:`\theta = 0`.

Known issues
------------

The following issues already existed and are not caused by temperature-dependent properties, but they affect
coupled thermoelastic analyses that use this feature.

* **pyTACS adjoint sensitivities of coupled thermoelastic problems.** ``StaticProblem.solveAdjoint`` solves the
  adjoint with the Jacobian :math:`K` rather than :math:`K^T`. This is only correct for symmetric Jacobians. The
  thermoelastic Jacobian is not symmetric, because the temperature drives the displacements but not the reverse, and
  temperature-dependent conductivity also makes the thermal block nonsymmetric. Total sensitivities from pyTACS for
  coupled thermoelastic problems can therefore be wrong, with or without temperature-dependent properties. Solving the
  adjoint with the transposed Jacobian (``assembleJacobian(..., TACS.TRANSPOSE)``) gives sensitivities that match
  finite differences. The element-level Jacobian and adjoint products are verified, and the assembler-level test
  harness, which assembles the transposed Jacobian, is not affected.
* **AverageTemperature node-location sensitivities.** In coupled thermoelastic pyTACS tests, the node-location
  sensitivity of :class:`~tacs.functions.AverageTemperature` disagrees with finite differences, even with constant
  material properties and with the transposed Jacobian. The cause has not been identified yet.
* **Boundary conditions from pyNastran BDF objects.** When a pyNastran ``BDF`` object (rather than a file name) is
  passed to pyTACS, it is deep-copied. With some pyNastran versions, ``copy.deepcopy`` truncates multi-node ``SPC``
  cards to their first two nodes, which silently drops boundary conditions. Use one single-node ``SPC`` card per node
  when building BDF objects in code.

The integration test ``tests/integration_tests/test_thermoelast_tempdep_quad_2d.py`` marks its total node-location
sensitivity check as an expected failure because of the first two issues.

Developer notes
---------------

The C++ implementation is organized in three layers.

* ``TACSMaterialProperties`` (``src/constitutive/TACSMaterialProperties.h``) stores the tables and provides
  ``setTemperatureDependentProperty``, ``getTemperatureDependentProperty``, ``isTemperatureDependent`` and
  ``evalTemperatureDependentProperty``. It also provides theta-taking overloads of
  ``evalTangentStiffness2D/3D``, ``evalTangentHeatFlux2D/3D``, ``evalThermalStrain2D/3D`` (which returns the full
  thermal strain, not the coefficient), and ``getSpecificHeat``. Each has a matching ``...TempSens`` derivative.
* ``TACSConstitutive`` (``src/constitutive/TACSConstitutive.h``, section "Temperature-dependent constitutive
  evaluations") adds virtual theta overloads. These include ``isTemperatureDependent``, ``evalSpecificHeat``,
  ``evalStress``, ``evalTangentStiffness``, ``evalThermalStrainTempSens``, ``evalHeatFlux``, ``evalTangentHeatFlux``,
  ``evalFailure`` and ``evalFailureTempSens``. The ``add...TempSens`` and ``add...DVSens`` functions add
  ``scale`` times the derivative at fixed strain or gradient. The default implementations ignore ``theta`` and call the
  original functions, so classes that do not override them behave exactly as before.

Verification helpers check the derivatives by finite differences:

* ``TacsTestMaterialPropertiesTempSens``, exposed as ``constitutive.TestMaterialPropertiesTempSens(props, theta)``,
  checks the material-level derivatives.
* ``TacsTestConstitutiveTempSens``, exposed as ``constitutive.TestConstitutiveTempSens(con, elem_index, pt, x, dvs, theta)``,
  checks the constitutive-level derivatives.
* ``TacsTestElementModelQuantitySens``, exposed as ``elements.TestElementModelQuantitySens(model, elem_index, quantity_type, time)``,
  checks the point-quantity sensitivities of an element model.

The corresponding tests are in ``tests/constitutive_tests/test_temperature_dependent_material.py``,
``tests/constitutive_tests/test_temperature_dependent_constitutive.py`` and
``tests/element_tests/test_temperature_dependent_thermoelasticity.py``.
