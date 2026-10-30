/*
  This file is part of TACS: The Toolkit for the Analysis of Composite
  Structures, a parallel finite-element code for structural and
  multidisciplinary design optimization.

  Copyright (C) 2014 Georgia Tech Research Corporation

  TACS is licensed under the Apache License, Version 2.0 (the
  "License"); you may not use this software except in compliance with
  the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
*/

#include "TACSConstitutiveVerification.h"

#include "tacslapack.h"

int TacsTestConstitutiveDensity(TACSConstitutive *con, int elemIndex,
                                const double pt[], const TacsScalar X[],
                                int ndvs, const TacsScalar *dvs, double dh,
                                int test_print_level, double test_fail_atol,
                                double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Sensitivity w.r.t. the density
  TacsScalar rho = con->evalDensity(elemIndex, pt, X);

  // Compute the sensitivity w.r.t. the density
  TacsScalar *xtemp = new TacsScalar[ndvs];
  TacsScalar *dfdx = new TacsScalar[ndvs];
  TacsScalar *dfdx_approx = new TacsScalar[ndvs];

  // Copy the design variable values
  memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

  // Generate a random array for the scale factor
  TacsScalar scale;
  TacsGenerateRandomArray(&scale, 1);

  // Evaluate the derivatives
  memset(dfdx, 0, ndvs * sizeof(TacsScalar));
  con->addDensityDVSens(elemIndex, scale, pt, X, ndvs, dfdx);

  // Compute the approximate derivative
  for (int i = 0; i < ndvs; i++) {
    TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
    xtemp[i] = x + TacsScalar(0.0, dh);
#else
    xtemp[i] = x + dh;
#endif
    con->setDesignVars(elemIndex, ndvs, xtemp);

    TacsScalar rho_forward = con->evalDensity(elemIndex, pt, X);
#ifdef TACS_USE_COMPLEX
    dfdx_approx[i] = scale * TacsImagPart(rho_forward) / dh;
#else
    dfdx_approx[i] = scale * (rho_forward - rho) / dh;
#endif
    xtemp[i] = x;
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the derivative of the density for %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(rho)/dx", dfdx, dfdx_approx, ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  delete[] xtemp;
  delete[] dfdx;
  delete[] dfdx_approx;

  return !allClose;
}

int TacsTestConstitutiveSpecificHeat(TACSConstitutive *con, int elemIndex,
                                     const double pt[], const TacsScalar X[],
                                     int ndvs, const TacsScalar *dvs, double dh,
                                     int test_print_level,
                                     double test_fail_atol,
                                     double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the sensitivity w.r.t. the density
  TacsScalar *xtemp = new TacsScalar[ndvs];
  TacsScalar *dfdx = new TacsScalar[ndvs];
  TacsScalar *dfdx_approx = new TacsScalar[ndvs];

  // Copy the design variable values
  memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

  // Generate a random array for the scale factor
  TacsScalar scale;
  TacsGenerateRandomArray(&scale, 1);

  // Evaluate the derivatives
  memset(dfdx, 0, ndvs * sizeof(TacsScalar));
  con->addSpecificHeatDVSens(elemIndex, scale, pt, X, ndvs, dfdx);

  // Evaluate the specific heat
  TacsScalar c = con->evalSpecificHeat(elemIndex, pt, X);

  // Compute the approximate derivative
  for (int i = 0; i < ndvs; i++) {
    TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
    xtemp[i] = x + TacsScalar(0.0, dh);
#else
    xtemp[i] = x + dh;
#endif
    con->setDesignVars(elemIndex, ndvs, xtemp);

    TacsScalar c_forward = con->evalSpecificHeat(elemIndex, pt, X);
#ifdef TACS_USE_COMPLEX
    dfdx_approx[i] = scale * TacsImagPart(c_forward) / dh;
#else
    dfdx_approx[i] = scale * (c_forward - c) / dh;
#endif
    xtemp[i] = x;
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the derivative of the specific heat for %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(c)/dx", dfdx, dfdx_approx, ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  delete[] xtemp;
  delete[] dfdx;
  delete[] dfdx_approx;

  return !allClose;
}

int TacsTestConstitutiveHeatFlux(TACSConstitutive *con, int elemIndex,
                                 const double pt[], const TacsScalar X[],
                                 int ndvs, const TacsScalar *dvs, double dh,
                                 int test_print_level, double test_fail_atol,
                                 double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  TacsScalar flux[3], grad[3], psi[3];
  TacsGenerateRandomArray(grad, 3);
  TacsGenerateRandomArray(psi, 3);

  // Allocate space for the derivatives
  TacsScalar *xtemp = new TacsScalar[ndvs];
  TacsScalar *dfdx = new TacsScalar[ndvs];
  TacsScalar *dfdx_approx = new TacsScalar[ndvs];

  // Copy the design variable values
  memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

  // Generate a random array for the scale factor
  TacsScalar scale;
  TacsGenerateRandomArray(&scale, 1);

  // Evaluate the derivatives
  memset(dfdx, 0, ndvs * sizeof(TacsScalar));
  con->addHeatFluxDVSens(elemIndex, scale, pt, X, grad, psi, ndvs, dfdx);

  flux[0] = flux[1] = flux[2] = 0.0;
  con->evalHeatFlux(elemIndex, pt, X, grad, flux);
  TacsScalar prod = 0.0;
  for (int j = 0; j < 3; j++) {
    prod += psi[j] * flux[j];
  }

  // Compute the approximate derivative
  for (int i = 0; i < ndvs; i++) {
    TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
    xtemp[i] = x + TacsScalar(0.0, dh);
#else
    xtemp[i] = x + dh;
#endif
    con->setDesignVars(elemIndex, ndvs, xtemp);

    flux[0] = flux[1] = flux[2] = 0.0;
    con->evalHeatFlux(elemIndex, pt, X, grad, flux);
    TacsScalar prod_forward = 0.0;
    for (int j = 0; j < 3; j++) {
      prod_forward += psi[j] * flux[j];
    }
#ifdef TACS_USE_COMPLEX
    dfdx_approx[i] = scale * TacsImagPart(prod_forward) / dh;
#else
    dfdx_approx[i] = scale * (prod_forward - prod) / dh;
#endif
    xtemp[i] = x;
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the derivative of the heat-flux product %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(psi^{T}flux)/dx", dfdx, dfdx_approx,
                             ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  delete[] xtemp;
  delete[] dfdx;
  delete[] dfdx_approx;

  return !allClose;
}

int TacsTestConstitutiveStress(TACSConstitutive *con, int elemIndex,
                               const double pt[], const TacsScalar X[],
                               int ndvs, const TacsScalar *dvs, double dh,
                               int test_print_level, double test_fail_atol,
                               double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  int nstress = con->getNumStresses();
  TacsScalar *s = new TacsScalar[nstress];
  TacsScalar *e = new TacsScalar[nstress];
  TacsScalar *psi = new TacsScalar[nstress];
  TacsGenerateRandomArray(e, nstress, -1e-3, 1e-3);
  TacsGenerateRandomArray(psi, nstress);

  // Allocate space for the derivatives
  TacsScalar *xtemp = new TacsScalar[ndvs];
  TacsScalar *dfdx = new TacsScalar[ndvs];
  TacsScalar *dfdx_approx = new TacsScalar[ndvs];

  // Copy the design variable values
  memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

  // Generate a random array for the scale factor
  TacsScalar scale;
  TacsGenerateRandomArray(&scale, 1);

  // Evaluate the derivatives
  memset(dfdx, 0, ndvs * sizeof(TacsScalar));
  con->addStressDVSens(elemIndex, scale, pt, X, e, psi, ndvs, dfdx);

  con->evalStress(elemIndex, pt, X, e, s);
  TacsScalar prod = 0.0;
  for (int j = 0; j < nstress; j++) {
    prod += psi[j] * s[j];
  }

  // Compute the approximate derivative
  for (int i = 0; i < ndvs; i++) {
    TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
    xtemp[i] = x + TacsScalar(0.0, dh);
#else
    xtemp[i] = x + dh;
#endif
    con->setDesignVars(elemIndex, ndvs, xtemp);

    con->evalStress(elemIndex, pt, X, e, s);
    TacsScalar prod_forward = 0.0;
    for (int j = 0; j < nstress; j++) {
      prod_forward += psi[j] * s[j];
    }
#ifdef TACS_USE_COMPLEX
    dfdx_approx[i] = scale * TacsImagPart(prod_forward) / dh;
#else
    dfdx_approx[i] = scale * (prod_forward - prod) / dh;
#endif
    xtemp[i] = x;
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the derivative of the stress-adjoint product %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(psi^{T}s)/dx", dfdx, dfdx_approx, ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  delete[] xtemp;
  delete[] dfdx;
  delete[] dfdx_approx;
  delete[] s;
  delete[] e;
  delete[] psi;

  return !allClose;
}

int TacsTestConstitutiveThermalStrain(TACSConstitutive *con, int elemIndex,
                                      const double pt[], const TacsScalar X[],
                                      int ndvs, const TacsScalar *dvs,
                                      double dh, int test_print_level,
                                      double test_fail_atol,
                                      double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  int nstress = con->getNumStresses();
  TacsScalar *e = new TacsScalar[nstress];
  TacsScalar *psi = new TacsScalar[nstress];
  TacsGenerateRandomArray(e, nstress, -1e-3, 1e-3);
  TacsGenerateRandomArray(psi, nstress);

  // Allocate space for the derivatives
  TacsScalar *xtemp = new TacsScalar[ndvs];
  TacsScalar *dfdx = new TacsScalar[ndvs];
  TacsScalar *dfdx_approx = new TacsScalar[ndvs];

  // Copy the design variable values
  memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

  // Generate a random array for the scale factor
  TacsScalar theta;
  TacsGenerateRandomArray(&theta, 1);

  // Evaluate the derivatives
  memset(dfdx, 0, ndvs * sizeof(TacsScalar));
  con->addThermalStrainDVSens(elemIndex, pt, X, theta, psi, ndvs, dfdx);

  con->evalThermalStrain(elemIndex, pt, X, theta, e);
  TacsScalar prod = 0.0;
  for (int j = 0; j < nstress; j++) {
    prod += psi[j] * e[j];
  }

  // Compute the approximate derivative
  for (int i = 0; i < ndvs; i++) {
    TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
    xtemp[i] = x + TacsScalar(0.0, dh);
#else
    xtemp[i] = x + dh;
#endif
    con->setDesignVars(elemIndex, ndvs, xtemp);

    con->evalThermalStrain(elemIndex, pt, X, theta, e);
    TacsScalar prod_forward = 0.0;
    for (int j = 0; j < nstress; j++) {
      prod_forward += psi[j] * e[j];
    }
#ifdef TACS_USE_COMPLEX
    dfdx_approx[i] = TacsImagPart(prod_forward) / dh;
#else
    dfdx_approx[i] = (prod_forward - prod) / dh;
#endif
    xtemp[i] = x;
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr,
            "Testing the derivative of the thermal strain-adjoint product %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(psi^{T}et)/dx", dfdx, dfdx_approx,
                             ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  delete[] xtemp;
  delete[] dfdx;
  delete[] dfdx_approx;
  delete[] e;
  delete[] psi;

  return !allClose;
}

int TacsTestConstitutiveFailure(TACSConstitutive *con, int elemIndex,
                                const double pt[], const TacsScalar X[],
                                int ndvs, const TacsScalar *dvs, double dh,
                                int test_print_level, double test_fail_atol,
                                double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  int nstress = con->getNumStresses();
  TacsScalar *e = new TacsScalar[nstress];
  TacsGenerateRandomArray(e, nstress, -1e-3, 1e-3);

  // Allocate space for the derivatives
  TacsScalar *xtemp = new TacsScalar[ndvs];
  TacsScalar *dfdx = new TacsScalar[ndvs];
  TacsScalar *dfdx_approx = new TacsScalar[ndvs];

  // Copy the design variable values
  memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

  // Generate a random array for the scale factor
  TacsScalar scale;
  TacsGenerateRandomArray(&scale, 1);

  // Evaluate the derivatives
  memset(dfdx, 0, ndvs * sizeof(TacsScalar));
  con->addFailureDVSens(elemIndex, scale, pt, X, e, ndvs, dfdx);

  TacsScalar failure = con->evalFailure(elemIndex, pt, X, e);

  // Compute the approximate derivative
  for (int i = 0; i < ndvs; i++) {
    TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
    xtemp[i] = x + TacsScalar(0.0, dh);
#else
    xtemp[i] = x + dh;
#endif
    con->setDesignVars(elemIndex, ndvs, xtemp);

    TacsScalar failure_forward = con->evalFailure(elemIndex, pt, X, e);
#ifdef TACS_USE_COMPLEX
    dfdx_approx[i] = scale * TacsImagPart(failure_forward) / dh;
#else
    dfdx_approx[i] = scale * (failure_forward - failure) / dh;
#endif
    xtemp[i] = x;
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr,
            "Testing the derivative of the failure index for object %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(failure)/dx", dfdx, dfdx_approx, ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  delete[] xtemp;
  delete[] dfdx;
  delete[] dfdx_approx;
  delete[] e;

  return !allClose;
}

int TacsTestConstitutiveFailureStrainSens(TACSConstitutive *con, int elemIndex,
                                          const double pt[],
                                          const TacsScalar X[], double dh,
                                          int test_print_level,
                                          double test_fail_atol,
                                          double test_fail_rtol) {
  int nstress = con->getNumStresses();
  TacsScalar *e = new TacsScalar[nstress];
  TacsGenerateRandomArray(e, nstress, -1e-3, 1e-3);

  // Allocate space for the derivatives
  TacsScalar *dfde = new TacsScalar[nstress];
  TacsScalar *dfde_approx = new TacsScalar[nstress];

  // Evaluate the derivatives
  con->evalFailureStrainSens(elemIndex, pt, X, e, dfde);

  TacsScalar failure = con->evalFailure(elemIndex, pt, X, e);

  // Compute the approximate derivative
  for (int i = 0; i < nstress; i++) {
    TacsScalar et = e[i];
#ifdef TACS_USE_COMPLEX
    e[i] = et + TacsScalar(0.0, dh);
#else
    e[i] = et + dh;
#endif

    TacsScalar failure_forward = con->evalFailure(elemIndex, pt, X, e);
#ifdef TACS_USE_COMPLEX
    dfde_approx[i] = TacsImagPart(failure_forward) / dh;
#else
    dfde_approx[i] = (failure_forward - failure) / dh;
#endif
    e[i] = et;
  }

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfde, dfde_approx, nstress, &max_err_index);
  double max_rel =
      TacsGetMaxRelError(dfde, dfde_approx, nstress, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr,
            "Testing the derivative of the failure index w.r.t. strain for "
            "object %s\n",
            con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(failure)/de", dfde, dfde_approx,
                             nstress);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfde, dfde_approx, nstress, test_fail_atol,
                                     test_fail_rtol);

  delete[] e;
  delete[] dfde;
  delete[] dfde_approx;

  return !allClose;
}

/*
  Test the derivative of the inner product of the adjoint vector and
  the residual with respect to material design variables.
*/
int TacsTestConstitutive(TACSConstitutive *con, int elemIndex, double dh,
                         int test_print_level, double test_fail_atol,
                         double test_fail_rtol) {
  // Retrieve the number of variables
  int ndvs = con->getDesignVarNums(elemIndex, 0, NULL);
  ndvs *= con->getDesignVarsPerNode();

  // Test the sensitivities w.r.t. the design variables first...
  TacsScalar *dvs = new TacsScalar[ndvs];
  con->getDesignVars(elemIndex, ndvs, dvs);

  srand(time(NULL));

  double pt[3];
  TacsGenerateRandomArray(pt, 3);

  if (test_print_level > 0) {
    fprintf(stderr,
            "Testing derivatives for the constitutive object %s at "
            "parametric point: \npt[0]  %15.6e\npt[1]  %15.6e\npt[2]"
            "  %15.6e\n",
            con->getObjectName(), pt[0], pt[1], pt[2]);
  }

  TacsScalar X[3];
  TacsGenerateRandomArray(X, 3);

  int flag, fail = 0;
  flag = TacsTestConstitutiveDensity(con, elemIndex, pt, X, ndvs, dvs, dh,
                                     test_print_level, test_fail_atol,
                                     test_fail_rtol);
  fail = flag || fail;

  flag = TacsTestConstitutiveSpecificHeat(con, elemIndex, pt, X, ndvs, dvs, dh,
                                          test_print_level, test_fail_atol,
                                          test_fail_rtol);
  fail = flag || fail;

  flag = TacsTestConstitutiveHeatFlux(con, elemIndex, pt, X, ndvs, dvs, dh,
                                      test_print_level, test_fail_atol,
                                      test_fail_rtol);
  fail = flag || fail;

  flag = TacsTestConstitutiveStress(con, elemIndex, pt, X, ndvs, dvs, dh,
                                    test_print_level, test_fail_atol,
                                    test_fail_rtol);
  fail = flag || fail;

  flag = TacsTestConstitutiveThermalStrain(con, elemIndex, pt, X, ndvs, dvs, dh,
                                           test_print_level, test_fail_atol,
                                           test_fail_rtol);
  fail = flag || fail;

  flag = TacsTestConstitutiveFailure(con, elemIndex, pt, X, ndvs, dvs, dh,
                                     test_print_level, test_fail_atol,
                                     test_fail_rtol);
  fail = flag || fail;

  flag = TacsTestConstitutiveFailureStrainSens(con, elemIndex, pt, X, dh,
                                               test_print_level, test_fail_atol,
                                               test_fail_rtol);
  fail = flag || fail;

  return fail;
}

/*
  Compare the approximate temperature derivative, formed from the
  unperturbed values f0 and the perturbed values f1, with the analytic
  derivative exact.
*/
static int TacsTestMaterialTempSensComponent(
    const char *descript, int size, const TacsScalar *exact,
    const TacsScalar *f0, const TacsScalar *f1, double dh, int test_print_level,
    double test_fail_atol, double test_fail_rtol) {
  TacsScalar *approx = new TacsScalar[size];

  // Compute the approximate derivative
  for (int i = 0; i < size; i++) {
#ifdef TACS_USE_COMPLEX
    approx[i] = TacsImagPart(f1[i]) / dh;
#else
    approx[i] = (f1[i] - f0[i]) / dh;
#endif
  }

  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError((TacsScalar *)exact, approx, size,
                                   &max_err_index);
  double max_rel = TacsGetMaxRelError((TacsScalar *)exact, approx, size,
                                      &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr, "Testing the temperature derivative of %s\n", descript);
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, "d(value)/d(theta)", (TacsScalar *)exact,
                             approx, size);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose((TacsScalar *)exact, approx, size,
                                     test_fail_atol, test_fail_rtol);

  delete[] approx;

  return !allClose;
}

int TacsTestMaterialPropertiesTempSens(TACSMaterialProperties *props,
                                       TacsScalar theta, double dh,
                                       int test_print_level,
                                       double test_fail_atol,
                                       double test_fail_rtol) {
  // The perturbed temperature change used for the approximate derivatives
#ifdef TACS_USE_COMPLEX
  TacsScalar theta_pert = theta + TacsScalar(0.0, dh);
#else
  TacsScalar theta_pert = theta + dh;
#endif

  int flag, fail = 0;

  // Specific heat
  TacsScalar c0, c1, dc;
  c0 = props->getSpecificHeat(theta);
  c1 = props->getSpecificHeat(theta_pert);
  dc = props->getSpecificHeatTempSens(theta);
  flag = TacsTestMaterialTempSensComponent(
      "the specific heat", 1, &dc, &c0, &c1, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Tangent stiffness matrix in 3D
  TacsScalar C03[21], C13[21], dC3[21];
  props->evalTangentStiffness3D(theta, C03);
  props->evalTangentStiffness3D(theta_pert, C13);
  props->evalTangentStiffness3DTempSens(theta, dC3);
  flag = TacsTestMaterialTempSensComponent(
      "the 3D tangent stiffness", 21, dC3, C03, C13, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Tangent stiffness matrix in 2D
  TacsScalar C02[6], C12[6], dC2[6];
  props->evalTangentStiffness2D(theta, C02);
  props->evalTangentStiffness2D(theta_pert, C12);
  props->evalTangentStiffness2DTempSens(theta, dC2);
  flag = TacsTestMaterialTempSensComponent(
      "the 2D tangent stiffness", 6, dC2, C02, C12, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Tangent thermal conductivity matrix in 3D
  TacsScalar Kc03[6], Kc13[6], dKc3[6];
  props->evalTangentHeatFlux3D(theta, Kc03);
  props->evalTangentHeatFlux3D(theta_pert, Kc13);
  props->evalTangentHeatFlux3DTempSens(theta, dKc3);
  flag = TacsTestMaterialTempSensComponent(
      "the 3D tangent heat flux", 6, dKc3, Kc03, Kc13, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Tangent thermal conductivity matrix in 2D
  TacsScalar Kc02[3], Kc12[3], dKc2[3];
  props->evalTangentHeatFlux2D(theta, Kc02);
  props->evalTangentHeatFlux2D(theta_pert, Kc12);
  props->evalTangentHeatFlux2DTempSens(theta, dKc2);
  flag = TacsTestMaterialTempSensComponent(
      "the 2D tangent heat flux", 3, dKc2, Kc02, Kc12, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Thermal strain in 3D
  TacsScalar e03[6], e13[6], de3[6];
  props->evalThermalStrain3D(theta, e03);
  props->evalThermalStrain3D(theta_pert, e13);
  props->evalThermalStrain3DTempSens(theta, de3);
  flag = TacsTestMaterialTempSensComponent(
      "the 3D thermal strain", 6, de3, e03, e13, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Thermal strain in 2D
  TacsScalar e02[3], e12[3], de2[3];
  props->evalThermalStrain2D(theta, e02);
  props->evalThermalStrain2D(theta_pert, e12);
  props->evalThermalStrain2DTempSens(theta, de2);
  flag = TacsTestMaterialTempSensComponent(
      "the 2D thermal strain", 3, de2, e02, e12, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Each temperature-dependent property, indexed by TemperatureDependentProperty
  static const char *prop_names[TACS_NUM_TEMP_DEP_PROPERTIES] = {
      "Young's modulus E", "Poisson's ratio nu", "the thermal expansion alpha",
      "the thermal conductivity kappa", "the specific heat property table"};
  for (int prop = 0; prop < TACS_NUM_TEMP_DEP_PROPERTIES; prop++) {
    TemperatureDependentProperty tprop = (TemperatureDependentProperty)prop;
    TacsScalar v0, v1, dv;
    v0 = props->evalTemperatureDependentProperty(tprop, theta, &dv);
    v1 = props->evalTemperatureDependentProperty(tprop, theta_pert);
    flag = TacsTestMaterialTempSensComponent(
        prop_names[prop], 1, &dv, &v0, &v1, dh, test_print_level,
        test_fail_atol, test_fail_rtol);
    fail = flag || fail;
  }

  return fail;
}

/*
  Compare the analytic design variable sensitivity of a temperature-dependent
  quantity with the approximate sensitivity formed by perturbing the design
  variables.
*/
static int TacsTestTempSensDVComponent(TACSConstitutive *con,
                                       const char *descript, const char *label,
                                       int ndvs, TacsScalar *dfdx,
                                       TacsScalar *dfdx_approx,
                                       int test_print_level,
                                       double test_fail_atol,
                                       double test_fail_rtol) {
  // Compute the error
  int max_err_index, max_rel_index;
  double max_err = TacsGetMaxError(dfdx, dfdx_approx, ndvs, &max_err_index);
  double max_rel = TacsGetMaxRelError(dfdx, dfdx_approx, ndvs, &max_rel_index);

  if (test_print_level > 0) {
    fprintf(stderr,
            "Testing the temperature-dependent %s DV sensitivity for %s\n",
            descript, con->getObjectName());
    fprintf(stderr, "Max Err: %10.4e in component %d.\n", max_err,
            max_err_index);
    fprintf(stderr, "Max REr: %10.4e in component %d.\n", max_rel,
            max_rel_index);
  }
  // Print the error if required
  if (test_print_level > 1) {
    TacsPrintErrorComponents(stderr, label, dfdx, dfdx_approx, ndvs);
  }
  if (test_print_level) {
    fprintf(stderr, "\n");
  }

  bool allClose = TacsAssertAllClose(dfdx, dfdx_approx, ndvs, test_fail_atol,
                                     test_fail_rtol);

  return !allClose;
}

/*
  Update the largest absolute difference between two arrays of values
*/
static void TacsUpdateMaxError(int size, TacsScalar *a, TacsScalar *b,
                               double *max_err) {
  int max_index;
  double err = TacsGetMaxError(a, b, size, &max_index);
  if (err > *max_err) {
    *max_err = err;
  }
}

int TacsTestConstitutiveTempSens(TACSConstitutive *con, int elemIndex,
                                 const double pt[], const TacsScalar X[],
                                 int ndvs, const TacsScalar *dvs,
                                 TacsScalar theta, double dh,
                                 int test_print_level, double test_fail_atol,
                                 double test_fail_rtol) {
  con->setDesignVars(elemIndex, ndvs, dvs);

  // The perturbed temperature change used for the approximate derivatives
#ifdef TACS_USE_COMPLEX
  TacsScalar theta_pert = theta + TacsScalar(0.0, dh);
#else
  TacsScalar theta_pert = theta + dh;
#endif

  // Generate random strains, thermal gradients and multiplier vectors
  int nstress = con->getNumStresses();
  TacsScalar e[9], psi_s[9], grad[3], psi_q[3];
  memset(e, 0, sizeof(e));
  memset(psi_s, 0, sizeof(psi_s));
  memset(grad, 0, sizeof(grad));
  memset(psi_q, 0, sizeof(psi_q));
  TacsGenerateRandomArray(e, nstress, -1e-3, 1e-3);
  TacsGenerateRandomArray(psi_s, nstress);
  TacsGenerateRandomArray(grad, 3);
  TacsGenerateRandomArray(psi_q, 3);

  int flag, fail = 0;

  // Derivative of the stress w.r.t. theta at fixed strain
  TacsScalar s0[9], s1[9], ds[9];
  memset(s0, 0, sizeof(s0));
  memset(s1, 0, sizeof(s1));
  memset(ds, 0, sizeof(ds));
  con->evalStress(elemIndex, pt, X, theta, e, s0);
  con->evalStress(elemIndex, pt, X, theta_pert, e, s1);
  con->addStressTempSens(elemIndex, 1.0, pt, X, theta, e, ds);
  flag = TacsTestMaterialTempSensComponent(
      "the stress", nstress, ds, s0, s1, dh, test_print_level, test_fail_atol,
      test_fail_rtol);
  fail = flag || fail;

  // Derivative of the thermal strain w.r.t. theta
  TacsScalar et0[9], et1[9], det[9];
  memset(et0, 0, sizeof(et0));
  memset(et1, 0, sizeof(et1));
  memset(det, 0, sizeof(det));
  con->evalThermalStrain(elemIndex, pt, X, theta, et0);
  con->evalThermalStrain(elemIndex, pt, X, theta_pert, et1);
  con->evalThermalStrainTempSens(elemIndex, pt, X, theta, det);
  flag = TacsTestMaterialTempSensComponent(
      "the thermal strain", nstress, det, et0, et1, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Derivative of the heat flux w.r.t. theta at fixed thermal gradient
  TacsScalar q0[3], q1[3], dq[3];
  memset(q0, 0, sizeof(q0));
  memset(q1, 0, sizeof(q1));
  memset(dq, 0, sizeof(dq));
  con->evalHeatFlux(elemIndex, pt, X, theta, grad, q0);
  con->evalHeatFlux(elemIndex, pt, X, theta_pert, grad, q1);
  con->addHeatFluxTempSens(elemIndex, 1.0, pt, X, theta, grad, dq);
  flag = TacsTestMaterialTempSensComponent(
      "the heat flux", 3, dq, q0, q1, dh, test_print_level, test_fail_atol,
      test_fail_rtol);
  fail = flag || fail;

  // Derivative of the specific heat w.r.t. theta
  TacsScalar c0, c1, dc;
  c0 = con->evalSpecificHeat(elemIndex, pt, X, theta);
  c1 = con->evalSpecificHeat(elemIndex, pt, X, theta_pert);
  dc = con->evalSpecificHeatTempSens(elemIndex, pt, X, theta);
  flag = TacsTestMaterialTempSensComponent(
      "the specific heat", 1, &dc, &c0, &c1, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Derivative of the failure index w.r.t. theta at fixed strain
  TacsScalar f0, f1, df;
  f0 = con->evalFailure(elemIndex, pt, X, theta, e);
  f1 = con->evalFailure(elemIndex, pt, X, theta_pert, e);
  df = con->evalFailureTempSens(elemIndex, pt, X, theta, e);
  flag = TacsTestMaterialTempSensComponent(
      "the failure index", 1, &df, &f0, &f1, dh, test_print_level,
      test_fail_atol, test_fail_rtol);
  fail = flag || fail;

  // Design variable sensitivities of the temperature-dependent evaluations
  if (ndvs > 0) {
    TacsScalar *xtemp = new TacsScalar[ndvs];
    TacsScalar *dfdx = new TacsScalar[ndvs];
    TacsScalar *dfdx_approx = new TacsScalar[ndvs];

    // Copy the design variable values
    memcpy(xtemp, dvs, ndvs * sizeof(TacsScalar));

    // Generate random arrays for the scale factors
    TacsScalar scale[3];
    TacsGenerateRandomArray(scale, 3);

    // Stress: f = scale * psi_s^{T} stress(theta, e)
    memset(dfdx, 0, ndvs * sizeof(TacsScalar));
    con->addStressDVSens(elemIndex, scale[0], pt, X, theta, e, psi_s, ndvs,
                         dfdx);

    TacsScalar prod_s = 0.0;
    for (int j = 0; j < nstress; j++) {
      prod_s += psi_s[j] * s0[j];
    }

    for (int i = 0; i < ndvs; i++) {
      TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
      xtemp[i] = x + TacsScalar(0.0, dh);
#else
      xtemp[i] = x + dh;
#endif
      con->setDesignVars(elemIndex, ndvs, xtemp);

      memset(s1, 0, sizeof(s1));
      con->evalStress(elemIndex, pt, X, theta, e, s1);
      TacsScalar prod_s_forward = 0.0;
      for (int j = 0; j < nstress; j++) {
        prod_s_forward += psi_s[j] * s1[j];
      }
#ifdef TACS_USE_COMPLEX
      dfdx_approx[i] = scale[0] * TacsImagPart(prod_s_forward) / dh;
#else
      dfdx_approx[i] = scale[0] * (prod_s_forward - prod_s) / dh;
#endif
      xtemp[i] = x;
    }

    // Reset the design variable values
    con->setDesignVars(elemIndex, ndvs, dvs);

    flag = TacsTestTempSensDVComponent(
        con, "stress", "d(psi^{T}s)/dx", ndvs, dfdx, dfdx_approx,
        test_print_level, test_fail_atol, test_fail_rtol);
    fail = flag || fail;

    // Heat flux: f = scale * psi_q^{T} flux(theta, grad)
    memset(dfdx, 0, ndvs * sizeof(TacsScalar));
    con->addHeatFluxDVSens(elemIndex, scale[1], pt, X, theta, grad, psi_q,
                           ndvs, dfdx);

    TacsScalar prod_q = 0.0;
    for (int j = 0; j < 3; j++) {
      prod_q += psi_q[j] * q0[j];
    }

    for (int i = 0; i < ndvs; i++) {
      TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
      xtemp[i] = x + TacsScalar(0.0, dh);
#else
      xtemp[i] = x + dh;
#endif
      con->setDesignVars(elemIndex, ndvs, xtemp);

      memset(q1, 0, sizeof(q1));
      con->evalHeatFlux(elemIndex, pt, X, theta, grad, q1);
      TacsScalar prod_q_forward = 0.0;
      for (int j = 0; j < 3; j++) {
        prod_q_forward += psi_q[j] * q1[j];
      }
#ifdef TACS_USE_COMPLEX
      dfdx_approx[i] = scale[1] * TacsImagPart(prod_q_forward) / dh;
#else
      dfdx_approx[i] = scale[1] * (prod_q_forward - prod_q) / dh;
#endif
      xtemp[i] = x;
    }

    // Reset the design variable values
    con->setDesignVars(elemIndex, ndvs, dvs);

    flag = TacsTestTempSensDVComponent(
        con, "heat flux", "d(psi^{T}q)/dx", ndvs, dfdx, dfdx_approx,
        test_print_level, test_fail_atol, test_fail_rtol);
    fail = flag || fail;

    // Specific heat: f = scale * c(theta)
    memset(dfdx, 0, ndvs * sizeof(TacsScalar));
    con->addSpecificHeatDVSens(elemIndex, scale[2], pt, X, theta, ndvs, dfdx);

    for (int i = 0; i < ndvs; i++) {
      TacsScalar x = xtemp[i];
#ifdef TACS_USE_COMPLEX
      xtemp[i] = x + TacsScalar(0.0, dh);
#else
      xtemp[i] = x + dh;
#endif
      con->setDesignVars(elemIndex, ndvs, xtemp);

      TacsScalar c_forward = con->evalSpecificHeat(elemIndex, pt, X, theta);
#ifdef TACS_USE_COMPLEX
      dfdx_approx[i] = scale[2] * TacsImagPart(c_forward) / dh;
#else
      dfdx_approx[i] = scale[2] * (c_forward - c0) / dh;
#endif
      xtemp[i] = x;
    }

    // Reset the design variable values
    con->setDesignVars(elemIndex, ndvs, dvs);

    flag = TacsTestTempSensDVComponent(con, "specific heat", "d(c)/dx", ndvs,
                                       dfdx, dfdx_approx, test_print_level,
                                       test_fail_atol, test_fail_rtol);
    fail = flag || fail;

    delete[] xtemp;
    delete[] dfdx;
    delete[] dfdx_approx;
  }

  // Check that the temperature-dependent evaluations reproduce the
  // temperature-independent ones when the properties do not depend on theta
  if (!con->isTemperatureDependent()) {
    double cons_err = 0.0;

    // Stress
    TacsScalar s_ti[9];
    memset(s_ti, 0, sizeof(s_ti));
    con->evalStress(elemIndex, pt, X, e, s_ti);
    TacsUpdateMaxError(nstress, s_ti, s0, &cons_err);

    // Tangent stiffness
    TacsScalar C_ti[45], C_td[45];
    memset(C_ti, 0, sizeof(C_ti));
    memset(C_td, 0, sizeof(C_td));
    con->evalTangentStiffness(elemIndex, pt, X, C_ti);
    con->evalTangentStiffness(elemIndex, pt, X, theta, C_td);
    TacsUpdateMaxError(nstress * (nstress + 1) / 2, C_ti, C_td, &cons_err);

    // Heat flux
    TacsScalar q_ti[3];
    memset(q_ti, 0, sizeof(q_ti));
    con->evalHeatFlux(elemIndex, pt, X, grad, q_ti);
    TacsUpdateMaxError(3, q_ti, q0, &cons_err);

    // Tangent heat flux
    TacsScalar Kc_ti[6], Kc_td[6];
    memset(Kc_ti, 0, sizeof(Kc_ti));
    memset(Kc_td, 0, sizeof(Kc_td));
    con->evalTangentHeatFlux(elemIndex, pt, X, Kc_ti);
    con->evalTangentHeatFlux(elemIndex, pt, X, theta, Kc_td);
    TacsUpdateMaxError(6, Kc_ti, Kc_td, &cons_err);

    // Specific heat
    TacsScalar c_ti = con->evalSpecificHeat(elemIndex, pt, X);
    TacsUpdateMaxError(1, &c_ti, &c0, &cons_err);

    // Failure index
    TacsScalar f_ti = con->evalFailure(elemIndex, pt, X, e);
    TacsUpdateMaxError(1, &f_ti, &f0, &cons_err);

    // Thermal strain derivative, which is the thermal strain at unit theta
    TacsScalar et_unit[9];
    memset(et_unit, 0, sizeof(et_unit));
    con->evalThermalStrain(elemIndex, pt, X, 1.0, et_unit);
    TacsUpdateMaxError(nstress, et_unit, det, &cons_err);

    if (test_print_level > 0) {
      fprintf(stderr, "Testing temperature-independent consistency for %s\n",
              con->getObjectName());
      fprintf(stderr, "Max Err: %10.4e\n", cons_err);
    }
    if (test_print_level) {
      fprintf(stderr, "\n");
    }

    if (cons_err > 0.0) {
      fail = 1;
    }
  }

  // Reset the design variable values
  con->setDesignVars(elemIndex, ndvs, dvs);

  return fail;
}
