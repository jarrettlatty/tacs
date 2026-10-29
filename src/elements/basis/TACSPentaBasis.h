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

#ifndef TACS_PENTA_BASIS_H
#define TACS_PENTA_BASIS_H

#include "TACSElementBasis.h"

/**
  Basis class for a linear triangular-prism (6-node wedge/CPENTA) element

  The parametric coordinates are pt = (u, v, w) where u, v are the
  triangular in-plane coordinates (with L0 = 1 - u - v, L1 = u, L2 = v,
  the same convention used by TACSLinearTriangleBasis and
  TACSLinearTetrahedralBasis) and w in [-1, 1] is the extrusion direction
  (the same convention used by TACSLinearHexaBasis).

  The node ordering matches the Nastran CPENTA6 convention (and VTK's
  wedge cell type): nodes 0, 1, 2 form the bottom triangle (w = -1) and
  nodes 3, 4, 5 form the top triangle (w = 1), with vertical edges
  (0, 3), (1, 4), (2, 5).
*/
class TACSLinearPentaBasis : public TACSElementBasis {
 public:
  ElementLayout getLayoutType();
  void getVisPoint(int n, double pt[]);
  int getNumNodes();
  int getNumParameters();
  int getNumQuadraturePoints();
  double getQuadratureWeight(int n);
  double getQuadraturePoint(int n, double pt[]);
  int getNumElementFaces();
  int getNumFaceQuadraturePoints(int face);
  double getFaceQuadraturePoint(int face, int n, double pt[], double t[]);
  void computeBasis(const double pt[], double N[]);
  void computeBasisGradient(const double pt[], double N[], double Nxi[]);
};

#endif  // TACS_PENTA_BASIS_H
