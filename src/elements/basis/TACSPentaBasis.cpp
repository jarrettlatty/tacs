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

#include "TACSPentaBasis.h"

#include "TACSGaussQuadrature.h"
#include "TACSTriangleQuadrature.h"

/*
  Linear Penta (triangular prism/wedge) basis class functions

  Parametric coordinates: pt = (u, v, w).

  u, v are the triangular in-plane coordinates with
  L0 = 1 - u - v, L1 = u, L2 = v (u, v >= 0, u + v <= 1), and w in [-1, 1]
  is the extrusion direction.

  Node ordering (matches Nastran CPENTA6 and VTK_WEDGE):
    node 0: (u,v,w) = (0, 0, -1)      node 3: (u,v,w) = (0, 0, 1)
    node 1: (u,v,w) = (1, 0, -1)      node 4: (u,v,w) = (1, 0, 1)
    node 2: (u,v,w) = (0, 1, -1)      node 5: (u,v,w) = (0, 1, 1)

  Faces:
    face 0: bottom triangle, w = -1, nodes (0, 1, 2)
    face 1: top triangle,    w = 1,  nodes (3, 4, 5)
    face 2: quad, v = 0,       nodes (0, 1, 3, 4)
    face 3: quad, u = 0,       nodes (0, 2, 3, 5)
    face 4: quad, u + v = 1,   nodes (1, 2, 4, 5)
*/

ElementLayout TACSLinearPentaBasis::getLayoutType() { return TACS_PENTA_ELEMENT; }

void TACSLinearPentaBasis::getVisPoint(int n, double pt[]) {
  if (n == 0) {
    pt[0] = 0.0;
    pt[1] = 0.0;
    pt[2] = -1.0;
  } else if (n == 1) {
    pt[0] = 1.0;
    pt[1] = 0.0;
    pt[2] = -1.0;
  } else if (n == 2) {
    pt[0] = 0.0;
    pt[1] = 1.0;
    pt[2] = -1.0;
  } else if (n == 3) {
    pt[0] = 0.0;
    pt[1] = 0.0;
    pt[2] = 1.0;
  } else if (n == 4) {
    pt[0] = 1.0;
    pt[1] = 0.0;
    pt[2] = 1.0;
  } else if (n == 5) {
    pt[0] = 0.0;
    pt[1] = 1.0;
    pt[2] = 1.0;
  }
}

int TACSLinearPentaBasis::getNumNodes() { return 6; }

int TACSLinearPentaBasis::getNumParameters() { return 3; }

int TACSLinearPentaBasis::getNumQuadraturePoints() { return 6; }

double TACSLinearPentaBasis::getQuadratureWeight(int n) {
  int i = n % 3;
  int j = n / 3;
  return TacsTriangleWts3[i] * TacsGaussQuadWts2[j];
}

double TACSLinearPentaBasis::getQuadraturePoint(int n, double pt[]) {
  int i = n % 3;
  int j = n / 3;

  pt[0] = TacsTrianglePts3[2 * i];
  pt[1] = TacsTrianglePts3[2 * i + 1];
  pt[2] = TacsGaussQuadPts2[j];

  return TacsTriangleWts3[i] * TacsGaussQuadWts2[j];
}

int TACSLinearPentaBasis::getNumElementFaces() { return 5; }

int TACSLinearPentaBasis::getNumFaceQuadraturePoints(int face) {
  if (face == 0 || face == 1) {
    return 3;
  }
  return 4;
}

double TACSLinearPentaBasis::getFaceQuadraturePoint(int face, int n,
                                                    double pt[], double t[]) {
  if (face == 0) {
    // Bottom triangle, w = -1, nodes (0, 1, 2).
    // Use the (u, v) triangle quadrature points directly. The tangent
    // order (v-dir, u-dir) is chosen so that cross(t1, t2) = -w, which is
    // the outward direction for this face.
    pt[0] = TacsTrianglePts3[2 * n];
    pt[1] = TacsTrianglePts3[2 * n + 1];
    pt[2] = -1.0;

    t[0] = 0.0;
    t[1] = 1.0;
    t[2] = 0.0;
    t[3] = 1.0;
    t[4] = 0.0;
    t[5] = 0.0;

    return TacsTriangleWts3[n];
  } else if (face == 1) {
    // Top triangle, w = 1, nodes (3, 4, 5).
    // cross(u-dir, v-dir) = +w, the outward direction for this face.
    pt[0] = TacsTrianglePts3[2 * n];
    pt[1] = TacsTrianglePts3[2 * n + 1];
    pt[2] = 1.0;

    t[0] = 1.0;
    t[1] = 0.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 1.0;
    t[5] = 0.0;

    return TacsTriangleWts3[n];
  } else if (face == 2) {
    // Side quad, v = 0, nodes (0, 1, 3, 4).
    // Parametrized by the native Gauss variables xi (-> u) and w.
    int a = n % 2;
    int b = n / 2;

    double u = 0.5 * (TacsGaussQuadPts2[a] + 1.0);

    pt[0] = u;
    pt[1] = 0.0;
    pt[2] = TacsGaussQuadPts2[b];

    // d(pt)/d(xi_a) = (0.5, 0, 0), d(pt)/d(xi_b) = (0, 0, 1).
    // cross(t1, t2) = (0, -0.5, 0), the outward (-v) direction.
    t[0] = 0.5;
    t[1] = 0.0;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 1.0;

    return TacsGaussQuadWts2[a] * TacsGaussQuadWts2[b];
  } else if (face == 3) {
    // Side quad, u = 0, nodes (0, 2, 3, 5).
    int a = n % 2;
    int b = n / 2;

    double v = 0.5 * (TacsGaussQuadPts2[a] + 1.0);

    pt[0] = 0.0;
    pt[1] = v;
    pt[2] = TacsGaussQuadPts2[b];

    // d(pt)/d(xi_b) = (0, 0, 1), d(pt)/d(xi_a) = (0, 0.5, 0).
    // cross(t1, t2) = (-0.5, 0, 0), the outward (-u) direction.
    t[0] = 0.0;
    t[1] = 0.0;
    t[2] = 1.0;
    t[3] = 0.0;
    t[4] = 0.5;
    t[5] = 0.0;

    return TacsGaussQuadWts2[a] * TacsGaussQuadWts2[b];
  } else {
    // Side quad, u + v = 1 (hypotenuse), nodes (1, 2, 4, 5).
    int a = n % 2;
    int b = n / 2;

    double eta = 0.5 * (TacsGaussQuadPts2[a] + 1.0);

    pt[0] = 1.0 - eta;
    pt[1] = eta;
    pt[2] = TacsGaussQuadPts2[b];

    // d(pt)/d(xi_a) = (-0.5, 0.5, 0), d(pt)/d(xi_b) = (0, 0, 1).
    // cross(t1, t2) = (0.5, 0.5, 0), the outward (+u+v) direction.
    t[0] = -0.5;
    t[1] = 0.5;
    t[2] = 0.0;
    t[3] = 0.0;
    t[4] = 0.0;
    t[5] = 1.0;

    return TacsGaussQuadWts2[a] * TacsGaussQuadWts2[b];
  }
}

void TACSLinearPentaBasis::computeBasis(const double pt[], double N[]) {
  double L0 = 1.0 - pt[0] - pt[1];
  double L1 = pt[0];
  double L2 = pt[1];

  double Nb = 0.5 * (1.0 - pt[2]);
  double Nt = 0.5 * (1.0 + pt[2]);

  N[0] = L0 * Nb;
  N[1] = L1 * Nb;
  N[2] = L2 * Nb;
  N[3] = L0 * Nt;
  N[4] = L1 * Nt;
  N[5] = L2 * Nt;
}

void TACSLinearPentaBasis::computeBasisGradient(const double pt[], double N[],
                                                double Nxi[]) {
  double L0 = 1.0 - pt[0] - pt[1];
  double L1 = pt[0];
  double L2 = pt[1];

  double Nb = 0.5 * (1.0 - pt[2]);
  double Nt = 0.5 * (1.0 + pt[2]);

  N[0] = L0 * Nb;
  N[1] = L1 * Nb;
  N[2] = L2 * Nb;
  N[3] = L0 * Nt;
  N[4] = L1 * Nt;
  N[5] = L2 * Nt;

  // Node 0: N0 = L0*Nb
  Nxi[0] = -Nb;
  Nxi[1] = -Nb;
  Nxi[2] = -0.5 * L0;

  // Node 1: N1 = L1*Nb
  Nxi[3] = Nb;
  Nxi[4] = 0.0;
  Nxi[5] = -0.5 * L1;

  // Node 2: N2 = L2*Nb
  Nxi[6] = 0.0;
  Nxi[7] = Nb;
  Nxi[8] = -0.5 * L2;

  // Node 3: N3 = L0*Nt
  Nxi[9] = -Nt;
  Nxi[10] = -Nt;
  Nxi[11] = 0.5 * L0;

  // Node 4: N4 = L1*Nt
  Nxi[12] = Nt;
  Nxi[13] = 0.0;
  Nxi[14] = 0.5 * L1;

  // Node 5: N5 = L2*Nt
  Nxi[15] = 0.0;
  Nxi[16] = Nt;
  Nxi[17] = 0.5 * L2;
}
