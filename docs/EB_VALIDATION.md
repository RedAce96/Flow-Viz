# General 2-D embedded-boundary validation

`SurfaceCurve2D` stores ordered SI coordinates, arc length, node and segment tangents, fluid-facing normals, open/closed status, component/side identity, source, confidence, and diagnostics. Duplicate points, self-intersections, unresolved open-curve sides, short contours, and unresolved volume-fraction normals are rejected.

Manufactured tests cover uniform pressure on a closed body, linear pressure resultant and moment, linear tangential velocity/shear, linear temperature/heat flux, analytical wedge panels, contour reversal, and exact equivalence with the reviewed flat-plate force integrator. Volume-fraction contours retain unsmoothed coordinates and smoothing displacement diagnostics.

General results are designated `validated_2d_eb_v1`, not certified. External benchmark evidence is required before changing that designation.
