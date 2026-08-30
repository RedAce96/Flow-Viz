# Future three-dimensional extension interfaces

Inspection may identify a three-dimensional PeleC plotfile, but every current recipe declares `supported_dimensions: [2]`. Planning returns an `UNSUPPORTED_DIMENSION` blocker before numerical loading.

Future work should add dimension-specific implementations behind the existing dataset, geometry, workflow, and artifact capabilities:

- slices and projections with explicit coordinate frames;
- native-AMR three-component vorticity;
- triangulated EB surface meshes with orientation and component topology;
- pressure/viscous traction and heat flux on surface elements;
- force and moment with explicit area/span conventions;
- bounded volume rendering.

No placeholder 3-D figure or unvalidated 3-D load may be registered as a supported artifact.
