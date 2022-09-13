# binvox
```shell
Usage: binvox [-d <voxel dimension>] [-t <voxel file type>] [-c] [-v] <model filespec>
  -license: show software license
  -d: specify voxel grid size (default 256, max 1024)(no max when using -e)
  -t: specify voxel file type (default binvox, also supported: hips, mira, vtk, raw, schematic, msh)
  -c: z-buffer based carving method only
  -dc: dilated carving, stop carving 1 voxel before intersection
  -v: z-buffer based parity voting method only (default is both -c and -v)
  -e: exact voxelization (any voxel intersecting a convex polygon gets set)(does not use graphics card)
Additional parameters:
  -bb <minx> <miny> <minz> <maxx> <maxy> <maxz>: force a different input model bounding box
  -ri: remove internal voxels
  -cb: center model inside unit cube
  -rotx: rotate object 90 degrees ccw around x-axis before voxelizing
  -rotz: rotate object 90 degrees cw around z-axis before voxelizing
    both -rotx and -rotz can be used multiple times
  -aw: _also_ render the model in wireframe (helps with thin parts)
  -fit: only write the voxels in the voxel bounding box
  -bi <id>: when converting to schematic, use block ID <id>
  -mb: when converting using -e from .obj to schematic, parse block ID from material spec 'usemtl blockid_<id>' (ids 1-255 only)
  -pb: use offscreen pbuffer instead of onscreen window
  -down: downsample voxels by a factor of 2 in each dimension (can be used multiple times)
  -dmin <nr>: when downsampling, destination voxel is on if >= <nr> source voxels are (default 4)
Supported 3D model file formats:
  VRML V2.0: almost fully supported
  UG, OBJ, OFF, DXF, XGL, POV, BREP, PLY, JOT: only polygons supported
Example:
binvox -c -d 200 -t mira plane.wrl
```


# viewvox

```shell
Usage
viewvox   [-ki] <model filename>

  -ki: keep internal voxels (removed by default)

Mouse left button = rotate
      middle      = pan
      right       = zoom
Key   r           = reset view
      arrow keys  = move 1 voxel step along x (left, right) or y (up, down)
      =,-         = move 1 voxel step along z

      q           = quit

      a           = toggle alternating colours
      p           = toggle between orthographic and perspective projection
      x, y, z     = set camera looking down X, Y, or Z axis
      X, Y, Z     = set camera looking up X, Y, or Z axis
      1           = toggle show x, y, and z coordinates

      s           = show single slice
      n           = show both/above/below slice neighbour(s)
      t           = toggle neighbour transparency
      j           = move slice down
      k           = move slice up
      g           = toggle show grid at slice level

A lot of the key commands were added to make viewvox more useful when building voxel models in minecraft.http://www.patrickmin.com/minecraft

```

