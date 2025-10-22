import mujoco
import mujoco.viewer
import numpy as np
from mujoco_playground._src.locomotion.go2 import go2_constants
from mujoco_playground._src.locomotion.go2.base import get_assets

# Load your model
model = mujoco.MjModel.from_xml_path(
      go2_constants.FULL_COLLISIONS_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
data = mujoco.MjData(model)

# Get the geom id for the one you want to track
geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "test_2")

# Viewer loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Step the simulation
        mujoco.mj_step(model, data)

        # --- Add a debug sphere at the geom location ---
        # Get world position of the geom
        pos = data.geom_xpos[geom_id].copy()
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.10, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=0.5*np.array([1, 0, 0, 1])
    )
        viewer.user_scn.ngeom = 1
        # Render
        viewer.sync()