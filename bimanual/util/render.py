import mujoco as mj
import numpy as np


class SimpleRenderer:
    def __init__(self, model, data, height: int = 720, width: int = 720) -> None:
        self.mj_renderer = mj.Renderer(model, height=height, width=width)
        self.model = model
        self.data = data

        self.scn = self.mj_renderer.scene

        self._markers = []

    def render(self):
        raise NotImplementedError(
            "use read_pixels for offscreen rendering or initialize robot with window=True"
        )

    def read_pixels(self, camid, depth: bool = False):
        if depth:
            raise NotImplementedError("Depth rendering not implemented")
        self.mj_renderer.update_scene(self.data, camid)

        for marker in self._markers:
            self._add_marker_to_scene(marker)

        retval = self.mj_renderer.render()

        self._markers[:] = []

        return self.mj_renderer.render()

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mj.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mj.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mj.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mj._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

        return
