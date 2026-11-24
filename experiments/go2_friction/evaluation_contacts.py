"""Extract the force of the ground on each foot from native MuJoCo contacts."""

import mujoco
import numpy as np


class FootGroundContacts:
    def __init__(self, env):
        self.backend = env._backend
        model = self.backend._model_for_env(0)
        native_feet = self.backend._canonical_to_native_body_np[
            env.feet_indices.numpy()
        ]
        self.foot_slots = {int(body): slot for slot, body in enumerate(native_feet)}
        self.ground = (model.geom_bodyid == 0) & (
            model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        )
        if not self.ground.any():
            raise ValueError("Go2 friction evaluation requires a world ground plane")

    def read(self):
        shape = (len(self.backend._datas), len(self.foot_slots))
        active = np.zeros(shape, dtype=bool)
        forces = np.zeros((*shape, 3), dtype=np.float64)
        wrench = np.zeros(6, dtype=np.float64)
        for world, data in enumerate(self.backend._datas):
            model = self.backend._model_for_env(world)
            for index, contact in enumerate(data.contact):
                if contact.efc_address < 0:
                    continue
                first, second = contact.geom
                if self.ground[first]:
                    foot_geom, sign = second, 1
                elif self.ground[second]:
                    foot_geom, sign = first, -1
                else:
                    continue
                foot = self.foot_slots.get(model.geom_bodyid[foot_geom])
                if foot is None:
                    continue

                mujoco.mj_contactForce(model, data, index, wrench)
                # The rows of frame are world-space contact axes. The wrench
                # acts on geom 2, so reverse its sign when the foot is geom 1.
                force_world = contact.frame.reshape(3, 3).T @ wrench[:3]
                forces[world, foot] += sign * force_world
                active[world, foot] = True
        return active, forces
