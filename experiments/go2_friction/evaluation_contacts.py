"""Extract ground contact forces from native MuJoCo contacts."""

import mujoco
import numpy as np


class FootGroundContacts:
    def __init__(self, env, *, record_distal=False):
        self.backend = env._backend
        model = self.backend._model_for_env(0)
        native_feet = self.backend._canonical_to_native_body_np[
            env.feet_indices.numpy()
        ]
        self.foot_slots = {int(body): slot for slot, body in enumerate(native_feet)}
        self.load_slots = np.full(len(model.geom_bodyid), len(self.foot_slots))
        for body, slot in self.foot_slots.items():
            self.load_slots[model.geom_bodyid == body] = slot
        self.ground = (model.geom_bodyid == 0) & (
            model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        )
        if not self.ground.any():
            raise ValueError("Go2 friction evaluation requires a world ground plane")
        self._support_groups = None
        self.distal_slots = None
        if record_distal:
            linked_bodies = {
                model.body(body).name.removesuffix("_foot") + "_calflower1": slot
                for body, slot in self.foot_slots.items()
            }
            self.distal_slots = {}
            for geom in np.flatnonzero(
                model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER
            ):
                name = model.body(int(model.geom_bodyid[geom])).name
                if name in linked_bodies:
                    self.distal_slots[int(geom)] = linked_bodies[name]
            if sorted(self.distal_slots.values()) != list(range(len(self.foot_slots))):
                raise ValueError("Expected one distal calf cylinder per Go2 foot")

    def read(self, *, include_non_foot=False, split_support=False):
        include_non_foot = include_non_foot or split_support
        shape = (len(self.backend._datas), len(self.foot_slots))
        active = np.zeros(shape, dtype=bool)
        forces = np.zeros((*shape, 3), dtype=np.float64)
        non_foot_load = np.zeros(shape[0]) if include_non_foot else None
        if split_support:
            group_loads = np.zeros((shape[0], 2))
            if self._support_groups is None:
                model = self.backend._model_for_env(0)
                self._support_groups = {}
                for body in np.unique(model.geom_bodyid):
                    name = model.body(int(body)).name.lower()
                    if "calf" in name:
                        self._support_groups[body] = 0
                    elif name.startswith("head"):
                        self._support_groups[body] = 1
        record_distal = split_support and self.distal_slots is not None
        if record_distal:
            distal_active = np.zeros(shape, dtype=bool)
            distal_forces = np.zeros_like(forces)
            other_non_foot_load = np.zeros(shape[0])
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
                body = model.geom_bodyid[foot_geom]
                foot = self.foot_slots.get(body)
                if foot is None and not include_non_foot:
                    continue

                mujoco.mj_contactForce(model, data, index, wrench)
                # The rows of frame are world-space contact axes. The wrench
                # acts on geom 2, so reverse its sign when the foot is geom 1.
                force_world = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
                if foot is None:
                    load = max(0.0, force_world[2])
                    non_foot_load[world] += load
                    if split_support and body in self._support_groups:
                        group_loads[world, self._support_groups[body]] += load
                    if record_distal:
                        slot = self.distal_slots.get(foot_geom)
                        if slot is None:
                            other_non_foot_load[world] += load
                        else:
                            distal_forces[world, slot] += force_world
                            distal_active[world, slot] = True
                else:
                    forces[world, foot] += force_world
                    active[world, foot] = True
        if record_distal:
            return (
                active,
                forces,
                non_foot_load,
                group_loads,
                distal_active,
                distal_forces,
                other_non_foot_load,
            )
        if split_support:
            return active, forces, non_foot_load, group_loads
        if include_non_foot:
            return active, forces, non_foot_load
        return active, forces

    def read_vertical_loads(self):
        """Read world Z loads only and skip unrelated contacts in bulk."""
        num_feet = len(self.foot_slots)
        loads = np.zeros((len(self.backend._datas), num_feet + 1))
        wrench = np.zeros(6)
        for world, data in enumerate(self.backend._datas):
            if not data.ncon:
                continue
            contacts = data.contact
            geoms = contacts.geom
            first_ground = self.ground[geoms[:, 0]]
            active = contacts.efc_address >= 0
            active &= first_ground | self.ground[geoms[:, 1]]
            if not active.any():
                continue
            model = self.backend._model_for_env(world)
            vertical = contacts.frame[:, 2::3]
            for idx in np.flatnonzero(active):
                ground_first = first_ground[idx]
                slot = self.load_slots[geoms[idx, int(ground_first)]]
                mujoco.mj_contactForce(model, data, int(idx), wrench)
                load = (1 if ground_first else -1) * (vertical[idx] @ wrench[:3])
                loads[world, slot] += max(0.0, load) if slot == num_feet else load
        return loads[:, :num_feet], loads[:, num_feet]
