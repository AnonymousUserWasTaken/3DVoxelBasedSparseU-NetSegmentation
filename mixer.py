# mix.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import bpy

from tof_helpers import extract_points_from_object

@dataclass
class MixerParams:
    pmf_counts: Dict[int, float] = None           # objects per scene
    yaw_only: bool = True
    sensor_dropout: Tuple[float,float] = (0.08, 0.18)
    region_xy_m: Tuple[float,float] = (2.5, 2.5)  # half-extent
    min_separation_m: Tuple[float, float] = (0.6, 1.0)
    overhead_mode: str = "off"  # {"off","heads_only"}
    z_crop_top_cm: Tuple[float,float] = (20.0, 60.0)
    class_sampling_pmf: Optional[Dict[str, float]] = None
    seed: Optional[int] = None

class SceneMixer:
    """
    Multi-class mixer: selects (class, type, object) across ALL classes and places them in 2D with yaw.
    Profiles are "class::type".
    """
    def __init__(self,
                 class_names: List[str],
                 types_map: Dict[str, List[str]],
                 samples: List[Dict],
                 cls2id: Dict[str,int],
                 prof2id: Dict[str,int]):
        self.class_names = class_names
        self.types_map = types_map
        self.samples = samples
        self.cls2id = cls2id
        self.prof2id = prof2id

        self.by_class_type: Dict[str, Dict[str, List[str]]] = {}
        for s in samples:
            c = s["class"]; t = s["type"]; name = s["object_name"]
            self.by_class_type.setdefault(c, {}).setdefault(t, []).append(name)

        # default PMF proportional to sample counts
        from collections import Counter
        cls_counts = Counter([s["class"] for s in samples])
        total = sum(cls_counts.values()) or 1
        self.default_class_pmf = {c: cls_counts[c]/total for c in class_names}

    def _sample_from_pmf(self, pmf: Dict[int,float]) -> int:
        items = sorted(pmf.items(), key=lambda kv: kv[0])
        keys = [k for k,_ in items]
        probs = np.array([v for _,v in items], dtype=np.float64)
        probs /= probs.sum() if probs.sum()>0 else 1.0
        return int(np.random.choice(keys, p=probs))

    def _choose_class(self, pmf: Optional[Dict[str,float]]):
        P = pmf if pmf else self.default_class_pmf
        keys = list(P.keys()); probs = np.array([P[k] for k in keys], dtype=np.float64)
        probs = probs / probs.sum()
        return str(np.random.choice(keys, p=probs))

    def _poisson_disk_sample(self, K: int, min_sep: float, xy_half: Tuple[float,float], max_tries:int=2000):
        ax, ay = xy_half
        pts = []; tries = 0
        while len(pts) < K and tries < max_tries:
            p = np.array([np.random.uniform(-ax, ax), np.random.uniform(-ay, ay)], dtype=np.float32)
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < min_sep:
                    ok = False; break
            if ok: pts.append(p)
            tries += 1
        if len(pts) < K:
            for _ in range(K - len(pts)):
                pts.append(np.array([np.random.uniform(-ax, ax), np.random.uniform(-ay, ay)], dtype=np.float32))
        return np.stack(pts, axis=0)

    def _get_points(self, obj_name: str) -> np.ndarray:
        pts_t = extract_points_from_object(bpy.data.objects[obj_name])
        return pts_t.detach().cpu().numpy().astype(np.float32)

    def sample_scene(self, mp: MixerParams) -> Dict[str, np.ndarray]:
        if mp.seed is not None:
            np.random.seed(mp.seed); random.seed(mp.seed)

        K = self._sample_from_pmf(mp.pmf_counts)
        tz = 0.0

        min_sep = float(np.random.uniform(mp.min_separation_m[0], mp.min_separation_m[1]))
        positions_xy = self._poisson_disk_sample(K, min_sep, mp.region_xy_m)
        yaw_list = [np.random.uniform(0, 2*np.pi) for _ in range(K)]

        P_list, Lc_list, Lp_list, I_list = [], [], [], []
        instance_id = 1

        for k in range(K):
            cls = self._choose_class(mp.class_sampling_pmf)
            types = self.by_class_type.get(cls, {})
            if not types:
                continue
            t = random.choice(list(types.keys()))
            obj_name = random.choice(types[t])

            Pk = self._get_points(obj_name)
            if Pk.shape[0] == 0:
                continue

            c = Pk.mean(0, keepdims=True)
            Pk_local = Pk - c

            th = yaw_list[k]
            R = np.array([[np.cos(th), -np.sin(th), 0],
                          [np.sin(th),  np.cos(th), 0],
                          [0,           0,          1]], dtype=np.float32)
            Pk_rot = (R @ Pk_local.T).T
            tx, ty = positions_xy[k]
            Pk_world = Pk_rot + np.array([tx, ty, tz], dtype=np.float32)

            # sensor dropout
            drop_p = float(np.random.uniform(mp.sensor_dropout[0], mp.sensor_dropout[1]))
            if drop_p > 0 and Pk_world.shape[0] > 0:
                keep = (np.random.rand(Pk_world.shape[0]) > drop_p)
                if keep.sum() >= 128:
                    Pk_world = Pk_world[keep]

            # optional overhead crop
            if mp.overhead_mode == "heads_only" and Pk_world.shape[0] > 0:
                z_top = float(Pk_world[:,2].max())
                z_keep = z_top - float(np.random.uniform(mp.z_crop_top_cm[0], mp.z_crop_top_cm[1]))/100.0
                Pk_world = Pk_world[Pk_world[:,2] >= z_keep]

            if Pk_world.shape[0] == 0:
                continue

            # labels
            class_id = self.cls2id[cls]
            prof_key = f"{cls}::{t}"
            pid = self.prof2id.get(prof_key, -1)

            P_list.append(Pk_world)
            Lc_list.append(np.full((Pk_world.shape[0],), class_id, np.int64))
            Lp_list.append(np.full((Pk_world.shape[0],), pid, np.int64))
            I_list.append(np.full((Pk_world.shape[0],), instance_id, np.int64))
            instance_id += 1

        if not P_list:
            return dict(points=np.zeros((0,3),np.float32),
                        labels_class=np.zeros((0,),np.int64),
                        labels_prof=np.zeros((0,),np.int64),
                        instance_ids=np.zeros((0,),np.int64))

        P = np.concatenate(P_list, 0)
        Lc = np.concatenate(Lc_list, 0)
        Lp = np.concatenate(Lp_list, 0)
        Ii = np.concatenate(I_list, 0)
        return dict(points=P, labels_class=Lc, labels_prof=Lp, instance_ids=Ii)
