"""Read-only snap geometry calculation. No RoomScene, renderer or physics replay.

Use the very same geometric kernels as snap_to_object. Detached object records
ensure neither source scenes nor assets are modified. Requires scenesmith's venv.
"""
import copy
import math
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from .offline_check import UnknownEffect, finite_vector, state_from_placed_objects

ROOT = Path(__file__).resolve().parents[1]


def resolve_asset(path):
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p.resolve()
    # Explicit historical output-root relocation, never match by basename.
    for marker in ('/outputs/',):
        if marker in str(path):
            mapped = ROOT / 'scenesmith/outputs' / str(path).split(marker, 1)[1]
            if mapped.is_file():
                return mapped
    raise UnknownEffect(f'snap missing geometry file: {path}')


class SnapGeometry:
    def __init__(self, snapshot, config, asset_root=None, entry_step=None):
        self.records = copy.deepcopy(snapshot['objects'])
        self.normals = copy.deepcopy(snapshot['room_geometry'].get('wall_normals'))
        self.config = copy.deepcopy(config)
        self.events = []
        self.asset_root = Path(asset_root) if asset_root else None
        self.entry_step = entry_step
        self._future_scales = None

    def future_scales(self):
        """Account for later in-place SDF edits using independent scale snapshots.

        A logged rescale alone does not prove success. Require its before/after
        states, including the actual shared SDF path, and compare the multiplier.
        """
        if self._future_scales is not None:
            return self._future_scales
        factors = {}
        if not self.asset_root or not (self.asset_root/'action_log.json').is_file():
            self._future_scales = factors
            return factors
        log = json.loads((self.asset_root/'action_log.json').read_text())
        counts = {}
        entry_global = None
        scales = []
        for a in log:
            name = a['tool_name']
            stage = 'wall' if 'wall_object' in name else 'ceiling' if 'ceiling_object' in name else 'furniture'
            counts[stage] = counts.get(stage,0)+1
            a = dict(a, snapshot_stage=stage, snapshot_step=counts[stage])
            if stage == 'furniture' and counts[stage] == self.entry_step:
                entry_global = a['step_number']
            if name.startswith('_rescale_'):
                scales.append(a)
        if scales and self.entry_step is None:
            raise UnknownEffect('snap SDF history requires entry_step (pre-action snapshot number)')
        if scales and entry_global is None:
            raise UnknownEffect('snap entry snapshot cannot be aligned to action log')
        snapshots = {}
        for stage in ('furniture','wall','ceiling'):
            for f in (self.asset_root/'scene_renders'/stage).glob(f'per_step_{stage}_*/scene_state.json'):
                snapshots[stage,int(f.parent.name.split('_')[3])] = f
        wanted = {o.get('sdf_path') for o in self.records.values()} - {None}
        for a in scales:
            n = a['step_number']
            if n < entry_global:
                continue
            stage, local = a['snapshot_stage'], a['snapshot_step']
            if (stage,local) not in snapshots:
                raise UnknownEffect(f'snap cannot establish on-disk SDF version: rescale {n} lacks adjacent snapshots')
            before = json.loads(snapshots[stage,local].read_text())['objects']
            oid = a['arguments']['object_id']
            if oid not in before:
                # An invalid object cannot reach the SDF rewrite in rescale helper.
                continue
            path = before[oid].get('sdf_path')
            if path not in wanted:
                continue
            if (stage,local+1) not in snapshots:
                raise UnknownEffect(f'snap cannot establish on-disk SDF version: rescale {n} lacks adjacent snapshots')
            after = json.loads(snapshots[stage,local+1].read_text())['objects']
            if oid not in after or after[oid].get('sdf_path') != path:
                raise UnknownEffect(f'snap ambiguous rescale history at {n}')
            ratio = after[oid]['scale_factor']/before[oid]['scale_factor']
            expected = a['arguments']['scale_factor']
            if not math.isfinite(ratio) or ratio <= 0 or not math.isclose(ratio,expected,rel_tol=1e-7):
                raise UnknownEffect(f'snap rescale {n} not confirmed successful by snapshots')
            factors[path] = factors.get(path,1.)*ratio
        self._future_scales = factors
        return factors

    def historical_sdf(self, key, path, state, stack):
        if path is None:
            return None
        rec = self.records[key]
        future = self.future_scales().get(rec.get('sdf_path'),1.)
        # Rescale writes SDF mesh scale AND updates runtime scale_factor. Preserve
        # both native effects; do not accidentally double-correct either one.
        desired = state[key]['scale'][0]/rec.get('scale_factor',1.)
        correction = desired/future
        if math.isclose(correction,1.,rel_tol=1e-12):
            return path
        tree = ET.parse(path)
        for mesh in tree.findall('.//mesh'):
            uri = mesh.find('uri')
            if uri is not None and uri.text:
                uri.text = str((path.parent/uri.text.strip()).resolve())
            scale = mesh.find('scale')
            old = [1.,1.,1.] if scale is None else [float(v) for v in scale.text.split()]
            if not finite_vector(old) or min(old)<=0:
                raise UnknownEffect(f'snap invalid SDF mesh scale: {path}')
            if scale is None:
                scale = ET.SubElement(mesh,'scale')
            scale.text = ' '.join(format(v*correction,'.17g') for v in old)
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='offline_snap_')))/path.name
        tree.write(tmp,encoding='utf-8',xml_declaration=True)
        return tmp

    def initial_state(self):
        records = []
        for oid, o in self.records.items():
            records.append(dict(object_id=oid, translation=o['transform']['translation'],
                                rotation_wxyz=o['transform']['rotation_wxyz'],
                                scale_factor=o.get('scale_factor', 1.)))
        state = state_from_placed_objects(records)
        for oid, o in self.records.items():
            state[oid]['immutable'] = o.get('immutable', False)
            sdf=o.get('sdf_path')
            if sdf and sum(r.get('sdf_path')==sdf for r in self.records.values())>1:
                state[oid]['shared_asset']=True
        return state

    def calculate(self, state, args):
        oid, tid = args.get('object_id'), args.get('target_id')
        mode = args.get('orientation', 'none')
        if set(args) - {'object_id', 'target_id', 'orientation'} or mode not in ('none', 'toward', 'away'):
            raise UnknownEffect('snap invalid arguments')
        if oid == tid:
            raise UnknownEffect('snap source and target alias; unsupported self-snap')
        for key in (oid, tid):
            if key not in state or key not in self.records:
                raise UnknownEffect(f'snap missing entry pose/geometry record: {key}')
            if not finite_vector(state[key].get('position')) or not finite_vector(state[key].get('rotation')):
                raise UnknownEffect(f'snap missing numeric entry pose: {key}')
        if self.records[oid].get('immutable'):
            raise UnknownEffect(f'snap immutable source: {oid}')
        # Check files BEFORE importing dependencies so missing data remains clear.
        paths = {}
        for key in (oid, tid):
            rec = self.records[key]
            paths[key] = {}
            for k in ('geometry_path', 'sdf_path'):
                path = rec.get(k)
                if path and not Path(path).is_absolute() and self.asset_root:
                    path = self.asset_root / path
                paths[key][k] = resolve_asset(path)
            sdf = paths[key]['sdf_path']
            if sdf:
                try:
                    tree = ET.parse(sdf)
                    for uri in tree.findall('.//collision/geometry/mesh/uri'):
                        if not uri.text or not (sdf.parent / uri.text.strip()).is_file():
                            raise UnknownEffect(f'snap missing collision mesh referenced by {sdf}: {uri.text}')
                except ET.ParseError as exc:
                    raise UnknownEffect(f'snap invalid SDF: {sdf}') from exc
        stack = ExitStack()
        try:
            for key in (oid, tid):
                paths[key]['sdf_path'] = self.historical_sdf(key, paths[key]['sdf_path'], state, stack)
            from src.runtime import SCENESMITH_ROOT
            import numpy as np
            from omegaconf import OmegaConf
            from pydrake.all import RigidTransform, RollPitchYaw
            from scenesmith.agent_utils.room import SceneObject, ObjectType
            from scenesmith.furniture_agents.tools.scene_tools import SceneTools, ALREADY_TOUCHING_THRESHOLD_M
            from scenesmith.furniture_agents.tools.snapping_helpers import resolve_collision_if_penetrating, select_and_execute_snap_algorithm
            from scenesmith.utils.shape_analysis import _CIRCULAR_DETECTION_CACHE
            cfg = OmegaConf.create({'snap_to_object': self.config})
            for k in ('snap_margin_m', 'iterative_snap_step_m', 'max_snap_distance_m',
                      'max_sample_vertices', 'circular_detection_volume_ratio_threshold'):
                if k not in self.config or not math.isfinite(self.config[k]):
                    raise UnknownEffect(f'snap missing/invalid config: {k}')
            if cfg.snap_to_object.iterative_snap_step_m <= 0 or cfg.snap_to_object.max_snap_distance_m <= 0 or cfg.snap_to_object.snap_margin_m < 0:
                raise UnknownEffect('snap invalid distance configuration')

            def detached(key):
                rec, pose = self.records[key], state[key]
                scale = pose.get('scale')
                if not finite_vector(scale) or min(scale) <= 0 or max(scale)-min(scale) > 1e-10:
                    raise UnknownEffect(f'snap requires absolute uniform scale: {key}')
                old_scale = rec.get('scale_factor', 1.)
                if not math.isfinite(old_scale) or old_scale <= 0:
                    raise UnknownEffect(f'snap invalid snapshot scale: {key}')
                lo, hi = rec.get('bbox_min'), rec.get('bbox_max')
                if not finite_vector(lo) or not finite_vector(hi) or any(a>b for a,b in zip(lo,hi)):
                    raise UnknownEffect(f'snap missing/invalid local bounds: {key}')
                # Stored bounds already include snapshot scale; rescale only by ratio.
                obj = SimpleNamespace(name=rec['name'], object_id=key,
                    object_type=ObjectType(rec['object_type']),
                    geometry_path=paths[key]['geometry_path'], sdf_path=paths[key]['sdf_path'],
                    scale_factor=scale[0], bbox_min=np.array(lo)*scale[0]/old_scale,
                    bbox_max=np.array(hi)*scale[0]/old_scale,
                    transform=RigidTransform(RollPitchYaw(np.radians(pose['rotation'])), pose['position']))
                obj.compute_world_bounds = lambda: SceneObject.compute_world_bounds(obj)
                return obj

            obj, target = detached(oid), detached(tid)
            # Cache is keyed only by path in native code; isolate comparisons with
            # different configs/bounds to avoid cross-case contamination.
            _CIRCULAR_DETECTION_CACHE.pop(str(target.geometry_path), None)
            calculator = SceneTools.__new__(SceneTools)
            calculator.cfg = cfg
            old_rpy = obj.transform.rotation().ToRollPitchYaw()
            push = resolve_collision_if_penetrating(obj, target, cfg, wall_normals=self.normals)
            applied, _ = calculator._compute_and_apply_orientation(obj, target, mode, old_rpy)
            result = select_and_execute_snap_algorithm(obj, target, mode, applied, oid, tid, cfg)
            if isinstance(result, str):
                raise UnknownEffect(f'snap geometry failed: {result}')
            movement, distance = result
            # Native early return does NOT undo push-out or orientation. Its JSON
            # new_position may still contain the pre-push pose. Model actual state.
            position = obj.transform.translation().copy()
            if distance >= ALREADY_TOUCHING_THRESHOLD_M:
                position += movement
            rpy = np.degrees(obj.transform.rotation().ToRollPitchYaw().vector()).tolist()
            if not finite_vector(position.tolist()) or not finite_vector(rpy):
                raise UnknownEffect('snap produced nonfinite pose')
            self.events.append({'object_id':oid, 'target_id':tid, 'orientation':mode,
                                'pushout_m':push.tolist(), 'snap_distance_m':float(distance),
                                'early_return':bool(distance < ALREADY_TOUCHING_THRESHOLD_M)})
            return position.tolist(), rpy
        except UnknownEffect:
            raise
        except ImportError as exc:
            raise UnknownEffect('snap geometry dependencies unavailable; use scenesmith/.venv/bin/python') from exc
        except Exception as exc:
            raise UnknownEffect(f'snap geometry calculation failed: {exc}') from exc
        finally:
            stack.close()
