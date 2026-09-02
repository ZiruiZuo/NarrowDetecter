# NarrowDetecter Package Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize NarrowDetecter into focused functional subpackages and expose independent detection, independent CPR, and combined detection-plus-marked-CPR workflows.

**Architecture:** Keep numerical algorithms intact while moving them behind `centerline`, `detection`, `cpr`, `io`, `visualization`, and `models` package boundaries. File-level orchestration lives in `api`; `run_combined` reads inputs once, runs detection once, and builds CPR directly from the stored branch measurements and frames.

**Tech Stack:** Python 3, NumPy, SciPy, scikit-image, SimpleITK, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-09-02-narrowdetecter-package-reorganization-design.md`

## Global Constraints

- Preserve all existing uncommitted changes in `README.md`, `cpr_cli.py`, `detector.py`, `models.py`, `myio.py`, and `visualization.py`; move their current working-tree contents, never restore them from `HEAD`.
- Do not change stenosis formulas, reference selection, signal metrics, anatomical naming, CPR geometry, background fill, or marker appearance.
- Do not retain `cli.py`, `cpr_cli.py`, or any other root compatibility forwarder in the completed package.
- The package root exposes exactly three workflow functions: `run_detection`, `run_cpr`, and `run_combined`; configuration and result data classes may also be exported.
- User-specified output directories are created when absent; no workflow adds an inferred subdirectory.
- Detection writes `stenosis_result.json` and enabled visualizations, but no CSV files.
- `run_combined` must not call `extract_centerline_tree` after detection and must not serialize/reload JSON to mark CPR.
- Product code remains under `E:/ZZR/Code/Tools/NarrowDetecter`; temporary tests live under `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization` and are removed after final verification.
- Use `apply_patch` for manual edits and `git mv` only for exact file relocation.
- Stage and commit only files belonging to the current task; never include unrelated working-tree changes accidentally.

---

## File Map

### Create

- `api/__init__.py`: API package exports.
- `api/detection.py`: file-level detection workflow and loaded-array detection helper.
- `api/cpr.py`: file-level standalone CPR workflow.
- `api/combined.py`: single-read combined workflow.
- `models/__init__.py`: model exports.
- `models/detection.py`: existing detection configuration and result models.
- `models/cpr.py`: `CPRConfig` and `CPRResult`.
- `models/combined.py`: `CombinedResult`.
- `centerline/__init__.py`: centerline public exports.
- `centerline/extraction.py`: existing centerline graph, tree, smoothing, and frame logic.
- `centerline/anatomy.py`: anatomical branch naming.
- `detection/__init__.py`: detection exports.
- `detection/detector.py`: existing stenosis pipeline.
- `detection/cross_section.py`: cross-section sampling and region growing.
- `cpr/__init__.py`: CPR generator exports.
- `cpr/generator.py`: straightened/stretched CPR and branch-result CPR generation.
- `io/__init__.py`: image/result I/O exports.
- `io/images.py`: SimpleITK reads and geometry validation.
- `io/results.py`: JSON write/read only.
- `visualization/__init__.py`: visualization exports.
- `visualization/detection.py`: profile and candidate visualizations plus detection visualization orchestration.
- `visualization/cpr.py`: CPR rendering and stenosis annotations.
- `visualization/volumes.py`: marker and anatomical centerline volumes.

### Delete After Migration

- `anatomy.py`
- `centerline.py`
- `cli.py`
- `cpr.py`
- `cpr_cli.py`
- `cross_section.py`
- `detector.py`
- `models.py`
- `myio.py`
- `visualization.py`

### Modify

- `__init__.py`: export only workflows and model/config types.
- `README.md`: document the new imports, combined API, directory structure, JSON-only results, and explicit output directories.
- `requirements.txt`: retain existing dependencies; change only if import verification proves a currently used dependency is missing.

---

### Task 1: Add Characterization Tests and Split Models

**Files:**
- Create: `models/__init__.py`
- Move: `models.py` -> `models/detection.py`
- Create: `models/cpr.py`
- Create: `models/combined.py`
- Modify: `cpr.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_models.py`

**Interfaces:**
- Consumes: current `DetectorConfig`, `DetectionResult`, `BranchResult`, and current `CPRResult` fields.
- Produces: `CPRConfig.validate()`, `CPRResult`, `CombinedResult`, and re-exports from `NarrowDetecter.models`.

- [ ] **Step 1: Record the pre-edit working tree and create the failing model test**

Run `git -C E:/ZZR/Code/Tools/NarrowDetecter status --short` and confirm the six pre-existing modified files remain present. Create `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization` with `New-Item -ItemType Directory -Force`, then create this temporary test with `apply_patch`:

```python
from NarrowDetecter.models import (
    CPRConfig,
    CPRResult,
    CombinedResult,
    DetectionResult,
    DetectorConfig,
)


def test_cpr_config_defaults_and_validation():
    config = CPRConfig()
    assert config.angles_degrees == (0.0,)
    assert config.mode == "straightened"
    assert config.sampling_line_length_mm == 40.0
    config.validate()


def test_cpr_config_rejects_invalid_mode():
    config = CPRConfig(mode="curved")
    try:
        config.validate()
    except ValueError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("invalid CPR mode was accepted")


def test_combined_result_keeps_both_output_directories():
    detection = DetectionResult(spacing_zyx=(1.0, 1.0, 1.0), vessels={})
    result = CombinedResult(
        detection=detection,
        cpr_results=[],
        detection_output_dir="detection-out",
        cpr_output_dir="cpr-out",
        warnings=["no branches"],
    )
    assert result.detection_output_dir == "detection-out"
    assert result.cpr_output_dir == "cpr-out"
    assert result.warnings == ["no branches"]
```

- [ ] **Step 2: Run the test and verify the new models are missing**

Run:

```powershell
$env:PYTHONPATH='E:\ZZR\Code\Tools'
python -m pytest E:\ZZR\Code\Tools\tdd_staging\narrowdetecter_reorganization\test_models.py -v
```

Expected: collection fails because `CPRConfig` or `CombinedResult` is not exported.

- [ ] **Step 3: Move detection models and add CPR/combined models**

Create `models` with `New-Item -ItemType Directory -Force`, then use `git mv models.py models/detection.py`. Move the current `CPRResult` dataclass body from root `cpr.py` into `models/cpr.py`, then add:

```python
@dataclass
class CPRConfig:
    angles_degrees: tuple[float, ...] = (0.0,)
    mode: str = "straightened"
    curve_resolution_mm: float = 1.0
    slice_resolution_mm: float = 1.0
    sampling_line_length_mm: float = 40.0
    interpolation_order: int = 1
    analyze_side_branches: bool = True
    centerline_smoothing_sigma_mm: float = 0.8
    min_branch_length_mm: float = 3.0
    min_branch_mean_radius_mm: float = 0.4
    max_branch_order: int | None = None

    def validate(self) -> None:
        angles = tuple(float(value) for value in self.angles_degrees)
        if not angles or any(not np.isfinite(value) for value in angles):
            raise ValueError("angles_degrees must contain finite values.")
        if self.mode.lower() not in {"straightened", "stretched"}:
            raise ValueError("mode must be 'straightened' or 'stretched'.")
        if self.curve_resolution_mm <= 0 or self.slice_resolution_mm <= 0:
            raise ValueError("CPR resolutions must be positive.")
        if not np.isfinite(self.sampling_line_length_mm) or self.sampling_line_length_mm <= 0:
            raise ValueError("sampling_line_length_mm must be positive and finite.")
        if not 0 <= self.interpolation_order <= 5:
            raise ValueError("interpolation_order must be between 0 and 5.")
        if self.min_branch_length_mm < 0 or self.min_branch_mean_radius_mm < 0:
            raise ValueError("Branch pruning thresholds cannot be negative.")
        if self.max_branch_order is not None and self.max_branch_order < 0:
            raise ValueError("max_branch_order cannot be negative.")
```

Create `models/combined.py` with the exact `CombinedResult` fields from the test. Create `models/__init__.py` that re-exports every detection model plus `CPRConfig`, `CPRResult`, and `CombinedResult`. Update root `cpr.py` to import `CPRResult` from `.models` and remove its local dataclass definition.

- [ ] **Step 4: Run model and import tests**

Run the test from Step 2, then run:

```powershell
python -c "from NarrowDetecter.models import CPRConfig, CPRResult, CombinedResult, DetectorConfig; CPRConfig().validate(); print('models-ok')"
```

Expected: all tests pass and the command prints `models-ok`.

- [ ] **Step 5: Commit the model split**

```powershell
git -C E:\ZZR\Code\Tools\NarrowDetecter add models cpr.py
git -C E:\ZZR\Code\Tools\NarrowDetecter commit -m "refactor: split NarrowDetecter result models"
```

### Task 2: Move Centerline and Detection Algorithms into Feature Packages

**Files:**
- Move: `centerline.py` -> `centerline/extraction.py`
- Move: `anatomy.py` -> `centerline/anatomy.py`
- Create: `centerline/__init__.py`
- Move: `detector.py` -> `detection/detector.py`
- Move: `cross_section.py` -> `detection/cross_section.py`
- Create: `detection/__init__.py`
- Modify: `cpr.py`, `cli.py`, `cpr_cli.py`, `visualization.py`, `__init__.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_feature_imports.py`

**Interfaces:**
- Consumes: all current function signatures from root `centerline.py`, `anatomy.py`, `detector.py`, and `cross_section.py`.
- Produces: `NarrowDetecter.centerline.extract_centerline_tree`, `NarrowDetecter.centerline.tangents_and_frames`, `NarrowDetecter.detection.detect_coronary_stenosis`, and `NarrowDetecter.detection.cross_section.segment_cross_section`.

- [ ] **Step 1: Write the failing feature-package import test**

```python
def test_feature_packages_export_existing_algorithms():
    from NarrowDetecter.centerline import (
        CenterlineBranch,
        extract_centerline_tree,
        smooth_and_resample_centerline,
        tangents_and_frames,
    )
    from NarrowDetecter.centerline.anatomy import build_anatomical_plan
    from NarrowDetecter.detection import detect_coronary_stenosis
    from NarrowDetecter.detection.cross_section import segment_cross_section

    assert callable(extract_centerline_tree)
    assert callable(smooth_and_resample_centerline)
    assert callable(tangents_and_frames)
    assert callable(build_anatomical_plan)
    assert callable(detect_coronary_stenosis)
    assert callable(segment_cross_section)
    assert CenterlineBranch.__name__ == "CenterlineBranch"
```

- [ ] **Step 2: Run the import test and verify package imports fail**

Run `python -m pytest E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_feature_imports.py -v` with `PYTHONPATH=E:/ZZR/Code/Tools`.

Expected: import fails because `centerline` and `detection` are not packages.

- [ ] **Step 3: Relocate files and correct all relative imports**

Use these exact moves:

```powershell
New-Item -ItemType Directory -Force centerline
New-Item -ItemType Directory -Force detection
git mv centerline.py centerline/extraction.py
git mv anatomy.py centerline/anatomy.py
git mv detector.py detection/detector.py
git mv cross_section.py detection/cross_section.py
```

Create package exports. `centerline/__init__.py` re-exports `CenterlineBranch`, `CenterlineTree`, `extract_centerline_tree`, `extract_main_centerline`, `smooth_and_resample_centerline`, and `tangents_and_frames`. `detection/__init__.py` re-exports `detect_coronary_stenosis`.

Apply these import rules consistently:

```python
# centerline/anatomy.py
from .extraction import CenterlineBranch

# detection/cross_section.py
from ..models import DetectorConfig

# detection/detector.py
from ..centerline import CenterlineBranch, extract_centerline_tree, tangents_and_frames
from ..centerline.anatomy import build_anatomical_plan, point_anatomical_label
from .cross_section import make_sampling_grid, sample_plane, segment_cross_section
from ..models import BranchResult, CrossSectionMeasurement, DetectionResult, DetectorConfig, StenosisCandidate, VesselResult
```

Update every remaining package-relative import found by `rg -n "from \\.(anatomy|centerline|cross_section|detector)"` so it targets the new package. Remove direct-execution fallback imports because the supported usage is package import, not running internal files as scripts.

- [ ] **Step 4: Run import, syntax, and lightweight frame tests**

Run:

```powershell
python -m pytest E:\ZZR\Code\Tools\tdd_staging\narrowdetecter_reorganization\test_feature_imports.py -v
python -m compileall -q E:\ZZR\Code\Tools\NarrowDetecter
python -c "import numpy as np; from NarrowDetecter.centerline import tangents_and_frames; p=np.array([[0.,0.,0.],[0.,0.,1.],[0.,0.,2.]]); t,u,v=tangents_and_frames(p,(1.,1.,1.)); assert np.allclose((u*t).sum(1),0); assert np.allclose((v*t).sum(1),0); print('frames-ok')"
```

Expected: tests pass, compilation succeeds, and the command prints `frames-ok`.

- [ ] **Step 5: Commit the algorithm package move**

Stage the new `centerline` and `detection` directories plus only the import updates in current files. Commit with `refactor: group centerline and detection algorithms`.

### Task 3: Move CPR Generation and Add Branch-Result CPR Reuse

**Files:**
- Move: `cpr.py` -> `cpr/generator.py`
- Create: `cpr/__init__.py`
- Modify: `cpr/generator.py`, `visualization.py`, `cpr_cli.py`, `__init__.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_branch_cpr.py`

**Interfaces:**
- Consumes: `BranchResult.measurements`, `CPRConfig`, existing straightened/stretched generators.
- Produces: `generate_branch_cprs(image, spacing_zyx, branch, config, image_direction_xyz) -> list[CPRResult]` without centerline-tree extraction.

- [ ] **Step 1: Write a failing test for straightened and stretched CPR from one stored branch**

Create four `CrossSectionMeasurement` objects with centers `(0, 5, 5)` through `(3, 5, 5)`, distances `0` through `3`, frames `u=(0,1,0)`, `v=(0,0,1)`, and anatomical label `LAD`. Build a `BranchResult(branch_id="LAD-B000", anatomical_label="LAD", distance_from_root_mm=4.0, measurements=measurements, candidates=[])`. The test must call `generate_branch_cprs` once with `CPRConfig(mode="straightened", angles_degrees=(0.0, 45.0), curve_resolution_mm=1.0, slice_resolution_mm=1.0, sampling_line_length_mm=6.0)` and once with the same config using `mode="stretched"`. Assert:

```python
assert len(straightened) == 2
assert all(item.branch_id == "LAD-B000" for item in straightened)
assert all(item.distances_from_root_mm[0] == 4.0 for item in straightened)
assert {item.mode for item in stretched} == {"stretched"}
assert all(item.sampling_line_length_mm == 6.0 for item in stretched)
```

- [ ] **Step 2: Run the branch CPR test and verify `generate_branch_cprs` is missing**

Run the test with `PYTHONPATH=E:/ZZR/Code/Tools`.

Expected: import fails for `generate_branch_cprs`.

- [ ] **Step 3: Move the generator and implement frame-aware branch resampling**

Create `cpr` with `New-Item -ItemType Directory -Force`, then use `git mv cpr.py cpr/generator.py`. Update imports to `..models`, `..centerline`, and `..centerline.anatomy`. Create `cpr/__init__.py` re-exporting `CPRResult` and every current public CPR generator.

Add a private helper that interpolates existing measurements onto `CPRConfig.curve_resolution_mm` without extracting a skeleton:

```python
def _resample_branch_measurements(branch, step_mm, spacing_zyx):
    old_s = np.asarray([item.distance_mm for item in branch.measurements], dtype=float)
    new_s = np.arange(0.0, old_s[-1], step_mm, dtype=float)
    new_s = np.unique(np.concatenate((new_s, [old_s[-1]])))
    centers = np.column_stack([
        np.interp(new_s, old_s, [item.center_voxel_zyx[axis] for item in branch.measurements])
        for axis in range(3)
    ])
    interpolated_u = np.column_stack([
        np.interp(new_s, old_s, [item.frame_u_zyx[axis] for item in branch.measurements])
        for axis in range(3)
    ])
    interpolated_v = np.column_stack([
        np.interp(new_s, old_s, [item.frame_v_zyx[axis] for item in branch.measurements])
        for axis in range(3)
    ])
    centers_mm = centers * np.asarray(spacing_zyx, dtype=float)
    tangents = np.gradient(centers_mm, new_s, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    frames_u = interpolated_u - np.sum(interpolated_u * tangents, axis=1, keepdims=True) * tangents
    frames_u /= np.linalg.norm(frames_u, axis=1, keepdims=True)
    frames_v = np.cross(tangents, frames_u)
    flip = np.sum(frames_v * interpolated_v, axis=1) < 0
    frames_u[flip] *= -1.0
    frames_v[flip] *= -1.0
    return centers, frames_u, frames_v, new_s
```

Reject missing frames, fewer than two measurements, non-increasing distances, and near-zero tangent/frame norms with specific `ValueError` messages. Implement `generate_branch_cprs` as a dispatcher over normalized angles: straightened calls `generate_centerline_cpr` with interpolated frames/distances; stretched calls `generate_stretched_cpr` with the same stored geometry and patient direction. Use `dataclasses.replace` to restore `branch_id`, `anatomical_label`, and nearest-neighbor point anatomical labels on every result. Extend the test to assert both frame axes are unit length, mutually perpendicular, and perpendicular to the resampled tangent within `1e-6`.

- [ ] **Step 4: Prove the branch path does not use label extraction**

In the temporary test, monkeypatch `NarrowDetecter.centerline.extract_centerline_tree` to raise `AssertionError("unexpected extraction")`, then call both modes again. Run all temporary tests and `python -m compileall -q E:/ZZR/Code/Tools/NarrowDetecter`.

Expected: both CPR modes pass while the patched extractor is never called.

- [ ] **Step 5: Commit the CPR package move**

Stage `cpr/` and the related imports only. Commit with `refactor: isolate CPR generation`.

### Task 4: Split Image and JSON I/O and Remove CSV Output

**Files:**
- Create: `io/__init__.py`
- Create: `io/images.py`
- Create: `io/results.py`
- Delete: `myio.py`
- Modify: `cli.py`, `cpr_cli.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_io.py`

**Interfaces:**
- Consumes: current `read_nifti_pair`, `read_mask_like`, `read_stenosis_candidates`, and finite JSON conversion.
- Produces: `write_detection_result(result, output_dir) -> pathlib.Path`, writing only `stenosis_result.json`.

- [ ] **Step 1: Write the failing JSON-only output test**

Use `tmp_path`, create `DetectionResult(spacing_zyx=(1,1,1), vessels={})`, call `write_detection_result`, and assert:

```python
assert result_path == tmp_path / "stenosis_result.json"
assert result_path.exists()
assert list(tmp_path.glob("*.csv")) == []
assert json.loads(result_path.read_text(encoding="utf-8"))["spacing_zyx"] == [1, 1, 1]
```

Also create a minimal SimpleITK image/mask pair and assert `read_nifti_pair` returns z-y-x spacing and rejects a mismatched grid.

- [ ] **Step 2: Run the I/O test and verify the new package is missing**

Run only `test_io.py`; expect import failure for `NarrowDetecter.io.results`.

- [ ] **Step 3: Extract image reads and JSON reads/writes**

Move the current image read functions unchanged into `io/images.py`. Move `read_stenosis_candidates` and `_finite_json` into `io/results.py`. Replace `write_results` with:

```python
def write_detection_result(result: DetectionResult, output_dir: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "stenosis_result.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(_finite_json(result.to_dict()), stream, ensure_ascii=False, indent=2)
    return result_path
```

Delete every CSV row builder, `csv.DictWriter` call, and `import csv`. Export the five I/O functions from `io/__init__.py`. Update current workflow imports and delete `myio.py` with `apply_patch` only after `rg -n "myio|write_results"` shows no remaining consumers.

- [ ] **Step 4: Run I/O and candidate round-trip tests**

Run `test_io.py`, then run all temporary tests. Assert that `read_stenosis_candidates` reads a manually created JSON candidate and groups it by exact `branch_id`.

- [ ] **Step 5: Commit JSON-only I/O**

Stage `io/`, deleted `myio.py`, and updated imports. Commit with `refactor: separate image and JSON result IO`.

### Task 5: Split Detection, CPR, and Volume Visualizations

**Files:**
- Create: `visualization/__init__.py`
- Create: `visualization/detection.py`
- Create: `visualization/cpr.py`
- Create: `visualization/volumes.py`
- Delete: `visualization.py`
- Modify: `cli.py`, `cpr_cli.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_visualization_imports.py`

**Interfaces:**
- Consumes: all current visualization functions and filename/marker behavior.
- Produces: `write_detection_visualizations` and `write_cpr_result` from focused modules.

- [ ] **Step 1: Write failing visualization package tests**

Test imports of `build_stenosis_marker_volume`, `build_anatomical_centerline_volume`, `plot_diameter_profiles`, `plot_intensity_profiles`, `plot_candidate_cross_sections`, `write_detection_visualizations`, and `write_cpr_result`. Add a numerical test for `_stretched_candidate_marker_geometry` using a three-point centerline and assert the returned line is perpendicular to the local tangent and has length equal to `reference_diameter_mm` within `1e-6`.

- [ ] **Step 2: Run the tests and verify package imports fail**

Run only the visualization test. Expected: `NarrowDetecter.visualization.cpr` is not a package.

- [ ] **Step 3: Move existing function bodies by responsibility**

Use `apply_patch` to create the three files with these exact function ownership rules:

```text
volumes.py:
  build_stenosis_marker_volume
  build_anatomical_centerline_volume

detection.py:
  _pyplot
  _shade_candidates
  detection-specific filename helpers
  plot_diameter_profiles
  plot_intensity_profiles
  _perpendicular_frame
  plot_candidate_cross_sections
  plot_candidate_overlays
  write_detection_visualizations (renamed from write_visualizations)

cpr.py:
  _pyplot
  CPR filename helpers
  _candidate_annotation
  _stretched_candidate_marker_geometry
  _write_cpr_result
  write_cpr_result
  write_cpr_visualizations
```

Move bodies without changing calculations, colors, labels, line placement, output names, or DPI. `detection.py` imports volume builders from `.volumes`; both rendering modules import models and algorithms from their functional packages. Create `visualization/__init__.py` with explicit exports. Update workflow imports, then delete root `visualization.py` after `rg -n "from \\.visualization import|visualization.py"` has no stale internal reference.

- [ ] **Step 4: Run visualization and PNG smoke tests**

Run all temporary tests. In `test_visualization_imports.py`, render one synthetic `CPRResult` using Matplotlib Agg into `tmp_path`, assert one nonempty JPEG or PNG exists, and assert the CPR image array is unchanged by rendering.

- [ ] **Step 5: Commit visualization split**

Stage `visualization/`, deleted `visualization.py`, and workflow import updates. Commit with `refactor: split NarrowDetecter visualizations`.

### Task 6: Create Standalone APIs and Remove Legacy CLI Modules

**Files:**
- Create: `api/__init__.py`
- Create: `api/detection.py`
- Create: `api/cpr.py`
- Delete: `cli.py`
- Delete: `cpr_cli.py`
- Modify: `__init__.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_public_api.py`

**Interfaces:**
- Consumes: feature packages, image/JSON I/O, and visualization writers.
- Produces: root-level `run_detection` and `run_cpr` with their current signatures and return types.

- [ ] **Step 1: Write failing public API tests**

```python
import importlib.util


def test_root_exports_workflow_functions_and_models():
    from NarrowDetecter import (
        CPRConfig,
        CPRResult,
        CombinedResult,
        DetectionResult,
        DetectorConfig,
        run_cpr,
        run_detection,
    )
    assert callable(run_detection)
    assert callable(run_cpr)


def test_legacy_cli_modules_are_removed():
    assert importlib.util.find_spec("NarrowDetecter.cli") is None
    assert importlib.util.find_spec("NarrowDetecter.cpr_cli") is None
```

- [ ] **Step 2: Run the tests and verify legacy modules still exist**

Expected: the second test fails and the new API package is absent.

- [ ] **Step 3: Move standalone orchestration into `api`**

Move the current `run_detection` implementation into `api/detection.py` and the current `run_cpr` implementation into `api/cpr.py`. Remove direct-execution fallbacks, `main` aliases, and the hard-coded `if __name__ == "__main__"` sample call.

Factor detection's already-loaded portion into this private helper for Task 7:

```python
def _run_detection_loaded(
    *, label, image, spacing_zyx, image_itk, aorta_mask,
    output_dir, config, print_summary,
    save_visualizations, marker_radius_mm,
    centerline_label_radius_mm,
) -> DetectionResult:
    result = detect_coronary_stenosis(label, image, spacing_zyx, config, aorta_mask=aorta_mask)
    write_detection_result(result, output_dir)
    if save_visualizations:
        write_detection_visualizations(
            result=result,
            image=image,
            label=label,
            reference_itk=image_itk,
            output_dir=output_dir,
            marker_radius_mm=marker_radius_mm,
            centerline_label_radius_mm=centerline_label_radius_mm,
            cross_section_radius_mm=config.cross_section_radius_mm,
            cross_section_pixel_mm=config.cross_section_pixel_mm,
            config=config,
        )
    return result
```

`run_detection` reads inputs and delegates exactly once to `_run_detection_loaded`. `run_cpr` retains every current parameter, validation, explicit-centerline mode, label mode, JSON marker input, save switch, and summary behavior.

- [ ] **Step 4: Replace root exports and delete old CLI files**

Set root `__init__.py` exports to workflows and models only:

```python
from .api.cpr import run_cpr
from .api.detection import run_detection
from .models import CPRConfig, CPRResult, CombinedResult, DetectionResult, DetectorConfig

__all__ = [
    "CPRConfig", "CPRResult", "CombinedResult", "DetectionResult",
    "DetectorConfig", "run_cpr", "run_detection",
]
```

Delete `cli.py` and `cpr_cli.py`. Run `rg -n "cpr_cli|from \\.cli|NarrowDetecter\.cli|NarrowDetecter\.cpr_cli"` and remove every stale reference except historical discussion in the design document.

- [ ] **Step 5: Run standalone API tests and commit**

Run all temporary tests plus `python -m compileall -q E:/ZZR/Code/Tools/NarrowDetecter`. Commit the API move as `refactor: expose standalone NarrowDetecter workflows`.

### Task 7: Implement the Combined In-Memory Workflow

**Files:**
- Create: `api/combined.py`
- Modify: `api/__init__.py`, `__init__.py`
- Test: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/test_combined.py`

**Interfaces:**
- Consumes: `_run_detection_loaded`, `generate_branch_cprs`, `write_cpr_result`, `DetectorConfig`, and `CPRConfig`.
- Produces: `run_combined(label_path, image_path, detection_output_dir, cpr_output_dir, detection_config, cpr_config, aorta_mask_path, print_summary, save_detection_visualizations, marker_radius_mm, centerline_label_radius_mm) -> CombinedResult` and root export `NarrowDetecter.run_combined`.

- [ ] **Step 1: Write a failing combined-flow orchestration test**

Monkeypatch `api.combined.read_nifti_pair` and `read_mask_like` to return one synthetic volume and count reads. Monkeypatch `_run_detection_loaded` to return a fabricated `DetectionResult` containing one vessel, one branch, two measurements, and one candidate. Monkeypatch `generate_branch_cprs` to assert it receives that exact `BranchResult` object and return one synthetic `CPRResult`. Monkeypatch `write_cpr_result` to record the candidate sequence. Assert:

```python
assert reads == {"pair": 1, "aorta": 1}
assert branch_objects_seen == [expected_branch]
assert candidates_seen == [expected_branch.candidates]
assert result.detection is expected_detection
assert len(result.cpr_results) == 1
assert result.detection_output_dir == str(detection_dir)
assert result.cpr_output_dir == str(cpr_dir)
assert detection_dir.is_dir()
assert cpr_dir.is_dir()
assert {path.name for path in tmp_path.iterdir()} == {detection_dir.name, cpr_dir.name}
```

Add a second test with both output arguments pointing to the same directory and assert no automatic `detection` or `cpr` child directory appears.

- [ ] **Step 2: Run the combined test and verify `run_combined` is missing**

Expected: import fails for `NarrowDetecter.api.combined` or root `run_combined`.

- [ ] **Step 3: Implement the exact combined signature and data flow**

```python
def run_combined(
    label_path: str,
    image_path: str,
    detection_output_dir: str,
    cpr_output_dir: str,
    detection_config: DetectorConfig | None = None,
    cpr_config: CPRConfig | None = None,
    aorta_mask_path: str | None = None,
    print_summary: bool = True,
    save_detection_visualizations: bool = True,
    marker_radius_mm: float = 1.5,
    centerline_label_radius_mm: float = 0.6,
) -> CombinedResult:
```

Implementation order is fixed:

1. Instantiate and validate both configs.
2. Create exactly the two specified directories with `Path.mkdir(parents=True, exist_ok=True)`.
3. Call `read_nifti_pair` once and optional `read_mask_like` once.
4. Call `_run_detection_loaded` once.
5. Iterate `detection.vessels.values()` and each retained `branch` with measurements.
6. Call `generate_branch_cprs` for each branch; never call `generate_label_cprs` or `extract_centerline_tree` here.
7. Pass `branch.candidates` directly to `write_cpr_result` for each generated CPR.
8. Convert branch-level `ValueError` into a warning containing vessel and branch IDs; let file read/write errors propagate.
9. Return every in-memory result and both caller-specified directory strings in `CombinedResult`.

- [ ] **Step 4: Add a no-second-extraction regression test**

Patch `NarrowDetecter.cpr.generator.generate_label_cprs` to raise `AssertionError("label CPR path used")` and patch `NarrowDetecter.centerline.extract_centerline_tree` after the fabricated detection result is returned. Call `run_combined`; expected: success, proving the CPR half only consumes branch measurements.

- [ ] **Step 5: Run tests and commit the combined API**

Run `test_combined.py`, then all temporary tests. Export `run_combined` from both `api/__init__.py` and root `__init__.py`. Commit as `feat: add combined stenosis and CPR workflow`.

### Task 8: Update Documentation, Run Full Regression, and Remove Temporary Tests

**Files:**
- Modify: `README.md`
- Verify: all package files
- Delete: `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization/*.py`

**Interfaces:**
- Consumes: the completed package.
- Produces: user-facing documentation matching the final API and a clean package without tests or CSV behavior.

- [ ] **Step 1: Update README imports, calls, and output lists**

Replace old CLI imports with:

```python
from NarrowDetecter import (
    CPRConfig,
    DetectorConfig,
    run_combined,
    run_cpr,
    run_detection,
)
```

Document all three workflows with executable examples. The combined example must pass explicit `detection_output_dir` and `cpr_output_dir`, `DetectorConfig`, `CPRConfig`, and optional `aorta_mask_path`. Remove all CSV output descriptions and state that standalone CPR linking reads `stenosis_result.json`, while combined linking uses memory.

- [ ] **Step 2: Run static boundary checks**

Run:

```powershell
rg -n "from NarrowDetecter\.(cli|cpr_cli)|from \\.(cli|cpr_cli)|import csv|diameter_profiles\.csv|centerline_branches\.csv|stenosis_candidates\.csv" E:\ZZR\Code\Tools\NarrowDetecter -g "*.py" -g "README.md"
rg -n "^def run_(detection|cpr|combined)" E:\ZZR\Code\Tools\NarrowDetecter\api
python -m compileall -q E:\ZZR\Code\Tools\NarrowDetecter
```

Expected: the first search returns no matches; the second returns exactly three workflow definitions; compileall succeeds.

- [ ] **Step 3: Run the entire temporary regression suite**

```powershell
$env:PYTHONPATH='E:\ZZR\Code\Tools'
python -m pytest E:\ZZR\Code\Tools\tdd_staging\narrowdetecter_reorganization -v
```

Expected: every model, import, I/O, CPR, visualization, standalone API, and combined-flow test passes.

- [ ] **Step 4: Verify final output behavior with synthetic NIfTI inputs**

Add an integration case to `test_combined.py` that writes aligned synthetic image, label, and aorta NIfTI files with SimpleITK. Patch only `NarrowDetecter.api.detection.detect_coronary_stenosis` to return the fabricated one-branch `DetectionResult`; keep `_run_detection_loaded`, JSON writing, branch CPR generation, and CPR rendering real. Call `run_combined` with those three input paths, separate temporary detection/CPR directories, `CPRConfig(mode="straightened")`, and `save_detection_visualizations=False`. Assert the detection directory contains `stenosis_result.json`, the CPR directory contains a nonempty rendered CPR image, no `*.csv` exists in either directory, and neither directory contains an automatically generated child directory. Invoke the same case with both output arguments pointing to one directory and assert that directory contains the JSON and CPR image directly.

- [ ] **Step 5: Remove temporary tests and caches**

Delete the temporary test files with `apply_patch`. Remove only generated `__pycache__` and `.pytest_cache` directories whose resolved absolute paths are under `E:/ZZR/Code/Tools/tdd_staging/narrowdetecter_reorganization`; verify the resolved paths before removal. Do not delete anything under a computed or unverified path.

- [ ] **Step 6: Review the final diff and commit documentation/cleanup**

Run:

```powershell
git -C E:\ZZR\Code\Tools\NarrowDetecter status --short
git -C E:\ZZR\Code\Tools\NarrowDetecter diff --check
git -C E:\ZZR\Code\Tools\NarrowDetecter diff --stat
```

Confirm no test files, CSV code, compatibility CLI files, or unrelated paths are staged. Commit README and final cleanup as `docs: update NarrowDetecter workflow documentation`.

- [ ] **Step 7: Perform final verification before reporting completion**

Invoke `superpowers:verification-before-completion`. Review the recorded output from Steps 2 through 4, then run these post-cleanup checks and inspect their actual output:

```powershell
python -m compileall -q E:\ZZR\Code\Tools\NarrowDetecter
python -c "from NarrowDetecter import CPRConfig, DetectorConfig, run_combined, run_cpr, run_detection; CPRConfig().validate(); DetectorConfig().validate(); print('public-api-ok')"
git -C E:\ZZR\Code\Tools\NarrowDetecter status --short
```

Expected: compilation succeeds, the import command prints `public-api-ok`, and status contains no temporary tests or unexpected generated outputs. Only then report completion. Include any test that could not be run and the remaining risk in the final response.
