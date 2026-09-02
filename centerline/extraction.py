from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


_OFFSETS_26 = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
]


@dataclass
class CenterlineBranch:
    branch_id: int
    parent_branch_id: Optional[int]
    branch_order: int
    is_main: bool
    path_voxel_zyx: np.ndarray
    distances_mm: np.ndarray
    length_mm: float
    mean_radius_mm: float
    distance_from_root_mm: float
    starts_at_junction: bool
    ends_at_junction: bool
    starts_at_endpoint: bool
    ends_at_endpoint: bool


@dataclass
class CenterlineTree:
    root_voxel_zyx: np.ndarray
    branches: List[CenterlineBranch]
    main_path_voxel_zyx: np.ndarray
    main_distances_mm: np.ndarray
    junction_voxels_zyx: np.ndarray


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(
        mask, structure=ndi.generate_binary_structure(3, 3)
    )
    if count <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _build_graph(skeleton: np.ndarray, spacing: np.ndarray):
    points = np.argwhere(skeleton)
    index = {tuple(point): i for i, point in enumerate(points)}
    adjacency: List[List[Tuple[int, float]]] = [
        [] for _ in range(len(points))
    ]
    for i, point in enumerate(points):
        point_tuple = tuple(int(value) for value in point)
        for offset in _OFFSETS_26:
            neighbor_point = (
                point_tuple[0] + offset[0],
                point_tuple[1] + offset[1],
                point_tuple[2] + offset[2],
            )
            j = index.get(neighbor_point)
            if j is None or j <= i:
                continue
            weight = float(
                np.linalg.norm(np.asarray(offset, dtype=float) * spacing)
            )
            adjacency[i].append((j, weight))
            adjacency[j].append((i, weight))
    return points, adjacency


def _shortest_paths(adjacency, source: int):
    distances = np.full(len(adjacency), np.inf, dtype=float)
    previous = np.full(len(adjacency), -1, dtype=np.int64)
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return distances, previous


def _trace_path(previous: np.ndarray, target: int) -> List[int]:
    path = []
    node = int(target)
    while node >= 0:
        path.append(node)
        node = int(previous[node])
    return path[::-1]


def _path_edge_set(path: Sequence[int]) -> set:
    return {
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in zip(path[:-1], path[1:])
    }


def _resample_polyline(
    points_mm: np.ndarray, step_mm: float
) -> Tuple[np.ndarray, np.ndarray]:
    segment_lengths = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-8))
    points_mm = points_mm[keep]
    cumulative = cumulative[keep]
    if len(points_mm) < 2 or cumulative[-1] <= 0:
        raise ValueError("Centerline is too short to resample.")
    distances = np.arange(0.0, cumulative[-1], step_mm)
    if (
        distances.size == 0
        or cumulative[-1] - distances[-1] > 0.25 * step_mm
    ):
        distances = np.append(distances, cumulative[-1])
    resampled = np.column_stack(
        [
            np.interp(distances, cumulative, points_mm[:, axis])
            for axis in range(3)
        ]
    )
    return resampled, distances


def _smooth_and_resample_path(
    path_voxel: np.ndarray,
    spacing: np.ndarray,
    step_mm: float,
    smoothing_sigma_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    path_mm = np.asarray(path_voxel, dtype=float) * spacing
    raw_step = max(step_mm * 0.5, 0.1)
    raw_mm, _ = _resample_polyline(path_mm, raw_step)
    sigma_samples = smoothing_sigma_mm / raw_step
    if sigma_samples > 0:
        smooth_mm = ndi.gaussian_filter1d(
            raw_mm, sigma=sigma_samples, axis=0, mode="nearest"
        )
        smooth_mm[0] = raw_mm[0]
        smooth_mm[-1] = raw_mm[-1]
    else:
        smooth_mm = raw_mm
    resampled_mm, distances_mm = _resample_polyline(smooth_mm, step_mm)
    return resampled_mm / spacing, distances_mm


def smooth_and_resample_centerline(
    path_voxel_zyx: np.ndarray,
    spacing_zyx: Sequence[float],
    step_mm: float,
    smoothing_sigma_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Public physical-space smoothing/resampling for an ordered centerline."""
    spacing = np.asarray(spacing_zyx, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three positive values.")
    if step_mm <= 0 or smoothing_sigma_mm < 0:
        raise ValueError("Centerline sampling parameters are invalid.")
    return _smooth_and_resample_path(
        np.asarray(path_voxel_zyx, dtype=float),
        spacing,
        step_mm,
        smoothing_sigma_mm,
    )


def _choose_root(
    points: np.ndarray,
    adjacency,
    mask: np.ndarray,
    spacing: np.ndarray,
    aorta_mask: Optional[np.ndarray],
) -> Tuple[int, np.ndarray]:
    endpoints = np.asarray(
        [
            index
            for index, neighbors in enumerate(adjacency)
            if len(neighbors) == 1
        ],
        dtype=np.int64,
    )
    if endpoints.size == 0:
        endpoints = np.arange(len(points), dtype=np.int64)

    if aorta_mask is not None:
        aorta_mask = np.asarray(aorta_mask, dtype=bool)
        if aorta_mask.shape != mask.shape:
            raise ValueError("aorta_mask must have the same shape as mask.")
        distance_map = ndi.distance_transform_edt(
            ~aorta_mask, sampling=spacing
        )
        endpoint_distances = distance_map[tuple(points[endpoints].T)]
        root = int(endpoints[int(np.argmin(endpoint_distances))])
    else:
        radius_map = ndi.distance_transform_edt(mask, sampling=spacing)
        endpoint_radii = radius_map[tuple(points[endpoints].T)]
        root = int(endpoints[int(np.argmax(endpoint_radii))])
    return root, endpoints


def _build_rooted_skeleton_tree(
    adjacency,
    root: int,
    endpoints: np.ndarray,
):
    graph_distances, previous = _shortest_paths(adjacency, root)
    finite_endpoints = endpoints[np.isfinite(graph_distances[endpoints])]
    targets = [
        int(endpoint)
        for endpoint in finite_endpoints
        if int(endpoint) != root
    ]
    if not targets:
        finite_nodes = np.flatnonzero(np.isfinite(graph_distances))
        target = int(
            finite_nodes[int(np.argmax(graph_distances[finite_nodes]))]
        )
        targets = [target]

    farthest_target = max(
        targets, key=lambda node: graph_distances[node]
    )
    main_path = _trace_path(previous, farthest_target)
    used_edges = set()
    for target in targets:
        used_edges.update(_path_edge_set(_trace_path(previous, target)))

    tree_adjacency = [set() for _ in adjacency]
    for first, second in used_edges:
        tree_adjacency[first].add(second)
        tree_adjacency[second].add(first)
    return tree_adjacency, graph_distances, main_path


def _critical_clusters(tree_adjacency, used_nodes: set):
    degrees = {
        node: len(tree_adjacency[node]) for node in used_nodes
    }
    critical_nodes = {
        node for node, degree in degrees.items() if degree != 2
    }
    clusters = []
    node_to_cluster = {}
    unvisited = set(critical_nodes)
    while unvisited:
        start = min(unvisited)
        stack = [start]
        unvisited.remove(start)
        members = []
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbor in tree_adjacency[node]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        cluster_id = len(clusters)
        members = sorted(members)
        clusters.append(members)
        for node in members:
            node_to_cluster[node] = cluster_id
    return clusters, node_to_cluster, degrees


def _trace_unique_segments(
    tree_adjacency,
    clusters,
    node_to_cluster,
):
    critical_nodes = set(node_to_cluster)
    visited_edges = set()
    segments = []
    for start_cluster, members in enumerate(clusters):
        for start_node in members:
            for neighbor in sorted(tree_adjacency[start_node]):
                edge = (
                    min(start_node, neighbor),
                    max(start_node, neighbor),
                )
                if edge in visited_edges:
                    continue
                neighbor_cluster = node_to_cluster.get(neighbor)
                if neighbor_cluster == start_cluster:
                    visited_edges.add(edge)
                    continue

                path = [start_node]
                previous = start_node
                current = neighbor
                visited_edges.add(edge)
                while current not in critical_nodes:
                    path.append(current)
                    next_nodes = [
                        node
                        for node in tree_adjacency[current]
                        if node != previous
                    ]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0]
                    visited_edges.add(
                        (
                            min(current, next_node),
                            max(current, next_node),
                        )
                    )
                    previous, current = current, next_node
                path.append(current)
                end_cluster = node_to_cluster.get(current)
                if end_cluster is None or end_cluster == start_cluster:
                    continue
                segments.append(
                    {
                        "start_cluster": start_cluster,
                        "end_cluster": end_cluster,
                        "path": path,
                    }
                )
    return segments


def extract_centerline_tree(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    step_mm: float,
    smoothing_sigma_mm: float,
    min_branch_length_mm: float = 3.0,
    min_branch_mean_radius_mm: float = 0.4,
    aorta_mask: Optional[np.ndarray] = None,
) -> CenterlineTree:
    """Extract a rooted tree of non-overlapping centerline segments."""
    mask = _largest_component(np.asarray(mask, dtype=bool))
    if np.count_nonzero(mask) < 8:
        raise ValueError("Vessel mask is empty or too small.")
    if (
        step_mm <= 0
        or min_branch_length_mm < 0
        or min_branch_mean_radius_mm < 0
    ):
        raise ValueError("Centerline steps and branch thresholds are invalid.")

    spacing = np.asarray(spacing_zyx, dtype=float)
    skeleton = _largest_component(skeletonize(mask))
    points, adjacency = _build_graph(skeleton, spacing)
    if len(points) < 2:
        raise ValueError("Could not extract a usable centerline.")

    root, endpoints = _choose_root(
        points, adjacency, mask, spacing, aorta_mask
    )
    tree_adjacency, root_distances, main_path_indices = (
        _build_rooted_skeleton_tree(adjacency, root, endpoints)
    )
    used_nodes = {
        node
        for node, neighbors in enumerate(tree_adjacency)
        if neighbors
    }
    used_nodes.add(root)
    clusters, node_to_cluster, degrees = _critical_clusters(
        tree_adjacency, used_nodes
    )
    segments = _trace_unique_segments(
        tree_adjacency, clusters, node_to_cluster
    )
    if not segments:
        raise ValueError("Could not split the centerline into branches.")

    root_cluster = node_to_cluster[root]
    cluster_segments: Dict[int, List[int]] = {
        cluster_id: [] for cluster_id in range(len(clusters))
    }
    main_edges = _path_edge_set(main_path_indices)
    radius_map = ndi.distance_transform_edt(mask, sampling=spacing)
    for segment_index, segment in enumerate(segments):
        cluster_segments[segment["start_cluster"]].append(segment_index)
        cluster_segments[segment["end_cluster"]].append(segment_index)
        segment_edges = _path_edge_set(segment["path"])
        segment["is_main"] = bool(segment_edges) and segment_edges.issubset(
            main_edges
        )

    oriented = []
    visited_segments = set()
    queue = [(root_cluster, None)]
    while queue:
        cluster_id, parent_record = queue.pop(0)
        candidate_indices = [
            index
            for index in cluster_segments[cluster_id]
            if index not in visited_segments
        ]
        candidate_indices.sort(
            key=lambda index: (
                not segments[index]["is_main"],
                -len(segments[index]["path"]),
                index,
            )
        )
        for segment_index in candidate_indices:
            visited_segments.add(segment_index)
            segment = segments[segment_index]
            if segment["start_cluster"] == cluster_id:
                path = list(segment["path"])
                end_cluster = segment["end_cluster"]
            else:
                path = list(reversed(segment["path"]))
                end_cluster = segment["start_cluster"]
            record_index = len(oriented)
            oriented.append(
                {
                    "parent_record": parent_record,
                    "start_cluster": cluster_id,
                    "end_cluster": end_cluster,
                    "path": path,
                    "is_main": bool(segment["is_main"]),
                }
            )
            queue.append((end_cluster, record_index))

    main_path_voxel, main_distances = _smooth_and_resample_path(
        points[main_path_indices],
        spacing,
        step_mm,
        smoothing_sigma_mm,
    )

    branches = []
    record_to_branch = {}
    for record_index, record in enumerate(oriented):
        raw_path = points[record["path"]].astype(float)
        raw_length = float(
            np.sum(
                np.linalg.norm(
                    np.diff(raw_path * spacing, axis=0), axis=1
                )
            )
        )
        raw_radii = radius_map[tuple(points[record["path"]].T)]
        mean_radius = float(np.mean(raw_radii))
        end_degree = len(cluster_segments[record["end_cluster"]])
        is_terminal = end_degree <= 1
        should_prune = (
            not record["is_main"]
            and is_terminal
            and (
                raw_length < min_branch_length_mm
                or mean_radius < min_branch_mean_radius_mm
            )
        )
        if should_prune or raw_length <= 0:
            continue

        path_voxel, branch_distances = _smooth_and_resample_path(
            raw_path,
            spacing,
            step_mm,
            smoothing_sigma_mm,
        )
        parent_branch_id = record_to_branch.get(
            record["parent_record"]
        )
        if record["is_main"]:
            branch_order = 0
        elif parent_branch_id is None:
            branch_order = 1
        else:
            parent_branch = branches[parent_branch_id]
            branch_order = (
                1
                if parent_branch.is_main
                else parent_branch.branch_order + 1
            )

        branch_id = len(branches)
        record_to_branch[record_index] = branch_id
        start_members = clusters[record["start_cluster"]]
        end_members = clusters[record["end_cluster"]]
        branches.append(
            CenterlineBranch(
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                branch_order=branch_order,
                is_main=record["is_main"],
                path_voxel_zyx=path_voxel,
                distances_mm=branch_distances,
                length_mm=float(branch_distances[-1]),
                mean_radius_mm=mean_radius,
                distance_from_root_mm=float(
                    np.min(root_distances[start_members])
                ),
                starts_at_junction=(
                    len(cluster_segments[record["start_cluster"]]) > 1
                ),
                ends_at_junction=end_degree > 1,
                starts_at_endpoint=any(
                    degrees[node] <= 1 for node in start_members
                ),
                ends_at_endpoint=any(
                    degrees[node] <= 1 for node in end_members
                ),
            )
        )

    if not branches:
        raise ValueError("All extracted branches were removed by pruning.")

    segment_branches = branches
    main_radii = radius_map[tuple(points[main_path_indices].T)]
    merged_branches = [
        CenterlineBranch(
            branch_id=0,
            parent_branch_id=None,
            branch_order=0,
            is_main=True,
            path_voxel_zyx=main_path_voxel,
            distances_mm=main_distances,
            length_mm=float(main_distances[-1]),
            mean_radius_mm=float(np.mean(main_radii)),
            distance_from_root_mm=0.0,
            starts_at_junction=False,
            ends_at_junction=False,
            starts_at_endpoint=True,
            ends_at_endpoint=True,
        )
    ]
    old_to_new = {}
    for old_branch in segment_branches:
        if old_branch.is_main:
            old_to_new[old_branch.branch_id] = 0
            continue
        parent_old = old_branch.parent_branch_id
        while (
            parent_old is not None
            and segment_branches[parent_old].is_main
        ):
            parent_old = segment_branches[parent_old].parent_branch_id
        parent_new = (
            0 if parent_old is None else old_to_new.get(parent_old, 0)
        )
        new_id = len(merged_branches)
        old_to_new[old_branch.branch_id] = new_id
        merged_branches.append(
            CenterlineBranch(
                branch_id=new_id,
                parent_branch_id=parent_new,
                branch_order=old_branch.branch_order,
                is_main=False,
                path_voxel_zyx=old_branch.path_voxel_zyx,
                distances_mm=old_branch.distances_mm,
                length_mm=old_branch.length_mm,
                mean_radius_mm=old_branch.mean_radius_mm,
                distance_from_root_mm=old_branch.distance_from_root_mm,
                starts_at_junction=old_branch.starts_at_junction,
                ends_at_junction=old_branch.ends_at_junction,
                starts_at_endpoint=old_branch.starts_at_endpoint,
                ends_at_endpoint=old_branch.ends_at_endpoint,
            )
        )
    branches = merged_branches

    junctions = [
        np.mean(points[members].astype(float), axis=0)
        for cluster_id, members in enumerate(clusters)
        if len(cluster_segments[cluster_id]) > 2
    ]
    junction_voxels = (
        np.asarray(junctions, dtype=float)
        if junctions
        else np.empty((0, 3), dtype=float)
    )
    return CenterlineTree(
        root_voxel_zyx=points[root].astype(float),
        branches=branches,
        main_path_voxel_zyx=main_path_voxel,
        main_distances_mm=main_distances,
        junction_voxels_zyx=junction_voxels,
    )


def extract_main_centerline(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    step_mm: float,
    smoothing_sigma_mm: float,
    aorta_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper returning only the root-to-farthest path."""
    tree = extract_centerline_tree(
        mask=mask,
        spacing_zyx=spacing_zyx,
        step_mm=step_mm,
        smoothing_sigma_mm=smoothing_sigma_mm,
        min_branch_length_mm=0.0,
        min_branch_mean_radius_mm=0.0,
        aorta_mask=aorta_mask,
    )
    return (
        tree.main_path_voxel_zyx,
        tree.main_distances_mm,
        tree.junction_voxels_zyx,
    )


def tangents_and_frames(
    path_voxel: np.ndarray, spacing_zyx: Sequence[float]
):
    """Build tangents and smoothly transported cross-section frames."""
    spacing = np.asarray(spacing_zyx, dtype=float)
    path_mm = np.asarray(path_voxel, dtype=float) * spacing
    tangents = np.gradient(path_mm, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(norms, 1e-8)

    axes = np.eye(3)
    first_axis = axes[int(np.argmin(np.abs(axes @ tangents[0])))]
    first_u = np.cross(tangents[0], first_axis)
    first_u /= max(np.linalg.norm(first_u), 1e-8)
    first_v = np.cross(tangents[0], first_u)
    first_v /= max(np.linalg.norm(first_v), 1e-8)

    frames_u = [first_u]
    frames_v = [first_v]
    for tangent in tangents[1:]:
        previous_u = frames_u[-1]
        frame_u = previous_u - np.dot(previous_u, tangent) * tangent
        if np.linalg.norm(frame_u) < 1e-6:
            axis = axes[int(np.argmin(np.abs(axes @ tangent)))]
            frame_u = np.cross(tangent, axis)
        frame_u /= max(np.linalg.norm(frame_u), 1e-8)
        frame_v = np.cross(tangent, frame_u)
        frame_v /= max(np.linalg.norm(frame_v), 1e-8)
        frames_u.append(frame_u)
        frames_v.append(frame_v)
    return tangents, np.asarray(frames_u), np.asarray(frames_v)
