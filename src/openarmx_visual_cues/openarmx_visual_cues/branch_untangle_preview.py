from __future__ import annotations

from dataclasses import dataclass
import json
import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Capsule:
    center: np.ndarray
    axis: np.ndarray
    half_length: float
    radius: float


@dataclass
class PoseSample:
    position: np.ndarray
    quaternion: np.ndarray


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if np.isfinite(norm) and norm > 1e-9:
        return vector / norm
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0])
    return _unit(np.asarray(fallback, dtype=float), np.array([1.0, 0.0, 0.0]))


def _quaternion_matrix_xyzw(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.eye(3)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    value = np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=float,
    )
    return value / max(float(np.linalg.norm(value)), 1e-9)


def _closest_point(capsule: Capsule, point: np.ndarray) -> np.ndarray:
    offset = np.asarray(point, dtype=float) - capsule.center
    distance = float(
        np.clip(np.dot(offset, capsule.axis), -capsule.half_length, capsule.half_length)
    )
    return capsule.center + distance * capsule.axis


def _clearance(capsule: Capsule, point: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(point) - _closest_point(capsule, point)) - capsule.radius
    )


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))
    )


def _interpolate(
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    include_start: bool = False,
) -> list[np.ndarray]:
    first = 0 if include_start else 1
    return [start + (end - start) * (index / count) for index in range(first, count + 1)]


class BranchUntanglePreview(Node):
    """Generate visualization-only untangling candidates from a target and cable capsules."""

    def __init__(self) -> None:
        super().__init__("branch_untangle_preview")
        self.declare_parameter("global_frame", "world")
        self.declare_parameter("active_arm", "right")
        self.declare_parameter("active_arm_topic", "/visual_cues/active_arm")
        self.declare_parameter("left_target_topic", "/visual_cues/left_selected_target_pose")
        self.declare_parameter("right_target_topic", "/visual_cues/right_selected_target_pose")
        self.declare_parameter("left_start_pose_topic", "/openarm_mini/left_target_pose")
        self.declare_parameter("right_start_pose_topic", "/openarm_mini/right_target_pose")
        self.declare_parameter("cable_capsules_topic", "/perception/cable_capsules")
        self.declare_parameter("left_path_topic", "/untangle/left_ee_path")
        self.declare_parameter("right_path_topic", "/untangle/right_ee_path")
        self.declare_parameter("status_topic", "/untangle/preview_status")
        self.declare_parameter("auto_generate", True)
        self.declare_parameter("bimanual_mode", True)
        self.declare_parameter("approach_distance", 0.12)
        self.declare_parameter("lift_distance", 0.16)
        self.declare_parameter("rotation_angle_deg", 55.0)
        self.declare_parameter("rotation_radius", 0.12)
        self.declare_parameter("pull_distance", 0.20)
        self.declare_parameter("retreat_distance", 0.10)
        self.declare_parameter("points_per_phase", 6)
        self.declare_parameter("tool_clearance", 0.035)

        self.global_frame = str(self.get_parameter("global_frame").value)
        self.active_arm = self._valid_arm(str(self.get_parameter("active_arm").value))
        self.auto_generate = bool(self.get_parameter("auto_generate").value)
        self.bimanual_mode = bool(self.get_parameter("bimanual_mode").value)
        self.approach_distance = max(0.01, float(self.get_parameter("approach_distance").value))
        self.lift_distance = max(0.01, float(self.get_parameter("lift_distance").value))
        self.rotation_angle = math.radians(float(self.get_parameter("rotation_angle_deg").value))
        self.rotation_radius = max(0.02, float(self.get_parameter("rotation_radius").value))
        self.pull_distance = max(0.01, float(self.get_parameter("pull_distance").value))
        self.retreat_distance = max(0.01, float(self.get_parameter("retreat_distance").value))
        self.points_per_phase = max(2, int(self.get_parameter("points_per_phase").value))
        self.tool_clearance = max(0.0, float(self.get_parameter("tool_clearance").value))

        self.targets: dict[str, PoseSample | None] = {"left": None, "right": None}
        self.starts: dict[str, PoseSample | None] = {"left": None, "right": None}
        self.capsules: list[Capsule] = []
        self.last_signature: tuple | None = None
        self.last_paths: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self.last_plan_metadata: dict = {}

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_pubs = {
            "left": self.create_publisher(
                Path, str(self.get_parameter("left_path_topic").value), latched
            ),
            "right": self.create_publisher(
                Path, str(self.get_parameter("right_path_topic").value), latched
            ),
        }
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), latched
        )
        self.create_subscription(
            String,
            str(self.get_parameter("active_arm_topic").value),
            self._active_arm_cb,
            10,
        )
        for arm in ("left", "right"):
            self.create_subscription(
                PoseStamped,
                str(self.get_parameter(f"{arm}_target_topic").value),
                lambda msg, selected_arm=arm: self._target_cb(selected_arm, msg),
                latched,
            )
            self.create_subscription(
                PoseStamped,
                str(self.get_parameter(f"{arm}_start_pose_topic").value),
                lambda msg, selected_arm=arm: self._start_cb(selected_arm, msg),
                10,
            )
        self.create_subscription(
            MarkerArray,
            str(self.get_parameter("cable_capsules_topic").value),
            self._capsules_cb,
            10,
        )
        self.create_service(Trigger, "/untangle_preview/generate", self._generate_service)
        self.create_service(Trigger, "/untangle_preview/clear", self._clear_service)
        self.create_timer(0.25, self._auto_generate)
        self._publish_status(
            "waiting",
            "Select left and right branch grasp targets and provide cable capsules",
        )
        self.get_logger().info(
            "Branch untangle PREVIEW ONLY node started. It publishes nav_msgs/Path "
            "and never sends robot commands."
        )

    @staticmethod
    def _valid_arm(value: str) -> str:
        return value.strip().lower() if value.strip().lower() in ("left", "right") else "right"

    def _active_arm_cb(self, msg: String) -> None:
        self.active_arm = self._valid_arm(msg.data)
        self.last_signature = None

    def _target_cb(self, arm: str, msg: PoseStamped) -> None:
        sample = self._pose_to_global(msg)
        if sample is not None:
            self.targets[arm] = sample
            self.last_signature = None

    def _start_cb(self, arm: str, msg: PoseStamped) -> None:
        sample = self._pose_to_global(msg)
        if sample is not None:
            self.starts[arm] = sample

    def _pose_to_global(self, msg: PoseStamped) -> PoseSample | None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quaternion = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        source = msg.header.frame_id.strip() or self.global_frame
        if source == self.global_frame:
            return PoseSample(position, quaternion / max(float(np.linalg.norm(quaternion)), 1e-9))
        transform = self._lookup(source)
        if transform is None:
            return None
        rotation, translation, tf_quaternion = transform
        return PoseSample(
            rotation @ position + translation,
            _quaternion_multiply_xyzw(tf_quaternion, quaternion),
        )

    def _capsules_cb(self, msg: MarkerArray) -> None:
        capsules: list[Capsule] = []
        for marker in msg.markers:
            if (
                marker.action in (Marker.DELETE, Marker.DELETEALL)
                or marker.type != Marker.CYLINDER
            ):
                continue
            if marker.scale.z <= 1e-6 or min(marker.scale.x, marker.scale.y) <= 1e-6:
                continue
            source = marker.header.frame_id.strip() or self.global_frame
            local_q = np.array(
                [
                    marker.pose.orientation.x,
                    marker.pose.orientation.y,
                    marker.pose.orientation.z,
                    marker.pose.orientation.w,
                ]
            )
            local_rotation = _quaternion_matrix_xyzw(local_q)
            center = np.array(
                [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
            )
            axis = local_rotation[:, 2]
            if source != self.global_frame:
                transform = self._lookup(source)
                if transform is None:
                    continue
                rotation, translation, _ = transform
                center = rotation @ center + translation
                axis = rotation @ axis
            capsules.append(
                Capsule(
                    center=center,
                    axis=_unit(axis),
                    half_length=0.5 * float(marker.scale.z),
                    radius=0.25 * float(marker.scale.x + marker.scale.y),
                )
            )
        if capsules:
            self.capsules = capsules

    def _lookup(self, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.global_frame, source, Time())
        except TransformException as exc:
            self.get_logger().warning(
                f"Cannot transform {source} to {self.global_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
        return (
            _quaternion_matrix_xyzw(quaternion),
            np.array([translation.x, translation.y, translation.z], dtype=float),
            quaternion / max(float(np.linalg.norm(quaternion)), 1e-9),
        )

    def _signature(self, arm: str) -> tuple | None:
        target = self.targets[arm]
        if target is None or not self.capsules:
            return None
        values = [arm, *np.round(target.position, 3)]
        if self.bimanual_mode:
            support_arm = "left" if arm == "right" else "right"
            support_target = self.targets[support_arm]
            if support_target is None:
                return None
            values.extend(np.round(support_target.position, 3))
        for capsule in self.capsules[:4]:
            values.extend(np.round(capsule.center, 3))
            values.extend(np.round(capsule.axis, 2))
        return tuple(values)

    def _auto_generate(self) -> None:
        if not self.auto_generate:
            return
        signature = self._signature(self.active_arm)
        if signature is not None and signature != self.last_signature:
            success, message = self._generate_preview()
            if success:
                self.last_signature = signature
            else:
                self._publish_status("waiting", message)

    def _generate_service(self, _request, response):
        success, message = self._generate_preview()
        response.success = success
        response.message = message
        return response

    def _clear_service(self, _request, response):
        for arm in ("left", "right"):
            path = Path()
            path.header.frame_id = self.global_frame
            path.header.stamp = self.get_clock().now().to_msg()
            self.path_pubs[arm].publish(path)
            self.last_paths[arm] = []
        self.last_signature = self._signature(self.active_arm)
        self._publish_status("cleared", "Preview paths cleared")
        response.success = True
        response.message = "Preview paths cleared"
        return response

    def _generate_preview(self) -> tuple[bool, str]:
        if self.bimanual_mode:
            return self._generate_bimanual(self.active_arm)
        return self._generate(self.active_arm)

    def _generate_bimanual(self, leader_arm: str) -> tuple[bool, str]:
        support_arm = "left" if leader_arm == "right" else "right"
        leader_target = self.targets[leader_arm]
        support_target = self.targets[support_arm]
        if leader_target is None or support_target is None:
            missing = [
                arm for arm in ("left", "right") if self.targets[arm] is None
            ]
            return False, f"Missing selected target for: {', '.join(missing)}"
        branch_vector = self.targets["right"].position - self.targets["left"].position
        grasp_spacing = float(np.linalg.norm(branch_vector))
        if grasp_spacing < 0.08:
            return False, "Left/right grasp targets are too close to define branch direction"

        success, message = self._generate(leader_arm, publish_status=False)
        if not success:
            return success, message
        leader_points = self.last_paths[leader_arm]
        grasp_index = 2 * self.points_per_phase
        if len(leader_points) <= grasp_index:
            return False, "Leader path did not contain the expected grasp phase"

        nearest = min(
            self.capsules,
            key=lambda capsule: _clearance(capsule, support_target.position),
        )
        cable_point = _closest_point(nearest, support_target.position)
        support_outward = support_target.position - cable_point
        support_outward -= nearest.axis * np.dot(support_outward, nearest.axis)
        support_outward = _unit(support_outward, branch_vector)
        support_start = self.starts[support_arm] or support_target
        support_approach = (
            support_target.position + self.approach_distance * support_outward
        )
        support_points = _interpolate(
            support_start.position,
            support_approach,
            self.points_per_phase,
            include_start=True,
        )
        support_points += _interpolate(
            support_approach,
            support_target.position,
            self.points_per_phase,
        )
        for leader_point in leader_points[grasp_index + 1:]:
            rigid_displacement = leader_point - leader_target.position
            support_points.append(support_target.position + rigid_displacement)

        if len(support_points) != len(leader_points):
            return False, "Generated bimanual paths have different waypoint counts"
        self.last_paths[support_arm] = support_points
        self._publish_path(support_arm, support_points, support_target.quaternion)

        branch_axis = _unit(branch_vector)
        metadata = dict(self.last_plan_metadata)
        metadata.update(
            {
                "planner": "bimanual_rigid_branch_release_v1",
                "bimanual": True,
                "leader_arm": leader_arm,
                "support_arm": support_arm,
                "branch_axis_world": [round(float(value), 4) for value in branch_axis],
                "grasp_spacing_m": round(grasp_spacing, 4),
                "synchronized_waypoints": len(leader_points),
            }
        )
        self.status_pub.publish(
            String(data=json.dumps(metadata, separators=(",", ":")))
        )
        message = (
            f"bimanual preview generated ({len(leader_points)} synchronized waypoints, "
            f"grasp spacing={grasp_spacing:.3f} m)"
        )
        self.get_logger().info(f"PREVIEW ONLY: {message}")
        return True, message

    def _generate(self, arm: str, publish_status: bool = True) -> tuple[bool, str]:
        target = self.targets[arm]
        if target is None:
            return False, f"No selected target for {arm} arm"
        if not self.capsules:
            return False, "No cable capsules received"

        nearest = min(self.capsules, key=lambda capsule: _clearance(capsule, target.position))
        cable_point = _closest_point(nearest, target.position)
        tangent = nearest.axis
        radial = target.position - cable_point
        radial -= tangent * np.dot(radial, tangent)
        if np.linalg.norm(radial) < 0.02:
            radial = np.array([0.0, 0.0, 1.0]) - tangent * tangent[2]
        outward = _unit(radial)
        global_up = np.array([0.0, 0.0, 1.0])
        lift_direction = global_up - tangent * np.dot(global_up, tangent)
        lift_direction = _unit(lift_direction, outward)
        if np.dot(lift_direction, outward) < -0.4:
            lift_direction *= -1.0

        start = self.starts[arm] or target
        approach = target.position + self.approach_distance * outward
        grasp = target.position.copy()
        lift = grasp + self.lift_distance * lift_direction

        candidates: list[tuple[float, list[np.ndarray], int, float, float]] = []
        for rotation_sign in (-1, 1):
            arc_center = _closest_point(nearest, lift)
            arc_vector = lift - arc_center
            arc_vector -= tangent * np.dot(arc_vector, tangent)
            arc_vector = _unit(arc_vector, outward) * max(
                self.rotation_radius, float(np.linalg.norm(arc_vector))
            )
            points = _interpolate(
                start.position, approach, self.points_per_phase, include_start=True
            )
            points += _interpolate(approach, grasp, self.points_per_phase)
            points += _interpolate(grasp, lift, self.points_per_phase)
            arc_start_index = len(points)
            for index in range(1, self.points_per_phase + 1):
                angle = rotation_sign * self.rotation_angle * index / self.points_per_phase
                points.append(arc_center + _rotate_about_axis(arc_vector, tangent, angle))
            pull_direction = points[-1] - arc_center
            pull_direction -= tangent * np.dot(pull_direction, tangent)
            pull_direction = _unit(pull_direction, outward)
            pull_end = points[-1] + self.pull_distance * pull_direction
            points += _interpolate(points[-1], pull_end, self.points_per_phase)
            retreat_end = points[-1] + self.retreat_distance * (
                0.65 * pull_direction + 0.35 * lift_direction
            )
            points += _interpolate(points[-1], retreat_end, self.points_per_phase)

            evaluated = points[arc_start_index:]
            minimum_clearance = min(
                _clearance(capsule, point) - self.tool_clearance
                for point in evaluated
                for capsule in self.capsules
            )
            final_clearance = min(_clearance(capsule, points[-1]) for capsule in self.capsules)
            path_length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
            score = final_clearance + 0.45 * minimum_clearance - 0.08 * path_length
            candidates.append((score, points, rotation_sign, minimum_clearance, final_clearance))

        score, points, rotation_sign, minimum_clearance, final_clearance = max(
            candidates, key=lambda item: item[0]
        )
        self.last_paths[arm] = points
        self._publish_path(arm, points, target.quaternion)
        metadata = {
            "mode": "PREVIEW_ONLY",
            "planner": "geometric_cable_release_v1",
            "arm": arm,
            "candidate_count": len(candidates),
            "selected_rotation": "ccw" if rotation_sign > 0 else "cw",
            "score": round(score, 4),
            "min_clearance_m": round(minimum_clearance, 4),
            "final_clearance_m": round(final_clearance, 4),
            "waypoints": len(points),
        }
        self.last_plan_metadata = metadata
        if publish_status:
            self.status_pub.publish(
                String(data=json.dumps(metadata, separators=(",", ":")))
            )
        message = f"{arm} preview generated ({len(points)} waypoints, score={score:.3f})"
        self.get_logger().info(f"PREVIEW ONLY: {message}")
        return True, message

    def _publish_path(
        self,
        arm: str,
        points: list[np.ndarray],
        quaternion: np.ndarray,
    ) -> None:
        path = Path()
        path.header.frame_id = self.global_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for point in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(
                float, point
            )
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ) = map(float, quaternion)
            path.poses.append(pose)
        self.path_pubs[arm].publish(path)

    def _publish_status(self, state: str, message: str) -> None:
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {"mode": "PREVIEW_ONLY", "state": state, "message": message},
                    separators=(",", ":"),
                )
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BranchUntanglePreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
