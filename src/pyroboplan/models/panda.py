""" Utilities to load example Franka Emika Panda model. """

import coal
import numpy as np
import quaternion
import os
import pinocchio

from ..core.utils import set_collisions
from .utils import get_example_models_folder


def load_models():
    """
    Gets the example Panda models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "panda_description")
    urdf_filename = os.path.join(package_dir, "urdf", "panda.urdf")

    return pinocchio.buildModelsFromUrdf(urdf_filename, package_dirs=models_folder)


def add_self_collisions(model, collision_model, srdf_filename=None):
    """
    Adds link self-collisions to the Panda collision model.

    This uses an SRDF file to remove any excluded collision pairs.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
        srdf_filename : str, optional
            Path to the SRDF file describing the excluded collision pairs.
            If not specified, uses a default file included with the Panda model.
    """
    if srdf_filename is None:
        models_folder = get_example_models_folder()
        package_dir = os.path.join(models_folder, "panda_description")
        srdf_filename = os.path.join(package_dir, "srdf", "panda.srdf")

    collision_model.addAllCollisionPairs()
    pinocchio.removeCollisionPairs(model, collision_model, srdf_filename)


def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0):
    """
    Adds obstacles and collisions to the Panda collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
        visual_model : `pinocchio.Model`
            The Panda visual geometry model.
        inflation_radius : float, optional
            An inflation radius, in meters, around the objects.
    """
    # Add the collision objects
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, -0.151])),
        coal.Box(2.0, 2.0, 0.3),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
        coal.Sphere(0.2 + inflation_radius),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_1)
    collision_model.addGeometryObject(obstacle_sphere_1)

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
        coal.Sphere(0.25 + inflation_radius),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_2)
    collision_model.addGeometryObject(obstacle_sphere_2)

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
        coal.Box(
            0.25 + 2.0 * inflation_radius,
            0.55 + 2.0 * inflation_radius,
            0.55 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_1)
    collision_model.addGeometryObject(obstacle_box_1)

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
        coal.Box(
            0.33 + 2.0 * inflation_radius,
            0.33 + 2.0 * inflation_radius,
            0.33 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_2)
    collision_model.addGeometryObject(obstacle_box_2)

    # Define the active collision pairs between the robot and obstacle links.
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = [
        "ground_plane",
        "obstacle_box_1",
        "obstacle_box_2",
        "obstacle_sphere_1",
        "obstacle_sphere_2",
    ]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)

    # Exclude the collision between the ground and the base link
    set_collisions(model, collision_model, "panda_link0", "ground_plane", False)


def setup_my_scene(model, collision_model, visual_model, inflation_radius=0.0):
    """
    Adds obstacles and collisions to the Panda collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
        visual_model : `pinocchio.Model`
            The Panda visual geometry model.
        inflation_radius : float, optional
            An inflation radius, in meters, around the objects.
    """
    # Add the collision objects
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, -0.151])),
        coal.Box(50.0, 50.0, 0.1),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    box_i = 0

    def add_box(size, pos, color=None, rotation=None):
        nonlocal box_i
        box_i += 1
        box = pinocchio.GeometryObject(
            f"obstacle_box_{box_i}",
            0,
            pinocchio.SE3(np.eye(3) if rotation is None else rotation, np.array(pos)),
            coal.Box(*size),
        )
        box.meshColor = (
            np.array(color) if color is not None else np.array([0.3, 0.3, 0.3, 0.8])
        )
        visual_model.addGeometryObject(box)
        collision_model.addGeometryObject(box)

    add_box((0.25, 1.3, 0.25), (0.0, -1.1, 0.125), np.array([0.0, 1.0, 0.0, 1.0]))

    add_box((0.33, 0.6, 1.13), (-0.5, -0.0, 0.565))

    add_box((0.5, 0.02, 1.0), (0.0, 0.8, 0.5))
    add_box((0.02, 0.4, 0.75), (-0.25 + 0.02 / 2, 0.8 - 0.4 / 2 - 0.02 / 2, 0.375))
    add_box(
        (0.02, 0.4, 0.75), (0.25 - 0.02 / 2, 0.8 - 0.4 / 2, 0.375), (0.3, 0.3, 0.3, 0.4)
    )
    add_box(
        (0.5, 0.4, 0.22), (0.0, 0.8 - 0.4 / 2 - 0.02 / 2, 0.11), (0.3, 0.3, 0.3, 1.0)
    )

    add_box(
        (0.06, 0.15, 0.04),
        (0.0, -0.6, 0.25 + 0.04 / 2),
        color=(0.4, 0.2, 0.5, 1.0),
        rotation=quaternion.as_rotation_matrix(
            quaternion.from_euler_angles(0, 0, np.pi / 7)
        ),
    )

    # Define the active collision pairs between the robot and obstacle bodies.
    all_names = [obj.name for obj in collision_model.geometryObjects]
    collision_names = [name for name in all_names if "panda" in name]
    obstacle_names = set(all_names) - set(collision_names)
    print("obstacles:", obstacle_names)
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)

    # Exclude the collision between the ground and the base link
    set_collisions(model, collision_model, "panda_link0", "ground_plane", False)
