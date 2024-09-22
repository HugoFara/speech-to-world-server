"""
Several techniques to create a 3D environment from a mesh or a point cloud.

It should be considered a legacy modules as the techniques experimented here are no longer used in the final pipeline.
"""
import os
import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import open3d
from PIL import Image

from .renderer import moving_average, get_depth
from .mesh_pipeline import mesh_impression_pipeline

# Parent directory,
PARENT_DIR = os.path.abspath(os.path.join(importlib.util.find_spec(__name__).origin, os.pardir))
# Generated files folder
OUTPUTS_FOLDER = os.path.join(PARENT_DIR, "outputs")


def cylindrical_projection(flat_vertices, total_angle):
    """Map from vertices from a flat panorama to a circular geometry."""
    vertices = flat_vertices
    far_plane = 10
    radii = vertices[:, 2] * far_plane
    angles = 2 * np.pi * vertices[:, 0] * total_angle
    new_vertices = np.dstack(
        (radii * np.cos(angles), vertices[:, 1], radii * np.sin(angles))
    )[0]
    return new_vertices


def spherical_projection(flat_vertices):
    """Map from vertices from a flat panorama to a circular geometry."""
    vertices = np.asarray(flat_vertices)
    radii = vertices[:, 2]
    # [0, 1] -> [0, tau]
    theta = 2 * np.pi * vertices[:, 0]
    # [1, 0] -> [pi / 2, -pi / 2]
    phi = -np.pi / 2 + vertices[:, 1] * np.pi
    new_vertices = np.dstack(
        (
            radii * np.cos(theta),
            radii * np.sin(theta) * np.sin(phi),
            radii * np.sin(theta) * np.cos(phi),
        )
    )[0]
    return new_vertices


def normalize_depth(vertices):
    """Simple depth normalization between 0 and 1."""
    vertices[:, 2] = (vertices[:, 2] - np.min(vertices[:, 2])) / (
        np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    )
    return vertices


def force_ground_closing(vertices):
    """
    Apply a force to the vertices of a 3D mesh that pulls the lowest vertices to the center.

    :param vertices: A numpy array representing the 3D coordinates of the vertices.
    :type vertices: numpy.ndarray

    :return: The modified 3D coordinates of the vertices after applying the force.
    :rtype: numpy.ndarray
    """
    heights = vertices[:, 1]
    # height = 0 -> 0
    # height > 0.2 -> ~1
    attractions = 1 - np.exp(-heights / 0.05)
    new_vertices = np.copy(vertices)
    new_vertices[:, 2] = 1 - new_vertices[:, 2]
    new_vertices[:, 2] -= np.min(new_vertices[:, 2])
    new_vertices[:, 2] *= -attractions
    return new_vertices


def remove_aberrant_triangles(mesh, limit=0.5):
    """
    This function removes triangles from a 3D mesh that have a normal on z below a specified limit.

    :param mesh: A :class:`open3d.geometry.TriangleMesh` object representing the 3D mesh.
    :type mesh: open3d.geometry.TriangleMesh

    :param limit: A float value representing the minimum z normal for the triangles to be kept.
    :type limit: float

    :return: A new :class:`open3d.geometry.TriangleMesh` object with the aberrant triangles removed.
    :rtype: open3d.geometry.TriangleMesh

    This function takes a 3D mesh and a height limit as input.
    It then removes all triangles from the mesh that have a height below the specified limit.
    The function returns a new 3D mesh object with the aberrant triangles removed.
    """
    new_mesh = open3d.geometry.TriangleMesh(mesh)
    triangles_list = np.nonzero(np.asarray(mesh.triangle_normals)[:, 2] < limit)[0]
    new_mesh.remove_triangles_by_index(triangles_list)
    return new_mesh


def fold_as_panorama(mesh, total_angle=0.5):
    """Fold a mesh as a panorama."""
    # To force depth=0 when y=0: new_vertices = force_ground_closing(np.asarray(mesh.vertices))
    new_vertices = normalize_depth(np.asarray(mesh.vertices))
    new_vertices = cylindrical_projection(new_vertices, total_angle)
    new_mesh = open3d.geometry.TriangleMesh(mesh)
    new_mesh.vertices = open3d.utility.Vector3dVector(new_vertices)
    return new_mesh


def display_meshes(mesh_list):
    """Remove the texture on a meshes to display them with open3d."""
    for mesh in mesh_list:
        mesh.textures = []
    open3d.visualization.draw_geometries(mesh_list)


def save_mesh(mesh, filename, view=False):
    """
    Save the mesh to a file.

    :param mesh: The input mesh data.
    :type mesh: open3d.geometry.TriangleMesh

    :param filename: The path to the file where the mesh will be saved.
    :type filename: str

    :param view: A boolean flag indicating whether to visualize the mesh before saving it.
    :type view: bool

    Example usage:

    ```python
    mesh = generate_mesh("sunny_mountain.png", OUTPUTS_FOLDER + "/3D view.obj", OUTPUTS_FOLDER + "/depth_map.png")
    save_mesh(mesh, OUTPUTS_FOLDER + "/3D view.obj", view=True)
    ```
    """
    # Disable texture for visualization
    if view:
        mesh.textures = []
        open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

    open3d.io.write_triangle_mesh(filename, mesh)



def horizon_height(depth_map):
    """Get the height of the horizon using gradient only, not precise."""
    y_depth_grad = np.gradient(depth_map, axis=0)
    reduced_grad = np.mean(y_depth_grad, axis=1)

    # The ground is the area of negative gradient, find the first occurrence of positive gradient
    smoothed_grad = moving_average(reduced_grad, depth_map.shape[0] // 50)

    return np.argmax(smoothed_grad)


def plot_horizon_computation(depth_map):
    """Compare different horizon line finding methods."""
    mean_depth = np.mean(depth_map, axis=1)
    y_depth_grad = np.gradient(depth_map, axis=0)
    reduced_grad = np.mean(y_depth_grad, axis=1)

    # The ground is the area of negative gradient, find the first occurrence of positive gradient
    smoothed_grad = moving_average(reduced_grad, depth_map.shape[0] // 50)

    horizon = horizon_height(depth_map)

    y_indices = np.arange(depth_map.shape[0])
    _, axes = plt.subplots(1, 2, sharey=True)
    axes[0].imshow(depth_map)
    axes[0].plot(mean_depth * depth_map.shape[1], y_indices, label="Average depth")
    axes[1].plot(reduced_grad, y_indices, label="Vertical depth gradient")
    axes[1].plot(smoothed_grad, y_indices, label="Smoothed depth gradient")
    axes[1].plot(
        (np.min(reduced_grad), np.max(reduced_grad)),
        (horizon, horizon),
        label="Detected horizon",
    )
    axes[1].plot(
        moving_average(smoothed_grad),
        y_indices,
    )
    axes[1].plot((0, 0), (0, depth_map.shape[0]))
    plt.grid()
    axes[1].set_position(
        [
            axes[1].get_position().x0,
            axes[0].get_position().y0,
            axes[1].get_position().width,
            axes[0].get_position().height,
        ]
    )
    plt.legend()
    plt.show()
    exit()


def generate_mesh(texture, depth_map, resolution=256):
    """
    Generate a 3D mesh from an image and a depth map.

    :param open3d.geometry.Image texture: The input RGB image to use as mesh texture.
    :param depth_map: The input depth map.
    :type depth_map: numpy.ndarray
    :param int resolution: Vertices per side in the generated mesh.

    :return: A 3D mesh created from the input RGB image and depth map.
    :rtype: open3d.geometry.TriangleMesh

    This function generates a 3D mesh from an image and a depth map using the following steps:

    1. Convert the input RGB image and depth map into a point cloud.
    2. Create a 3D mesh from the point cloud using a Poisson surface reconstruction.
    3. Remove the vertices with low density values.
    4. Save the resulting 3D mesh to a file specified by the `output_mesh` parameter.

    Example usage:

    ```python
    generate_mesh("sunny_mountain.png", OUTPUTS_FOLDER + "/3D view.obj", OUTPUTS_FOLDER + "/depth_map.png")
    ```
    """
    # To use the point cloud alternative: mesh = environment.point_cloud_pipeline(input_image, depth_map)
    mesh = mesh_impression_pipeline(depth_map, resolution, texture)
    new_mesh = remove_aberrant_triangles(mesh, 0.1)
    new_mesh = fold_as_panorama(new_mesh, 1)
    # display_meshes([mesh, new_mesh])
    return new_mesh


def mesh_panorama_from_files(
    input_image, output_mesh, depth_image=None, resolution=256
):
    """
    Generate a 3D mesh from an image and a depth map.

    :param str input_image: The input RGB image.
    :param str output_mesh: The path to the file where the mesh will be saved.
    :param depth_image: The input depth map. If not provided, it will be generated.
    :type depth_image: str or None
    :param int resolution: Vertices per side in the generated mesh.

    :return: A 3D mesh created from the input RGB image and depth map.
    :rtype: open3d.geometry.TriangleMesh

    Example usage:

    ```python
    mesh_panorama_from_files("sunny_mountain.png", OUTPUTS_FOLDER + "/3D view.obj", OUTPUTS_FOLDER + "/depth_map.png")
    ```
    """
    if depth_image is None:
        depth_map = get_depth(Image.open(input_image))
    else:
        # Image im I;16 format with depth on 16 bits.
        depth_map = np.asarray(open3d.io.read_image(depth_image)) / (2**16 - 1)
        plot_horizon_computation(depth_map)
    main_texture = open3d.io.read_image(input_image)
    new_mesh = generate_mesh(main_texture, depth_map, resolution)

    if output_mesh is not None:
        save_mesh(new_mesh, output_mesh)
        print(f"Mesh saved as '{output_mesh}'.")

    return new_mesh


def save_as_scene(skybox, terrain, depth_map, _objects):
    """
    Save all the elements as objects in a scene.

    :param PIL.Image.Image skybox: Skybox to save.
    :param PIL.Image.Image terrain: Terrain texture.
    :param numpy.ndarray depth_map: Terrain depth map.
    :param _objects: Objects to save.
    :type _objects: list[tuple[PIL.Image.Image, numpy.ndarray]]
    """
    skybox_path = OUTPUTS_FOLDER + "/complete_skybox.png"
    skybox.save(OUTPUTS_FOLDER + "/complete_skybox.png")
    print("Saved the skybox under " + skybox_path)
    terrain_texture_path = OUTPUTS_FOLDER + "/terrain_texture.png"
    terrain.save(terrain_texture_path)
    terrain_mesh = generate_mesh(open3d.io.read_image(terrain_texture_path), depth_map)
    terrain_mesh_path = OUTPUTS_FOLDER + "/terrain_mesh.obj"
    save_mesh(terrain_mesh, terrain_mesh_path)
    print("Saved the mesh under " + terrain_mesh_path)
    warnings.warn("Objects are not handled yet.")


if __name__ == "__main__":
    # To generate a new mesh
    mesh_panorama_from_files(
        os.path.join(PARENT_DIR, '../sunny_mountain.png'),
        OUTPUTS_FOLDER + "/sunny 3D.obj",
        os.path.join(PARENT_DIR, 'sunny_depth_map.png')
    )
