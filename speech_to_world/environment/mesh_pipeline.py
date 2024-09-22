"""
Functions to generate a mesh from an RGBD image.

This module aims at the creation of a mesh by "impression".
"""
import numpy as np
import open3d
import torch


def reduce_image_size(initial_data, resolution):
    """
    Take an input 2D array and averages it to reduce its size.

    :param numpy.ndarray initial_data: Initial array of shape (W, H)
    :param int resolution: Target resolution to match.
    :return numpy.ndarray: A new array of size (resolution, resolution)
    """
    # We have more depth pixels than vertices, hence the average
    dilate = initial_data.shape[0] // resolution, initial_data.shape[1] // resolution

    averager = torch.nn.AvgPool2d(dilate, stride=dilate)
    # Reshape as (N, C, H, W) and pass to torch
    tensor_data = torch.from_numpy(
        initial_data.reshape(1, 1, initial_data.shape[0], initial_data.shape[1])
    )
    average = np.asarray(averager(tensor_data)[0, 0])
    return average


def create_triangle_list(rows, cols):
    """
    Create an indices of triangles that maps the vertices in a planar mesh.

    :param int rows: Number of rows
    :param int cols: Number of columns.
    :return numpy.ndarray: Triangles indices of shape (rows * cols * 2, 3)
    """
    # Create a grid of vertex indices
    indices = np.arange(rows * cols).reshape(rows, cols)

    # Generate triangles
    triangles = []

    # Upper-left triangles
    upper_left = indices[:-1, :-1].reshape(-1, 1)
    upper_right = indices[:-1, 1:].reshape(-1, 1)
    lower_left = indices[1:, :-1].reshape(-1, 1)
    triangles.append(np.hstack((upper_left, lower_left, upper_right)))

    # Lower-right triangles
    lower_right = indices[1:, 1:].reshape(-1, 1)
    triangles.append(np.hstack((upper_right, lower_left, lower_right)))

    # Combine all triangles
    return np.vstack(triangles)


def create_mesh_geometry(max_resolution, depth_data):
    """
    Create a set of vertices and triangles as a plan with deformation.

    :param max_resolution: The approximate maximum resolution of the generated mesh.
    :type max_resolution: int

    :param depth_data: The depth data of the input image.
    :type depth_data: numpy.ndarray

    :return: A tuple containing the vertices and triangles of the generated mesh.
    :rtype: tuple

    This function creates a grid of vertices positions based on the depth data of the input image.
    It then assigns the vertices position and depth to the grid.
    Finally, it creates a grid of triangles for the planar mesh.

    The function first calculates the average depth value for each pixel in the depth data.
    It then generates a grid of vertices positions based on the average depth values.
    The vertices are assigned positions based on their corresponding pixel coordinates
    in the depth data.

    The function then creates a grid of triangles for the planar mesh.

    The function returns a tuple containing the vertices and triangles of the generated mesh.
    """
    # Grid of vertices positions
    resized_depth = reduce_image_size(depth_data, max_resolution)
    resolution = np.shape(resized_depth)
    # plot_arrays(data, average)

    # Assign vertices position and depth
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 1, resolution[1]), np.linspace(1, 0, resolution[0])
    )

    # Legacy code? view_height = 0.3
    # Legacy code? z_grid = np.sqrt(np.abs(resized_depth ** 2 - (y_grid - view_height) ** 2))

    vertices = np.column_stack(
        (x_grid.flatten(), y_grid.flatten(), resized_depth.flatten())
    )

    # Create a grid of triangles for the planar mesh
    triangles = create_triangle_list(resolution[0], resolution[1])

    return vertices, triangles


def generate_uv(triangles, vertices):
    """
    Generate the uv coordinates for a planar mesh.

    :param triangles: An array of shape (n, 3) for the indices of the vertices in the mesh.
    :type triangles: numpy.ndarray

    :param vertices: An array of shape (n, 3) for the 3D coordinates of the vertices in the mesh.
    :type vertices: numpy.ndarray

    :return: An array of shape (n*3, 2) representing the uv coordinates of the vertices in the mesh.
    :rtype: numpy.ndarray

    This function generates the uv coordinates for a planar mesh.
    It takes as input the indices and 3D coordinates of the vertices in the mesh,
    and returns a numpy array containing the uv coordinates of the vertices.

    Example usage:

    ```python
    triangles = np.array([[0, 1, 2]])
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    uv_coords = generate_uv(triangles, vertices)
    print(uv_coords)
    ```
    """
    v_uv = np.empty((len(triangles) * 3, 2))
    for i, t in enumerate(triangles):
        for j in range(3):
            v_uv[i * 3 + j] = vertices[t[j]][:2] * [1, -1]
    return v_uv


def mesh_impression_pipeline(depth_map, max_resolution=256, texture_image=None):
    """
    Pipeline to create a mesh from a depth map.

    :param depth_map: The input depth map.
    :type depth_map: numpy.ndarray

    :param max_resolution: Approximate maximum resolution of the generated mesh.
    :type max_resolution: int

    :param texture_image: The texture image for the mesh.
    :type texture_image: open3d.geometry.Image or None

    :return: A 3D mesh created from the input depth map.
    :rtype: open3d.geometry.TriangleMesh

    This function creates a 3D mesh from a depth map using the following steps:

    1. Create a grid of vertices positions based on the depth data of the input depth map.
    2. Assign the vertices position and depth to the grid.
    3. Create a grid of triangles for the planar mesh.
    4. Generate the uv coordinates for the mesh.
    5. Load a texture image (if provided) and assign it to the mesh.
    6. Compute the vertex normals for the mesh.

    Example usage:

    ```python
    depth_map = np.asarray(Image.open("depth_map.png")) / 65535
    mesh = mesh_impression_pipeline(depth_map, 256, "texture.png")
    ```
    """
    # Create the mesh
    vertices, triangles = create_mesh_geometry(max_resolution, np.asarray(depth_map))

    mesh = open3d.geometry.TriangleMesh(
        open3d.utility.Vector3dVector(vertices), open3d.utility.Vector3iVector(triangles)
    )

    # Load a texture image (change the file path accordingly)
    v_uv = generate_uv(triangles, vertices)
    mesh.triangle_uvs = open3d.utility.Vector2dVector(v_uv)

    if texture_image is not None:
        texture_image = open3d.io.read_image(texture_image)
        mesh.textures = [texture_image]

    mesh.compute_vertex_normals()

    return mesh
