"""
RGBD Image to mesh using a point cloud strategy.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def generate_point_cloud(rgbd_image):
    """
    Generate a point cloud from an RGBD image.

    :param rgbd_image: An RGBDImage containing the color and depth components of an image.
    :type rgbd_image: open3d.geometry.RGBDImage

    :return: 3D point cloud generated from the input RGBD image.
    :rtype: open3d.geometry.PointCloud

    This function creates a point cloud from an RGBD image using the provided camera
    intrinsic and extrinsic parameters.
    The generated point cloud contains the 3D coordinates of the image's pixels,
    with depth values corresponding to the depth information in the input RGBD image.
    """
    # Default camera: o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    focal_distance = 200
    distant_camera = o3d.camera.PinholeCameraIntrinsic(
        1, 1, focal_distance, focal_distance, 0, 0
    )
    extrinsic_parameters = [
        [1, 0, 0, 0.5],
        [0, -1, 0, 0.5],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic=distant_camera, extrinsic=extrinsic_parameters
    )
    return pcd


def pcd_from_image(rgb_image, depth_map):
    """
    Convert an RGB image and a depth map into a point cloud (pcd).

    :param str rgb_image: The input RGB image.

    :param open3d.io.Image depth_map: The input depth map.

    :return: The 3D point cloud generated from the input RGB image and depth map.
    :rtype: open3d.geometry.PointCloud

    This function creates a point cloud from an RGB image and a depth map using the Open3D library.
    The generated point cloud contains the 3D coordinates of the image's pixels,
    with depth values corresponding to the depth information in the input depth map.

    Note: The input RGB image and depth map should be in the same coordinate system and
    have the same dimensions.
    """
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.io.read_image(rgb_image),
        depth=depth_map,
        depth_scale=1,
        convert_rgb_to_intensity=False,
    )
    # For debugging: view_flat_estimation(rgbd_image)
    # To use a point cloud: pcd = generate_point_cloud(rgbd_image)
    pcd = o3d.geometry.PointCloud()
    shape = np.shape(rgbd_image.depth)
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 1, shape[0]), np.linspace(1, 0, shape[1])
    )
    points = np.column_stack(
        (x_grid.flatten(), y_grid.flatten(), 1 - np.asarray(rgbd_image.depth).flatten())
    )
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.asarray(rgbd_image.color).reshape(shape[0] * shape[1], -1) / 256
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return pcd


def clustering(pcd):
    """
    Cluster the point cloud data.

    :param pcd: The input point cloud data.
    :type pcd: open3d.geometry.PointCloud.PointCloud
    :return: The clustered point cloud data.
    :rtype: open3d.geometry.PointCloud.PointCloud

    This function clusters the input point cloud data using the DBSCAN algorithm.
    The function first creates a new point cloud object from the input point cloud.
    Then, it sets the minimum distance between points to be the square root of the number of points
    in the input point cloud.
    With this distance, it performs the DBSCAN clustering algorithm
    with a minimum number of points set to 300.
    The function then assigns colors to the points based on their cluster labels and returns
    the clustered point cloud data.

    Example usage:

    ```python
    pcd = generate_point_cloud(rgbd_image)
    clustered_pcd = clustering(pcd)
    ```
    """
    new_pcd = o3d.geometry.PointCloud(pcd)
    min_dist = len(pcd.points) ** -0.5
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        print(min_dist)
        labels = np.array(
            pcd.cluster_dbscan(eps=min_dist * 20, min_points=300, print_progress=True)
        )

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    new_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([new_pcd])
    return new_pcd


def view_densities(mesh, densities):
    """
    Visualize the density values of a mesh using Open3D.

    :param mesh: The input mesh data.
    :type mesh: open3d.geometry.TriangleMesh

    :param densities: The input density values.
    :type densities: numpy.ndarray

    :return: The visualized mesh with density values.
    :rtype: open3d.geometry.TriangleMesh

    This function visualizes the density values of a mesh using Open3D.
    Then, it assigns these colors to the vertices of the input mesh based on their density values.
    Finally, it returns the visualized mesh with density values.

    Example usage:

    ```python
    mesh = generate_mesh("sunny_mountain.png", "outputs/3D view.obj", "outputs/depth_map.png")
    density_mesh = view_densities(mesh, densities)
    ```
    """
    densities = np.asarray(densities)
    density_colors = plt.get_cmap("plasma")(
        (densities - densities.min()) / (densities.max() - densities.min())
    )
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh])
    return density_mesh


def point_cloud_pipeline(rgb_image, depth_map):
    """
    Pipeline to create a 3D mesh from an image and a depth map.

    :param str rgb_image: The input RGB image.
    :param depth_map: The input depth map.

    :return: A 3D mesh created from the input RGB image and depth map.
    :rtype: open3d.geometry.TriangleMesh

    This function first converts the input RGB image and depth map into a point cloud.
    Then, it creates a 3D mesh from the point cloud.
    Finally, it removes the vertices with low density values and returns the resulting 3D mesh.

    Example usage:

    ```python
    mesh = point_cloud_pipeline("sunny_mountain.png", "outputs/depth_map.png")
    ```
    """
    pcd = pcd_from_image(rgb_image, depth_map)

    # Basic clustering: clustering(pcd)

    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd=pcd, depth=10
    )
    # view_densities(poisson_mesh, densities)

    vertices_to_remove = densities < np.quantile(densities, 0.018)
    trimmed_mesh = o3d.geometry.TriangleMesh(poisson_mesh)
    trimmed_mesh.remove_vertices_by_mask(vertices_to_remove)

    print("Displaying reconstructed mesh ...")
    o3d.visualization.draw_geometries([trimmed_mesh])
    return poisson_mesh
