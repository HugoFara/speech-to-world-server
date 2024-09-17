"""
Image segmentation techniques using the depth map.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import skimage
from sklearn.preprocessing import minmax_scale, scale
from sklearn.cluster import KMeans, DBSCAN
import torch

from .mask_former import mask_former, panoptic_segmentation, get_sky_ids


RANDOM_SEED = 0

DEFAULT_IMAGE = "../sunny_mountain.png"


def planar_grid(image):
    """
    Create a planar grid of values.

    The output grid will have dimension (*image.shape, 2) that respects
    ``grid[x, y] = [image.shape[0] / x, image.shape[1] / y]``.
    """
    # Grid from [0, image.shape[axis]] on each axis
    grid = np.indices(image.shape).T
    # Reduce each axis to [0, 1]
    return grid / (image.shape - np.ones(image.ndim))


def do_kmeans(data, n_clusters):
    """Apply a K-mean clustering."""
    flat_data = data.reshape((-1, data.shape[2]))

    # Create an instance of the K-Means clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    # Fit the K-Means algorithm to the pixel data
    kmeans.fit(flat_data)
    # Predict the cluster labels for each pixel
    labels = kmeans.predict(flat_data)

    return labels.reshape(*data.shape[:2])


def deep_scale_stack_data(*data):
    """Stack data on depth, normalized."""
    normalized_data = tuple(
        minmax_scale(d.reshape(-1, 1)).reshape(*d.shape) for d in data
    )
    return np.dstack(normalized_data)


def depth_clustering(image, n_clusters=5):
    """Cluster using depth only."""
    return do_kmeans(image, n_clusters)


def spatial_clustering(image, n_clusters=15):
    """Cluster using XYD data."""
    grid = planar_grid(image)
    xyd_image = deep_scale_stack_data(image, grid)
    return do_kmeans(xyd_image, n_clusters)


def rgbd_spatial_clustering(depth_image, rgb_image, n_clusters=15):
    """Cluster using spatial K-means on RGBD data."""
    grid = planar_grid(depth_image)
    xy_rgbd_image = deep_scale_stack_data(grid, rgb_image, depth_image)
    return do_kmeans(xy_rgbd_image, n_clusters)


def mask_former_clustering(depth_image, mask_former_labels, n_clusters=15):
    """Cluster using MaskFormer."""
    grid = planar_grid(depth_image)
    xyd_mask_image = deep_scale_stack_data(grid, depth_image, mask_former_labels)
    return do_kmeans(xyd_mask_image, n_clusters)


def spatial_fz_mf_clustering(image, segments_fz, mask_former_labels, n_clusters=None):
    """Cluster using Felzenszwalbs's method."""
    grid = planar_grid(image)
    if n_clusters is None:
        sizes = np.unique(segments_fz).shape[0], np.unique(mask_former_labels).shape[0]
        n_clusters = max(sizes)
        print(f"Segmenting in {n_clusters} clusters")

    stacked_data = deep_scale_stack_data(grid, image, segments_fz, mask_former_labels)
    return do_kmeans(stacked_data, n_clusters)


def compare_segmentations(image_path, depth_path):
    """Compare various segmentation methods."""
    # Load the grayscale image
    image = skimage.io.imread(depth_path, as_gray=True)

    # Remove background
    far_clip = np.quantile(image, 0.7)
    clipped = np.clip(image, 0, far_clip)

    # Apply Gaussian filtering to reduce noise (optional)
    filtered_image = skimage.filters.gaussian(clipped, sigma=1)

    original_spatial = spatial_clustering(skimage.filters.gaussian(image, sigma=1), 12)
    spatial_clusters = spatial_clustering(filtered_image, 12)

    clustering_labels = spatial_clusters.reshape(filtered_image.shape)
    mask_former_labels = mask_former(Image.open(image_path))

    mask_former_clusters = mask_former_clustering(clipped, mask_former_labels, 10)
    k_mask_former_labels = mask_former_clusters.reshape(filtered_image.shape)

    segments_fz = skimage.segmentation.felzenszwalb(
        clipped, scale=1, min_size=int(np.sqrt(image.shape[0] * image.shape[1]) * 10)
    )

    fz_mf_labels = spatial_fz_mf_clustering(
        clipped, segments_fz, mask_former_labels, n_clusters=None
    )

    _fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    ax[0, 0].set_title("Spatial K-means")
    ax[0, 0].imshow(
        skimage.segmentation.mark_boundaries(
            image, original_spatial.reshape(filtered_image.shape)
        )
    )
    ax[0, 1].set_title("Clipped spatial K-means")
    ax[0, 1].imshow(skimage.segmentation.mark_boundaries(image, clustering_labels))
    ax[0, 2].set_title("Felzenszwalbs's method")
    ax[0, 2].imshow(skimage.segmentation.mark_boundaries(image, segments_fz))
    ax[1, 0].set_title("MaskFormer segmentation")
    ax[1, 0].imshow(skimage.segmentation.mark_boundaries(image, mask_former_labels))
    ax[1, 1].set_title("K-Mean + MaskFormer segmentation")
    ax[1, 1].imshow(skimage.segmentation.mark_boundaries(image, k_mask_former_labels))
    ax[1, 2].set_title("MaskFormer + Felzenszwalbs,\nK-mean")
    ax[1, 2].imshow(skimage.segmentation.mark_boundaries(image, fz_mf_labels))
    ax[2, 0].set_title("Spatial K-means")
    ax[2, 0].imshow(clustering_labels)
    ax[2, 1].set_title("K-Mean + MaskFormer segmentation")
    ax[2, 1].imshow(k_mask_former_labels)
    ax[2, 2].set_title("MaskFormer + Felzenszwalbs,\nK-mean")
    ax[2, 2].imshow(fz_mf_labels)

    plt.show()


def segmentation_maps(image_path, depth_path):
    """Segment the image and display the result."""
    # Load the grayscale image
    image = skimage.io.imread(depth_path, as_gray=True)

    # Remove background
    far_clip = np.quantile(image, 0.7)
    clipped = np.clip(image, 0, far_clip)

    grid = planar_grid(image)
    xyd_image = deep_scale_stack_data(image, grid)

    mask_former_labels = mask_former(Image.open(image_path))

    segments_fz = skimage.segmentation.felzenszwalb(
        clipped, scale=1, min_size=int(np.sqrt(image.shape[0] * image.shape[1]) * 10)
    )

    fz_mf_labels = spatial_fz_mf_clustering(
        clipped, segments_fz, mask_former_labels, n_clusters=None
    )

    _fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    ax[0, 0].set_title("Clipped xyD image")
    ax[0, 0].imshow(xyd_image)
    ax[0, 1].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(segments_fz)
    ax[1, 0].set_title("MaskFormer segmentation")
    ax[1, 0].imshow(mask_former_labels)
    ax[1, 1].set_title("Final segmentation")
    ax[1, 1].imshow(fz_mf_labels)

    plt.show()


def segment_image(image_path, depth_path, far_clip_pos=0.7):
    """Segment the RGB image with all methods."""
    # Load the grayscale image
    image = skimage.io.imread(depth_path, as_gray=True)

    # Remove background
    far_clip = np.quantile(image, far_clip_pos)
    clipped = np.clip(image, 0, far_clip)

    mask_former_labels = mask_former(Image.open(image_path))

    segments_fz = skimage.segmentation.felzenszwalb(
        clipped, scale=1, min_size=int(np.sqrt(image.shape[0] * image.shape[1]) * 10)
    )

    return spatial_fz_mf_clustering(clipped, segments_fz, mask_former_labels)


def show_images_grid(np_images):
    """Show images in a nice grid."""
    n_cols = int(np.sqrt(len(np_images)))
    n_lines = int(np.ceil(len(np_images) / n_cols))
    _fig, ax = plt.subplots(n_lines, n_cols)
    for i, result in enumerate(np_images):
        ax[i // n_cols, i % n_cols].imshow(result)
    plt.show()


def split_image(image, labels):
    """Split image in the different labels."""
    labels_values = np.unique(labels)
    outputs = []
    for target_label in labels_values:
        mask = (labels == target_label).astype(image.dtype)
        image_masked = image * np.dstack((mask, mask, mask))
        outputs.append(image_masked)
        Image.fromarray(image_masked).save(f"outputs/image_{target_label}.png")
    # show_images_grid(outputs)
    return outputs


def crop_to_mask(np_image, mask):
    """Crop an image to a specific mask."""
    # Find the indices of non-False
    non_null_rows, non_null_cols = np.nonzero(mask)

    # Find the bounding box
    crop = np_image[
        np.min(non_null_rows) : np.max(non_null_rows),
        np.min(non_null_cols) : np.max(non_null_cols),
    ]
    return crop


def crop_to_content(np_images):
    """Crop an image to a not empty space."""
    crops = []
    for i, np_im in enumerate(np_images):
        mask = np.sum(np_im, axis=2) > 0
        crop = crop_to_mask(np_im, mask)
        crops.append(crop)

        cropped_mask = crop_to_mask(mask, mask)
        transparent = np.dstack((crop, cropped_mask * 255)).astype(np.uint8)
        Image.fromarray(transparent).save(f"outputs/cropped_{i}.png")
    return crops


def segment_and_save(image_path, depth_path):
    """Segment an image and save each segment."""
    labels = segment_image(image_path, depth_path)
    images = split_image(skimage.io.imread(image_path), labels)
    segments = crop_to_content(images)
    # show_images_grid(segments)
    return segments


def is_segment_skybox(
    depth_field, mask, far_threshold, near_threshold, vertical_gradient=0.2
):
    """Check if a given segment is a good skybox candidate."""
    data = depth_field[mask]
    mean_depth = np.mean(data)
    # Object is very far: skybox
    if mean_depth > far_threshold:
        return True
    # Object very near: not a skybox
    if mean_depth < near_threshold:
        return False
    # Otherwise check if it is deep with a nice vertical decreasing gradient
    masked_image = depth_field * mask / mean_depth
    return np.mean(masked_image[1:] - masked_image[:-1]) < vertical_gradient


def mask_skybox(image_path, depth_path, labeled_image):
    """List of the labels corresponding to the sky."""
    rgb_image = skimage.io.imread(image_path)
    depth_map = skimage.io.imread(depth_path)
    _images = split_image(rgb_image, labeled_image)
    labels = np.unique(labeled_image)

    skybox_list = []
    far_plane = np.quantile(depth_map, 0.55)
    near_plane = np.quantile(depth_map, 0.30)
    # Check if elements belong to the skybox
    for label_id in labels:
        mask = labeled_image == label_id
        depth_field = crop_to_mask(depth_map, mask)
        is_skybox = is_segment_skybox(
            depth_field, crop_to_mask(mask, mask), far_plane, near_plane
        )
        if is_skybox:
            # Mark the label
            skybox_list.append(label_id)

    return skybox_list


def mask_terrain(image_path, depth_path, labeled_image, ignore_labels=None):
    """Create a mask for the labels belonging to the terrain."""
    rgb_image = skimage.io.imread(image_path)
    depth_map = minmax_scale(skimage.io.imread(depth_path).flatten()).reshape(
        rgb_image.shape[:-1]
    )
    labels = np.unique(labeled_image)
    images = split_image(rgb_image, labeled_image)

    terrain_list = []
    # Check if elements belong to the skybox
    for label_id, _im in zip(labels, images):
        if label_id in ignore_labels:
            continue
        mask = labeled_image == label_id
        depth_field = crop_to_mask(depth_map, mask)
        # Otherwise check if it is deep with a nice vertical decreasing gradient
        masked_image = depth_field * crop_to_mask(mask, mask)

        plt.imshow(depth_field)
        plt.show()
        plt.imshow(masked_image)
        plt.show()
        plt.imshow((masked_image[1:] - masked_image[:-1]))
        plt.show()
        slope = np.mean(
            (masked_image[1:] - masked_image[:-1])[crop_to_mask(mask, mask)[1:]]
        )
        print(label_id, "has slope", slope)
        if slope < 0.002:
            # Mark the label as terrain
            print(label_id, "is terrain")
            terrain_list.append(label_id)

    return terrain_list


def segment_parts():
    """Segment an image into skybox and terrain parts."""
    facebook_mask_former_labels = mask_former(Image.open(DEFAULT_IMAGE))
    skybox_indices = mask_skybox(
        DEFAULT_IMAGE, "../outputs/depth_map.png", facebook_mask_former_labels
    )
    terrain_indices = mask_terrain(
        DEFAULT_IMAGE,
        "../outputs/depth_map.png",
        facebook_mask_former_labels,
        skybox_indices,
    )
    print("terrain indices are ", terrain_indices)


def segment_skybox(segmentation, depth_map):
    """
    Probability for each segment to be a skybox part.

    :param dict segmentation: The segmentation data.
    :param np.ndarray depth_map: The depth of each pixel.

    :return torch.Tensor: Tensor of probability for each segment.
    """
    reduced_depth = force_monotonous(depth_map, bottom_to_top=False)
    mean_depth = np.mean(reduced_depth, axis=1)
    norm_mean_depth = (mean_depth - mean_depth.min()) / (
        mean_depth.max() - mean_depth.min()
    )

    y_indices = torch.linspace(torch.pi / 2, -torch.pi / 2, depth_map.shape[0])
    height_distribution = (torch.sin(y_indices) + 1) / 2
    segmented = segmentation["segmentation"]
    has_undefined = 0 in segmented
    masks = np.empty(
        (
            len(np.unique(segmented)) + has_undefined + 1,
            depth_map.shape[1],
            depth_map.shape[0],
        )
    )
    if has_undefined:
        masks[0] = segmented.T == 0
    sky_ids = get_sky_ids()
    sky_detected = []
    for i, info in enumerate(segmentation["segments_info"]):
        masks[i + has_undefined] = segmented.T == (i + has_undefined)
        if info["label_id"] in sky_ids:
            sky_detected.append(i + has_undefined)

    """
    # Just some visualization code (to delete)
    
    plt.plot(height_distribution, label="Probability following y")
    plt.plot(norm_mean_depth, label="Probability following mean depth")
    plt.plot(height_distribution * norm_mean_depth, label="Combined probability")
    plt.xlabel("Height in pixel coordinates (0 = image top)")
    plt.ylabel("Probability of being above the horizon")
    plt.title("Probability of an horizontal line to be above the horizon (on y coordinate)")
    plt.legend()
    plt.grid()
    plt.show()
    """

    sky_probability = torch.mean(
        height_distribution * norm_mean_depth * masks, axis=(1, 2)
    )
    return sky_probability


def get_skybox_mask(segmentation, depth_map, closest_plane=0.3, farthest_plane=0.7):
    """
    Return the skybox mask for a given image.

    :param dict segmentation: Panoptic segmentation from Mask2Former
    :param numpy.ndarray depth_map: Array of depth for each pixel.
    :param float closest_plane: Pixels closer to this plane cannot be a part of the skybox.
    :param float farthest_plane: Pixels above this plane are automatically a part of the skybox.
    :return numpy.ndarray: A binary mask of the same size as the input image.
    """
    sky_probability = segment_skybox(segmentation, depth_map)
    # Use a threshold or at least one element
    threshold = min(0.5, torch.max(sky_probability))
    passing_sky = torch.argwhere(sky_probability >= threshold)
    masks = [segmentation["segmentation"] == i for i in passing_sky]
    # Far plane has to be a part of the skybox
    far_plane = torch.from_numpy(depth_map > farthest_plane)
    masks.append(far_plane)
    composite_mask = torch.logical_or(*masks)
    # Near plane cannot be a part of the skybox
    not_near_plane = torch.from_numpy(depth_map > closest_plane)
    return torch.logical_and(composite_mask, not_near_plane).numpy()


def force_monotonous(data, bottom_to_top=True):
    """Check area where depth coordinate increase monotonously."""
    output = np.empty(data.shape)
    flipper = -bottom_to_top
    prog_depth = data[flipper]
    for i in range(data.shape[0] + flipper - 1):
        slicer_index = (-1 if bottom_to_top else 1) * (i + 1)
        prog_depth = np.max([data[slicer_index], prog_depth], axis=0)
        output[slicer_index] = prog_depth

    return output


def increasing_depth(monotonous_depth):
    """
    Enhance the monotonous depth map by rewriting all points with positive of null gradient.

    The idea is to get a strictly monotonous map that would be similar
    to the natural view direction.

    :param numpy.ndarray monotonous_depth: Depth of monotonous progression
    :return numpy.ndarray: A natural progression of the depth.
    """
    grad = np.gradient(monotonous_depth, axis=0)
    # Skip indices of null gradient
    corrupt_depth = np.copy(monotonous_depth)
    corrupt_depth[grad >= 0] = np.nan
    median_depths = np.nanmedian(corrupt_depth, axis=1)
    # Replace by average depth
    corrupt_depth[np.isnan(corrupt_depth)] = np.tile(
        median_depths, (monotonous_depth.shape[1], 1)
    ).T[np.isnan(corrupt_depth)]
    return force_monotonous(corrupt_depth)


def get_ground_mask(depth_map, ground_mask=None):
    """
    Identify and segment the ground in a given depth map.

    This function uses a combination of depth gradient, monotonicity,
    and clustering to identify the ground.
    It then applies a Gaussian filter to smoothen the result and
    DBSCAN to further segment the ground.

    :param numpy.ndarray depth_map: The depth map of the image.
    :param numpy.ndarray | None ground_mask: A mask indicating the initial ground pixels.

    :return numpy.ndarray: A binary mask indicating the ground pixels.
    """
    monotonous_depth = force_monotonous(depth_map)
    corrupt_depth = increasing_depth(monotonous_depth)
    adherence = corrupt_depth - monotonous_depth
    grad = np.gradient(monotonous_depth, axis=0)
    great_map = np.logical_and((adherence * corrupt_depth) < 0.1, grad < 0).astype(np.float32)
    conv = gaussian_filter(great_map, sigma=10)
    zones = conv > 0.5
    # Now select which are is a part of the ground
    points = np.argwhere(zones)
    clustering = DBSCAN(eps=50, min_samples=2000).fit(points)

    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")

    clustered = -np.ones(depth_map.shape)
    clustered[points[:, 0], points[:, 1]] = labels

    new_ground_mask = clustered >= 0
    if ground_mask is not None:
        new_ground_mask = np.logical_and(new_ground_mask, ground_mask)

    return new_ground_mask


def segments_objects(depth_map, mask=None):
    """
    Segment the objects in the given depth map using DBSCAN clustering.

    :param numpy.ndarray depth_map: The depth map of the image.
    :param numpy.ndarray mask: A mask indicating the points to be considered. Defaults to None.

    :return numpy.ndarray: The clustered labels for the objects in the depth map.
    """
    if mask is None:
        remaining_indices = planar_grid(depth_map)
    else:
        remaining_indices = np.argwhere(mask)
    remaining_deep_points = (
        depth_map[remaining_indices[:, 0],
        remaining_indices[:, 1]].reshape(-1, 1)
    )

    points = np.hstack((remaining_indices / 1024, scale(remaining_deep_points)))
    clustering = DBSCAN(eps=0.1, min_samples=2000).fit(points)

    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of object clusters: {n_clusters_}")
    print(f"Estimated number of objects noise points: {n_noise_}")

    clustered = -np.ones(depth_map.shape)
    clustered[np.uint64(points[:, 0] * 1024), np.uint64(points[:, 1] * 1024)] = labels
    return clustered


def get_horizon_level(depth_map, sky_mask):
    """Return the horizon line level."""
    up = force_monotonous(depth_map)
    down = force_monotonous(depth_map, False)

    horizon_mask = np.logical_or(up == down, sky_mask)
    horizon_line = np.argmin(horizon_mask, axis=0)
    plt.imshow(horizon_mask)
    plt.show()
    return np.mean(horizon_line)


def segment_anything(image, depth_map):
    """
    Segment the image into skybox, ground, and objects using depth map and panoptic segmentation.

    :param PIL.Image.Image image: The input image.
    :param numpy.ndarray depth_map: The depth map of the image.

    :return: A combined mask: 0 for unidentified, 1 for sky, 2 for ground, >=3 for objects.
    :rtype: numpy.ndarray
    """
    segmentation = panoptic_segmentation(image)[0]
    sky_mask = get_skybox_mask(segmentation, depth_map)
    ground_mask = get_ground_mask(depth_map, np.logical_not(sky_mask))

    object_clusters = segments_objects(
        depth_map, np.logical_not(np.logical_or(sky_mask, ground_mask))
    )

    return sky_mask, ground_mask, object_clusters


def segmentation_demo(image_path, depth_map_path):
    """
    Demonstrate the segmentation process on an image using a depth map.

    :param str image_path: The path to the input image file.
    :param str depth_map_path: The path to the depth map file.
    """
    with open(depth_map_path, "rb") as file:
        depth_map = np.load(file)
    image = Image.open(image_path)
    clusters = segment_anything(image, depth_map)
    masks_aggregation = clusters[0] + clusters[1] * 2 + (clusters[2] + 3) * (clusters[2] >= 0)
    Image.fromarray(masks_aggregation == 2).show()
    plt.imshow(image)
    plt.imshow(masks_aggregation, alpha=0.7)
    plt.show()


if __name__ == "__main__":
    """
    Different segmentation techniques
    
    compare_segmentations(DEFAULT_IMAGE, '../outputs/depth_map.png')
    segmentation_maps(DEFAULT_IMAGE, '../outputs/depth_map.png')
    image_segments = segment_and_save(DEFAULT_IMAGE, '../outputs/depth_map.png')
    segment_parts()
    prepare_ground_mask("../forest.png", "depth.npy", "mask.npy")
    """
    segmentation_demo(DEFAULT_IMAGE, "depth.npy")
