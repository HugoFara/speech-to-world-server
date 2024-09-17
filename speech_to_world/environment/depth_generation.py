"""
Generate an RGBD image from a simgle image.
"""

from PIL import Image
from diffusers import MarigoldDepthPipeline
import matplotlib.pyplot as plt
import numpy as np


def get_depth_pipeline():
    """Main pipeline for the depth model."""
    return MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-v1-0",
        custom_pipeline="marigold_depth_estimation",
    )


def compute_image_depth(image, color_map="Spectral"):
    """
    Compute the depth of the image.

    :param PIL.Image.Image image: Input RGB image path.
    :param str | None color_map: Colorize depth image, set to None to skip colormap generation.
    :return: Pipeline
    """
    # Original DDIM version (higher quality)
    pipe = get_depth_pipeline()
    # Note: a 16-bit variant is also available, just use torch_dtype=torch.float16, variant="fp16"

    pipe.to("cuda")

    return pipe(
        image,
        # (optional) Maximum resolution of processing. If set to 0: will not resize at all.
        # Defaults to 768.
        # processing_res=768,
        # (optional) Resize depth prediction to match input resolution.
        # match_input_res=True,
        # (optional) Inference batch size, no bigger than `num_ensemble`.
        # If set to 0, the script will automatically decide the proper batch size. Defaults to 0.
        # batch_size=0,
        # (optional) Random seed can be set to ensure additional reproducibility.
        # Default: None (unseeded).
        # Note: forcing --batch_size 1 helps to increase reproducibility.
        # To ensure full reproducibility, deterministic mode needs to be used.
        # seed=2024,
        # (optional) Colormap used to colorize the depth map. Defaults to "Spectral".
        # Set to `None` to skip colormap generation.
        color_map=color_map,
        # (optional) If true, will show progress bars of the inference progress.
        show_progress_bar=False,
    )


def get_depth(image):
    """Return a depth map of the image."""
    pipeline_output = compute_image_depth(image, color_map=None)
    return pipeline_output.depth_np


def get_depth_image(image, depth_map_path=None, color_map_path=None):
    """
    Return the colored depth image, save both grey and colored depth.

    :param PIL.Image.Image image: Input RGB image path.
    :param depth_map_path: Path to the depth map if it should be saved as a file
    :type depth_map_path: str or None
    :param color_map_path: Path to the colored depth map if it should be saved as a file
    :type color_map_path: str or None
    :return np.ndarray: Depth map, between 0 and 1
    """
    pipeline_output = compute_image_depth(image)
    # Predicted depth map
    depth = pipeline_output.depth_np

    if depth_map_path is not None:
        # Save as uint16 PNG
        depth_uint16 = (depth * (2**16 - 1)).astype(np.uint16)
        grey_depth_image = Image.fromarray(depth_uint16)
        grey_depth_image.save(depth_map_path, mode="I;16")

    if color_map_path is not None:
        # Colorized prediction
        depth_colored: Image.Image = pipeline_output.depth_colored
        # Save colorized depth map
        depth_colored.save(color_map_path)
    return depth


def plot_arrays(array1, array2, titles=None):
    """
    Plot two matrix arrays as images.

    Create a figure with two subplots and plots the given matrix arrays as grayscale images.
    If the `titles` parameter is provided, it sets the titles for the two plots.

    :param array1: The first matrix array to be plotted.
    :type array1: numpy.ndarray

    :param array2: The second matrix array to be plotted.
    :type array2: numpy.ndarray

    :param titles: Optional titles for the two plotted images.
    :type titles: tuple, default is None
    """
    # Create a figure and grid objects
    _fig, axes = plt.subplots(1, 2)

    # Plot the arrays as an images
    axes[0].imshow(array1, cmap="gray")
    axes[1].imshow(array2, cmap="gray")

    if titles is not None:
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])
    plt.show()


def view_flat_estimation(rgbd_image):
    """
    Plot the color and depth components of an RGBD image.

    :param rgbd_image: An RGBD Image containing the color and depth components of an image.
    :type rgbd_image: open3d.geometry.RGBDImage
    """
    plot_arrays(
        rgbd_image.color,
        rgbd_image.depth,
        ["Mountain grayscale image", "Mountain depth image"],
    )


def get_horizon_height(depth_map):
    """Return the height of the horizon line in pixel coordinates."""
    average_depth = np.median(depth_map, axis=1)
    return np.argmax(average_depth)


def main(image=None):
    """
    Main demo function for depth generation.

    :param PIL.Image.Image | None image: The image to generate depth from.
    """
    depth_map_path = "outputs/" + ("sunny_" if image is None else "") + "depth_map.png"
    color_map_path = (
        "outputs/" + ("sunny_" if image is None else "") + "depth_colored.png"
    )
    if image is None:
        image = Image.open("../sunny_mountain.png")
    get_depth_image(image, depth_map_path, color_map_path)


if __name__ == "__main__":
    main(Image.open("../sunny_mountain.png"))
