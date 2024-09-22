"""
Generate a 2.5D view of a 2D image.

"""
import os
import importlib
import warnings

import numpy as np
from PIL import Image
import scipy
from skimage.restoration.inpaint import inpaint_biharmonic

from ..skybox.inpainting import inpaint_image
from .depth_generation import get_depth
from . import image_segmentation as im_seg
from .depth_inpainting import inpaint_depth_controlled

# Parent directory,
PARENT_DIR = os.path.abspath(os.path.join(importlib.util.find_spec(__name__).origin, os.pardir))
# Generated files folder
OUTPUTS_FOLDER = os.path.join(PARENT_DIR, "outputs")


def moving_average(data, window_size=5):
    """Moving average over the given input data."""
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def segment_stuff(image_path, depth_path=None):
    """
    This function generates a mask of the skybox in an input image.

    :param str image_path: The path to the input image.
    :param str depth_path: The path to the input depth map. If not provided, it will be generated.

    :return: A binary mask of the skybox in the input image.
    :rtype: numpy.ndarray

    Example usage:

    ```python
    skybox_mask = mask_skybox("../sunny_mountain.png", "sunny_depth_map.npy")
    plt.imshow(skybox_mask)
    plt.show()
    ```
    """
    image = Image.open(image_path)
    if depth_path is None:
        depth_map = get_depth(image)
    elif depth_path.endswith(".npy"):
        depth_map = np.load(depth_path)
    else:
        # Image im I;16 format with depth on 16 bits.
        depth_map = np.asarray(Image.open(depth_path)) / (2**16 - 1)
        np.save(OUTPUTS_FOLDER + "/depth.npy", depth_map)

    return im_seg.segment_anything(image, depth_map)


def mask_image(image, mask):
    """Extract the part of an image that matches the given mask."""
    indices = np.argwhere(mask)
    pixels = np.zeros_like(image)
    pixels[indices[:, 0], indices[:, 1]] = image[indices[:, 0], indices[:, 1]]
    return pixels


def filling_strategy(image_np, large_mask):
    """Create a very large image with many mirrored views."""
    cropped = im_seg.crop_to_mask(image_np, large_mask)
    # Fill holes

    # Map in big size
    large_skybox = np.empty(
        (cropped.shape[0] * 3, cropped.shape[1] * 3, cropped.shape[2])
    )
    elem = cropped[:, ::-1]
    for i in range(3):
        elem = elem[::-1]
        for j in range(3):
            if j != 0:
                elem = elem[:, ::-1]
            large_skybox[
                i * cropped.shape[0] : (i + 1) * cropped.shape[0],
                j * cropped.shape[1] : (j + 1) * cropped.shape[1],
            ] = elem
    Image.fromarray(large_skybox.astype(np.uint8)).show()

    # Fill holes
    # For testing: return large_skybox

    raise NotImplementedError("This function is not finished.")


def enlarge_mask(initial_mask, iterations=20):
    """Enlarge an input mask by applying a binary dilatation repetitively."""
    large_mask = initial_mask
    for _ in range(iterations):
        large_mask = scipy.ndimage.binary_dilation(large_mask)
    return large_mask


def inpaint_skybox(image_np, skybox):
    """Apply inpainting to complete the skybox."""
    large_mask = enlarge_mask(np.logical_not(skybox))
    inpainted_skybox = Image.fromarray(
        (inpaint_biharmonic(image_np, large_mask, channel_axis=-1) * 255).astype(
            np.uint8
        )
    )
    inpainted_skybox.show()
    complete_skybox = inpaint_image(
        "the sky", inpainted_skybox, Image.fromarray(large_mask), num_inference_steps=50
    )[0]
    return complete_skybox


def inpaint_ground(image, image_np, depth_map, ground, filling_mask):
    """Apply inpainting to complete the ground."""
    ground_segment = mask_image(image_np, ground)

    stretched_ground = (
        Image.fromarray(ground_segment)
        .resize((image.width * 5, image.height))
        .crop((image.width * 2, 0, image.width * 3, image.height))
    )

    inpainted_ground = Image.fromarray(ground_segment)
    inpainted_ground.paste(stretched_ground)
    inpainted_ground.paste(
        Image.fromarray(ground_segment), mask=Image.fromarray(ground)
    )

    print("Completing ground, inpainting")
    complete_ground = inpaint_image("the ground", inpainted_ground, filling_mask)[0]
    complete_ground.show()
    print("Completing ground with controlnet")
    ground_depth = 1 - im_seg.increasing_depth(im_seg.force_monotonous(depth_map))
    complete_ground_controlled = inpaint_depth_controlled(
        inpainted_ground,
        filling_mask,
        Image.fromarray(ground_depth),
        "the ground",
    )[0]
    return complete_ground_controlled


def complete_segments(image, depth_map, skybox, ground, objects):
    """
    Process the image parts.

    - Sky: Reshape image size, inpaint holes.
    - Ground: Inpaint holes, reshape rectangle.
    - Objects: Store depth, normal map (?).

    :param PIL.Image.Image image: The initial image
    :param numpy.ndarray depth_map: The depth map
    :param PIL.Image.Image skybox: The masked skybox
    :param PIL.Image.Image ground: The masked ground
    :param list[PIL.Image.Image] objects: Each masked object

    :return: The completed segments.
    :rtype: tuple[PIL.Image.Image, PIL.Image.Image, list[tuple[PIL.Image.Image, numpy.ndarray]]]
    """
    image_np = np.asarray(image)

    # Complete the skybox
    complete_skybox = inpaint_skybox(image_np, skybox)
    complete_skybox.show()

    # Complete the terrain
    filling_mask = np.logical_and(
        np.logical_not(ground), np.logical_not(skybox), objects >= 0
    )
    filling_mask = enlarge_mask(filling_mask)
    complete_ground = inpaint_ground(
        image, image_np, depth_map, ground, Image.fromarray(filling_mask)
    )
    complete_ground.show()

    objects_data = []
    # Save the objects
    for i in range(int(np.max(objects)) + 1):
        mask = objects == i
        cropping = im_seg.crop_to_mask(image_np, mask)
        depth = im_seg.crop_to_mask(depth_map, mask)
        # Handle occlusions
        objects_data.append((cropping, depth))
    warnings.warn("Objects occlusions cannot be handled yet.")

    return complete_skybox, complete_ground, objects_data


if __name__ == "__main__":
    """
    # Skybox mask only
    # skybox_mask = im_seg.mask_skybox("../sunny_mountain.png", OUTPUTS_FOLDER + "/depth_map.png")

    # To regenerate data
    skybox, ground, objects = segment_stuff(
        os.path.join(PARENT_DIR, '../sunny_mountain.png'),
        os.path.join(PARENT_DIR, 'sunny_depth_map.png'),
    )

    # Save the data to files
    np.save(OUTPUTS_FOLDER + "/skybox.npy", skybox)
    np.save(OUTPUTS_FOLDER + "/ground.npy", ground)
    np.save(OUTPUTS_FOLDER + "/objects.npy", objects)
    """

    SKYBOX = np.load(OUTPUTS_FOLDER + "/skybox.npy")
    GROUND = np.load(OUTPUTS_FOLDER + "/ground.npy")
    OBJECTS = np.load(OUTPUTS_FOLDER + "/objects.npy")

    complete_segments(
        Image.open(os.path.join(PARENT_DIR, '../sunny_mountain.png')),
        np.asarray(Image.open(os.path.join(PARENT_DIR, 'sunny_depth_map.png'))) / (2**16 - 1),
        SKYBOX,
        GROUND,
        OBJECTS,
    )

    # np.save("mask.npy", skybox_mask)
