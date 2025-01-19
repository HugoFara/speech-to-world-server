"""
Create mask for an image so that it can be locally inpainted.
"""

import enum

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

from . import image_processing

RANDOM_SEED = 42

FRAME_CONFIG = {
    "width": 2048 + 256,
    "height": 1024,
    # Initial image to map
    "base_image": {"width": 2048, "height": 1024},
    # Extension of the image to make a cylinder
    "horizontal_extensions": {"width": 256, "display": "both"},
    # Sky extension to make a hemisphere
    "top_extension": {"radius": 1024},
}


class ExtensionFilling(enum.Enum):
    """
    What type of filling to apply to an extended image.

    STRETCH: stretch the image border.
    MEAN: use th mean value of the image border.
    GRADIENT: apply a continuous gradient from the mean value of the image border.
    SMART: auto-extension depending on context.
    """
    STRETCH = 1
    MEAN = 2
    GRADIENT = 3
    SMART = 4


# Horizontal panorama zone

def central_vertical_mask(image, center_width):
    """
    Create a vertical mask for the edition zone.

    The mask will be in centered.

    :param PIL.Image.Image image: Input image
    :param int center_width: Width of the vertical mask.
    """
    mask = Image.new("L", image.size, "black")

    mask.paste(
        Image.new("L", (center_width, image.height), "white"),
        (image.width // 2 - center_width // 2, 0),
    )

    return mask


def central_circular_mask(canvas_size, inner_radius=None):
    """
    Create a circular mask with a specified canvas size and inner radius.

    :param int canvas_size: The width or height of the output square image.
    :param inner_radius: The radius of the inner circle. If None, a full circle is created.
    :type inner_radius: int | None

    :return PIL.Image.Image: A black square image with a white circle in the center.
    """
    mask = Image.new("L", (canvas_size, canvas_size), color="black")
    # Draw a white circle on the image
    if inner_radius is not None:
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (
                canvas_size / 2 - inner_radius,
                canvas_size / 2 - inner_radius,
                canvas_size / 2 + inner_radius,
                canvas_size / 2 + inner_radius,
            ),
            fill="white",
        )

    return mask


def add_top_mask(img, radius):
    """Add a mask on the top of the image."""
    # Get the width and height of the two images
    width, height = img.size

    # Calculate the width and height of the new image
    new_height = height + radius * 2

    # Create a new blank image with white background
    new_image = Image.new("RGB", (width, new_height), "white")
    mask = Image.new("L", (width, new_height), "white")

    # Paste the first image onto the new image at position (0, 0)
    new_image.paste(img, (0, radius * 2))
    mask.paste(Image.new("L", (width, new_height), color="black"), (0, radius * 2))

    return new_image, mask


def add_center_image(background, box_size):
    """Add a white square to the center of an image."""
    # Create a white square image
    size = (box_size, box_size)
    white_square = Image.new("RGB", size, "white")
    # Find the center of the second image
    width, height = background.size
    center = (width // 2, height // 2)

    # Calculate the top-left corner of the white square image
    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)

    # Paste the white square image onto the second image
    background.paste(white_square, top_left)
    return white_square


def draw_top_image(background, radius, base_height):
    """Draw a white circle in the top image."""
    draw = ImageDraw.Draw(background)

    # Add a white box in the center of the square
    center_x, center_y = background.size[0] // 2, background.size[1] // 2

    circle_center = (center_x, center_y - base_height // 2 - radius)
    draw.ellipse(
        [
            circle_center[0] - radius,
            circle_center[1] - radius,
            circle_center[0] + radius,
            circle_center[1] + radius,
        ],
        fill="white",
    )


# Completion functions


def draw_masks(base_file):
    """
    Draw a mask for inpainting from a base image.

    The image is drawn upon, instead of creating a new image.
    """
    frame_object = FRAME_CONFIG
    # Create a black square image of size 2048x1024
    img = Image.new(
        "RGB", (frame_object["width"], frame_object["height"]), color="black"
    )

    base_image = Image.open(base_file)
    base_image.thumbnail(
        (frame_object["base_image"]["width"], frame_object["base_image"]["height"])
    )

    left_img, right_img = image_processing.split_base_image(base_image)

    add_center_image(img, frame_object["base_image"]["width"])

    # Add borders
    image_processing.paste_borders(img, left_img, right_img)

    # Add a circle on top of the box
    draw_top_image(
        img,
        frame_object["top_extension"]["radius"],
        frame_object["base_image"]["height"],
    )

    img.show()


def horizontal_tiling_mask(img, frame_object=None):
    """
    Create an image with masks so that an IA can complete it.

    :param PIL.Image.Image img: Image to apply masks to
    :param frame_object: Frame configuration object,
    containing information about the image dimensions and extensions.
    :type frame_object: dict
    :return: A tuple containing the image with masks applied and the corresponding mask.
    """
    if frame_object is None:
        frame_object = FRAME_CONFIG

    inpaint_canvas = image_processing.flip_image_sides(img)
    mask = central_vertical_mask(
        img, frame_object["width"] - frame_object["base_image"]["width"]
    )

    return inpaint_canvas, mask


def create_gradient_mask(width, height, is_horizontal=True):
    """
    Create a gradient mask of specified dimensions.

    :param int width: Width of the mask
    :param int height: Height of the mask
    :param bool is_horizontal: If True, the gradient will be horizontal.
    Otherwise, it will be vertical.

    :return: A mask image with a gradient fill.
    """
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)

    if is_horizontal:
        for i in range(width):
            draw.line([(i, 0), (i, height)], fill=int(255 * (i / width)))
    else:
        for i in range(height):
            draw.line([(0, i), (width, i)], fill=int(255 * (i / height)))

    return mask


def gradient_fill(img, size):
    """
    Create a gradient mask of specified dimensions and apply it to the input image.

    The background color is chosen as an average of the 10% brighter pixels from the top
    of the image and the K-mean group with the least contrast.

    The best solution is probably to use the deepest pixels.

    :param Image.Image img: Input image to apply the gradient mask to.
    :param int size: Height of the gradient mask.

    :return: A new image with the input image's content blended with a gradient mask.
    """
    background = Image.new("RGBA", (img.width, size), color="white")
    blend_mask = create_gradient_mask(background.width, background.height, False)

    # Take only the 10% brighter pixels
    pixels_array = np.asarray(img.convert("L"))
    threshold = np.quantile(pixels_array, 0.9)
    valid_indices = np.argwhere(pixels_array > threshold)
    selection = np.asarray(img)[valid_indices[:, 0], valid_indices[:, 1]]
    mean_pixel_value = np.mean(selection, axis=0).astype(np.uint8)

    # Alternative path: use pixels with less contrast
    pixels_stack = np.vstack(np.asarray(img))
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED).fit(pixels_stack)
    dispersions = np.empty(n_clusters)
    for label in range(n_clusters):
        indices = np.argwhere(kmeans.labels_ == label)
        dispersions[label] = np.sum(
            (kmeans.cluster_centers_[label] - pixels_stack[indices]) ** 2
        ) / len(indices)
    mean_pixel_value2 = kmeans.cluster_centers_[np.argmin(dispersions)]

    mean_pixel_value = (mean_pixel_value + mean_pixel_value2) / 2

    foreground = Image.fromarray(
        np.full((size, img.width, 3), mean_pixel_value).astype(np.uint8)
    ).convert("RGBA")
    blended_image = Image.composite(foreground, background, blend_mask)

    return blended_image


def add_top_frame(img, size, border_size=10, extension_filling=ExtensionFilling.MEAN):
    """
    Add a new frame on top of the current image.

    :param Image.Image img: Input image to apply the gradient mask to.
    :param int size: Height of the gradient mask.
    :param int border_size: Size of the border to be added to the top of the image.
    :param extension_filling: Method to fill the top zone with a crop from the original image.
    :type extension_filling: ExtensionFilling

    :return: A new image with the input image's content blended with a gradient mask.
    """
    canvas_size = img.width, img.height + size
    # Create a white background image
    new_img = Image.new("RGB", canvas_size, color="white")
    mask = Image.new("L", canvas_size, color="white")

    new_img.paste(img, (0, size))
    # Fill the top zone with a crop from the original image
    cropped = img.crop((0, 0, img.width, border_size))
    if extension_filling == ExtensionFilling.GRADIENT:
        blended_image = gradient_fill(cropped, size)
        new_img.paste(blended_image)
    elif extension_filling == ExtensionFilling.STRETCH:
        new_img.paste(cropped.resize((cropped.width, size)))
    else:
        arr = np.asarray(cropped)
        mean_pixel_value = np.mean(arr.reshape(-1, 3), axis=0).astype(np.uint8)
        averaged_img = Image.fromarray(
            np.full((size, cropped.width, 3), mean_pixel_value)
        )
        new_img.paste(averaged_img)

    mask.paste(
        Image.new("L", (img.width, img.height - border_size), color="black"),
        (0, size + border_size),
    )
    return new_img, mask


def create_top_mask(img, sky_size=None, extension_filling=ExtensionFilling.MEAN):
    """
    Create a mask to the top of the image.

    :param Image.Image img: Input image to apply the gradient mask to.
    :param int sky_size: Height of the gradient mask.
    If None, it will be set to the default value in the FRAME_CONFIG dictionary.
    :param extension_filling: Method to fill the top zone with a crop from the original image.
    Default is ExtensionFilling.MEAN.
    :type extension_filling: ExtensionFilling

    :return: A tuple containing the new image with the input image's
    content blended with a gradient mask and the corresponding mask.
    :rtype: tuple[PIL.Image.Image, PIL.Image.Image]
    """
    if sky_size is None:
        sky_size = FRAME_CONFIG["top_extension"]["radius"]
    return add_top_frame(img, sky_size, extension_filling=extension_filling)


def display_masks(base_file):
    """Display the image with the mask applied."""
    base_image = Image.open(base_file)
    img, _ = horizontal_tiling_mask(base_image, FRAME_CONFIG)
    img.show("Central mask applied")
    img, _ = create_top_mask(base_image)
    img.show("Top mask applied")


if __name__ == "__main__":
    display_masks("../sunny_mountain.png")
