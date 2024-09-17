"""
Various image edition functions.
"""

import itertools

from PIL import Image, ImageDraw
import numpy as np


def split_base_image(img):
    """Split an image in two and return left and right parts."""
    width, height = img.size
    position = width // 2
    left_image = img.crop((0, 0, position, height))
    right_image = img.crop((position, 0, width, height))
    return left_image, right_image


def flip_image_sides(img):
    """
    Take an input image, split it in the middle and flip both parts.

    :param PIL.Image.Image img: Base input image (won't be changed)
    :return PIL.Image.Image: Image with the same dimension but parts flipped
    """
    left_img, right_img = split_base_image(img)

    n_right_img = left_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    n_left_img = right_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    out_image = Image.new(img.mode, img.size)
    out_image.paste(n_right_img, (0, 0))
    out_image.paste(n_left_img, (n_right_img.width, 0))
    return out_image


def paste_borders(background, left_img, right_img):
    """Paste the borders onto an image."""
    size = background.size

    n_right_img = left_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    n_left_img = right_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # Find the center of the second image
    width, height = background.size
    center = (width // 2, height // 2)

    # Calculate the top-left corner of the white square image
    top_left = (center[0] - size[0] // 2, center[1] - size[1] // 2)
    top_right = (center[0] + size[0] // 2, center[1] + size[1] // 2)

    # Paste the images onto the background image
    background.paste(n_right_img, top_left)
    background.paste(n_left_img, top_right)


def concatenate_borders(left_image, right_image):
    """
    Create a new image with the border added, return the image.

    :param PIL.Image.Image left_image: Image to concatenate on the left
    :param PIL.Image.Image right_image: Image to concatenate on the right
    """
    # Get the width and height of the two images
    width1, height1 = left_image.size
    width2, height2 = right_image.size

    # Calculate the width and height of the new image
    new_width = width1 + width2
    new_height = max(height1, height2)

    # Create a new blank image with white background
    new_image = Image.new("RGB", (new_width, new_height))

    # Paste the first image onto the new image at position (0, 0)
    new_image.paste(left_image, (0, 0))

    # Paste the second image onto the new image at position (width1 + center_width, 0)
    new_image.paste(right_image, (width1, 0))

    return new_image


def horizontal_carrousel(base_image, left_translation):
    """
    Crop the image at a specific horizontal point to do a carrousel.

    The right side of the image will be sent on the left like in Pacman.

    :param PIL.Image.Image base_image: Image to carrousel.
    :param int left_translation: Number of pixels to translate the image by.
    A negative value translates to the left.

    :return PIL.Image.Image: New image translated.
    """
    if left_translation < 0:
        left_translation = base_image.width + left_translation
    left_image = base_image.crop((0, 0, left_translation, base_image.height))
    right_image = base_image.crop((left_translation, 0, base_image.width, base_image.height))
    output_image = base_image.copy()
    output_image.paste(right_image)
    output_image.paste(left_image, (base_image.width - left_translation, 0))
    return output_image


def box_mean_color(img, box):
    """
    Get the mean pixel value of a portion of image.

    :param PIL.Image.Image img: Image to take pixels from
    :param box: Box delimitation in format (left, top, right, bottom)
    :type box: tuple[int, int, int, int]

    :return tuple[int, int, int]: mean pixel color
    """
    diffs = box[2] - box[0], box[3] - box[1]
    pixels = img.crop(box).load()
    average_sky = np.mean(
        [pixels[pos] for pos in itertools.product(range(diffs[0]), range(diffs[1]))],
        axis=0,
    )
    return tuple(map(int, average_sky))


def draw_gradient_box(img, position, size, start_color, end_color):
    """
    Draw a box as a gradient between two points.

    :param PIL.Image.Image img: Base image to draw rectangle on
    :param tuple[int, int] position: top-left corner where to start the box
    :param tuple[int, int] size: Size of the box to draw
    :param tuple[int, int, int] start_color: Color at the beginning of the box
    :param tuple[int, int, int] end_color: Color at the end of the box
    """
    draw = ImageDraw.Draw(img)

    x, y = position
    width, height = size
    for i in range(width):
        color = [
            int(start_color[c] + (end_color[c] - start_color[c]) * i / width)
            for c in range(3)
        ]
        draw.line([(x + i, y), (x + i, y + height)], tuple(color))


# 2D polar geometry functions


def cartesian_to_polar(pos, origin=(0, 0)):
    """
    Polar coordinates from cartesian one.

    :param tuple[float, float] pos: (x, y) position in cartesian coordinates
    :param tuple[float, float] origin: Relative to
    :return tuple[float, float]: Radius and angle
    """
    vector = np.array(pos) - origin
    return np.linalg.norm(vector), np.arctan2(vector[1], vector[0])


def cartesian_to_polar_batch(pos, origin=(0, 0)):
    """
    Polar coordinates from cartesian one, applied on a batched of positions.

    :param numpy.ndarray | tuple | list pos: (x, y) positions in cartesian coordinates
    :param tuple[float, float] | float origin: Relative to
    :return numpy.ndarray: Array of radii and angles
    """
    vector = np.array(pos) - origin
    return np.dstack(
        (np.linalg.norm(vector, axis=1), np.arctan2(vector[:, 1], vector[:, 0]))
    )[0]


def polar_to_cartesian(pos, origin=(0, 0)):
    """
    Convert from polar coordinates to cartesian ones.

    :param tuple[float, float] pos: (radius, angle) position in polar coordinates
    :param tuple[float, float] origin: Cartesian axis origin
    :return tuple[float, float]: Position as (x, y)
    """
    return origin + pos[0] * np.array([np.cos(pos[1]), np.sin(pos[1])])


# Image distortion


def distort_image(img, inner_radius=None):
    """
    Create an image distorted to fit on a circle.

    With an initial image of dimensions (width, height),
    the new image has dimensions ((height + inner_radius) * 2, (height + inner_radius) * 2).
    All modified pixels are in a circle of radius height.

    :param PIL.Image.Image img: Base image.
    :param int | None inner_radius: Radius of a white circle to add (optional)
    :return PIL.Image.Image: New image in a circle.
    """
    # New image dimensions
    canvas_size = img.height * 2 + (inner_radius or 0) * 2

    # Get the polar coordinate of each pixel in the new image, format [[radius, angle], ...]
    # Max radius is sqrt(2) * canvas_size / 2 (corners)
    grid = tuple(itertools.product(range(canvas_size), range(canvas_size)))
    polar_coordinates = cartesian_to_polar_batch(grid, canvas_size / 2)

    # Select the indices of the pixels that should be changed in the new image
    insiders = np.nonzero(
        np.logical_and(
            # Inside painted circle
            polar_coordinates[:, 0] < canvas_size / 2,
            # And outside mask
            polar_coordinates[:, 0] >= (inner_radius or 0),
        )
    )[0]

    # Acquire position on base image, cast to [0, img.size - 1]
    adapted_pos = (
        polar_coordinates[insiders]
        * (np.array(img.size[::-1]) - 1)
        / (canvas_size / 2, 2 * np.pi)
    )
    adapted_pos[:, 1] += (img.width - 1) / 2

    # Round to int and swap the last dimensions (return to image format)
    slicer = np.round(adapted_pos).astype(np.uint16)
    new_pixels = np.asarray(img)[slicer[:, 0], slicer[:, 1]]

    new_img_data = np.zeros((canvas_size * canvas_size, 3), dtype=np.uint8)
    # Assign to the new image
    new_img_data[insiders] = new_pixels

    return Image.fromarray(new_img_data.reshape(canvas_size, canvas_size, 3))


def unroll_top_image(img, width=None):
    """
    Unroll a polar projected image to standard format.

    Take an image fitting in a circle, and unrolls it.
    """
    canvas_size = (width or img.size[0] // 2), img.size[1] // 2
    # Create a white background image
    new_img = Image.new("RGB", canvas_size, color="black")

    for pos in itertools.product(range(new_img.width), range(new_img.height)):
        # Acquire polar coordinates corresponding to this position (radius, angle)
        adapted_pos = (
            pos[1] * img.height / 2 / new_img.height,
            pos[0] * 2 * np.pi / new_img.width,
        )
        # Position in the base image space
        adapted_pos = polar_to_cartesian(adapted_pos, (img.width / 2, img.height / 2))
        # Reduce to base image pixel space
        pixel_pos = int(adapted_pos[0]), int(adapted_pos[1])
        # print(pos, polar_pos, pixel_pos)
        new_img.putpixel(pos, img.getpixel(pixel_pos))

    return new_img


def image_polar_to_rect(img, width=None):
    """Take a polar (fisheye) image, and display it on a rectangle."""
    base_radius = img.size[1] // 2
    rect_size = (base_radius if width is None else width), base_radius
    new_img = Image.new("RGB", rect_size, "white")
    for radius in np.linspace(0, base_radius, base_radius - 1, endpoint=False):
        for theta in np.linspace(0, 2 * np.pi, img.size[0], endpoint=False):
            canvas_pos = (
                int(base_radius + radius * np.cos(theta)),
                int(base_radius + radius * np.sin(theta)),
            )
            new_img.putpixel(
                canvas_pos,
                img.getpixel(
                    (int(theta / (2 * np.pi) * (img.size[0] - 1)), int(radius))
                ),
            )
    new_img.show()
