"""
A Python script using Stable Diffusion and Inpainting to create panorama and skyboxes.
"""

from PIL import Image, ImageFilter, ImageDraw

import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

from .diffusion import generate_images
from .inpainting import inpaint_image
from . import mask_editor as me
from . import image_processing


def clamp(x, low=0, high=1):
    """
    Clamp a value between two extremes.

    :param float x: Value to clamp
    :param float low: Min value
    :param float high: Max value
    :return float: Clamped value
    """
    return max(min(x, high), low)


def equirectangular_projection(img):
    """
    Compute an equirectangular projection from a flat image.

    The formula to convert a set of coordinates (latitude, longitude) on a sphere to
    equirectangular projection is:

    x = (longitude + 180) * (image width / 360) y = (90 - latitude) * (image height / 180)

    But we won't be using such formula.

    :param PIL.Image.Image img: Input image
    :return PIL.Image.Image: Projected image.
    """
    width, height = img.size
    equirectangular_image = Image.new("RGB", (width, height), "white")

    # Convert each pixel in the equirectangular image
    for x in range(width):
        for y in range(height):
            v = y
            # [-1, 1]
            lon, lat = (x - width / 2) * 2 / width, (y - height / 2) * 2 / height
            u = x + (
                width / 2 * np.sin(lon * np.pi / 2) * 1 * (1 - np.cos(lat * np.pi / 2))
            )

            u, v = int(clamp(u, 0, width - 1)), int(clamp(v, 0, height - 1))

            # Map the pixel from the input image to the equirectangular image
            if (
                u >= width
                or v >= height
                or x >= equirectangular_image.size[0]
                or y >= equirectangular_image.size[1]
            ):
                continue
            equirectangular_image.putpixel((x, y), img.getpixel((int(u), int(v))))

    return equirectangular_image


def cylindrical_projection(img):
    """
    Compute a cylindrical projection from a flat image.

    The x-axis is preserved, by the y-axis will be changed.
    This is the inverse operation of a Lambert projection.

    :param PIL.Image.Image img: Input image
    :return PIL.Image.Image: Output image in cylindrical projection
    """
    image = pil_to_tensor(img)
    height, _width = image.shape[1:3]
    cylindrical_image = torch.empty(image.shape)

    # Convert each pixel in the equirectangular image, from [0, height] to [0, height]
    # As the view is essentially from a cylinder to a sphere, a cosine transformation is applied
    # We then apply a reverse cosine
    lines = height * (1 - torch.arccos(torch.linspace(-1, 1, height)) / torch.pi)
    ratios = lines - torch.round(lines)
    for y in range(height):
        v = int(lines[y].item())
        ratio = ratios[y]
        if v + 1 < height:
            interpolates = image[:, v + 1] * ratio + (1 - ratio) * image[:, v]
        else:
            interpolates = image[:, height - 1]

        cylindrical_image[:, y] = interpolates

    # Convert as a pillow image
    cylindrical_image = Image.fromarray(
        np.transpose(cylindrical_image.numpy(), (1, 2, 0)).astype("uint8")
    )
    return cylindrical_image


def horizontal_tiling(img):
    """
    Simple tiling function to view if an image can be tilled with itself.

    :param PIL.Image.Image img: Base image to tile.
    :return PIL.Image.Image: Horizontal concatenation of the base image.
    """
    width, height = img.size
    # Create a new image with twice the width
    new_image = Image.new("RGB", (width * 2, height))

    # Paste the original image twice
    new_image.paste(img, (0, 0))
    new_image.paste(img, (width, 0))

    return new_image


def blend_borders(img, size=10):
    """
    Blend the borders of an image to make them match. The new image is centered on the borders.

    :param PIL.Image.Image img: Input image.
    :param int size: Number of pixels to use

    :return PIL.Image.Image img: Auto-blended image.
    """

    width, height = img.size
    position = width // 2

    right_crop = img.crop((position, 0, width, height))

    translated = img.transform(
        img.size, Image.Transform.AFFINE, (1, 0, -position, 0, 1, 0)
    )
    translated.paste(right_crop, (0, 0))

    box = (width // 2 - size // 2, 0, width // 2 + size // 2, height)
    central_crop = translated.crop(box)

    central_crop = central_crop.filter(ImageFilter.SMOOTH)

    translated.paste(central_crop, box)

    return translated


def rewrite_image_borders(image, steps=20):
    """
    Inpaint the borders of an image to remove a seam line.

    :param PIL.Image.Image image: Initial image.
    :param int steps: Number of steps for inpainting.
    :return PIL.Image.Image: The inpainted image."""
    img, mask = me.horizontal_tiling_mask(image)
    inv_panorama = inpaint_image(
        "", img, mask, negative_prompt="a logo, a text", num_inference_steps=steps
    )[0]
    panorama = image_processing.flip_image_sides(inv_panorama)
    return panorama


def add_ground(base_image, steps, step_callback=None):
    """
    Add a ground to an image.

    The process is the following:
    1. The bottom part of the base image is selected, copied and stretched.
    2. The image is then distorted into a circle, centered on the lower part of the new image.
    3. An inpainting process is ran to redraw the ground.
    4. The image unrolled to the initial dimensions.

    :param PIL.Image.Image base_image: The input image to be extended as a ground.
    :param int steps: The number of inference steps for each inpainting process.
    :param step_callback: Optional callback function to be called after each inference step.
    :type step_callback: Callable | None
    :return PIL.Image.Image: The new ground of the image.
    """
    # Reverse the image, add a frame to the top part
    # 2048x256 image
    half_image = base_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).crop(
        (0, 0, base_image.width, base_image.height // 2)
    )
    # 2048x512
    img, _ = me.add_top_frame(
        half_image,
        half_image.height,
        half_image.height // 8,
        extension_filling=me.ExtensionFilling.STRETCH,
    )
    # Distort on the ground
    img = image_processing.distort_image(img)

    mask = Image.new("L", img.size, color="black")
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            half_image.height,
            half_image.height,
            img.height - half_image.height,
            img.height - half_image.height,
        ),
        fill="white",
    )
    img_with_ground = inpaint_image(
        "the ground seen from above, uniform color",
        img,
        mask,
        negative_prompt="a logo, a text, clouds, birds",
        num_inference_steps=steps,
        callback_on_step_end=step_callback,
    )[0]
    extended_ground = (
        # Unroll from (1024x1024) to (1024x512)
        image_processing.unroll_top_image(
            img_with_ground.transpose(Image.Transpose.ROTATE_270), base_image.width
        )
        .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        .transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    )

    # Stitch the new ground to the upper part without seam
    bottom_mask = linear_gradient_mask(
        (base_image.width, base_image.height // 2), extended_ground.height // 10
    ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    blend_mask = Image.new("L", extended_ground.size, "white")
    blend_mask.paste(bottom_mask)
    initial_ground_frame = Image.new(base_image.mode, extended_ground.size)
    initial_ground_frame.paste(
        base_image.crop((0, base_image.height // 2, base_image.width, base_image.height))
    )
    new_ground_frame = Image.new(base_image.mode, extended_ground.size)
    new_ground_frame.paste(extended_ground)
    ground_blend = Image.composite(new_ground_frame, initial_ground_frame, blend_mask)

    return ground_blend


def linear_gradient_mask(size, margin_height=10):
    """
    Create a gradient mask for an image that has a logistic curve shape.

    The mask is a grayscale image where the top half is darker and the bottom half is lighter.
    This is useful for creating a seamless transition between the top and bottom halves of an image.

    :param tuple[int, int] size: The size of the output mask.
    :param int margin_height: The height of the margin from the mask.
    :return: A grayscale image representing the gradient mask.
    """
    mask = Image.new("L", size)

    gradient = (
        Image
        .linear_gradient("L")
        .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        .resize((size[0], margin_height))
    )
    mask.paste(gradient)
    return mask


def sigmoid_gradient_mask(width, height, decay=50):
    """
    Create a gradient mask for an image that has a logistic curve shape.

    The mask is a grayscale image where the top half is darker and the bottom half is lighter.
    This is useful for creating a seamless transition between the top and bottom halves of an image.

    :param int width: The width of the output mask.
    :param int height: The height of the output mask.
    :param float decay: The speed at which the blending changes.
    :return: A grayscale image representing the gradient mask.
    """
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)

    indices = np.linspace(0, 1, height)

    # Logistic curve shape
    shades = 255 / (1 + np.exp(-decay * (indices - 0.5)))

    for i, shade in enumerate(shades):
        draw.line([(0, i), (width, i)], fill=int(shade))

    return mask


def add_sky(input_image, steps, step_callback=None):
    """
    Create a sky from the top half of the base image as a sky.

    The sky has the same dimensions as the base image.

    :param PIL.Image.Image input_image: The input image to be extended as a sky.
    :param int steps: The number of inference steps for each inpainting process.
    :param step_callback: Optional callback function to be called after each inference step.
    :type step_callback: Callable | None
    :return PIL.Image.Image: The final image with more sky.
    """
    # Base image is 2504x416, this is too much VRAM, need to reduce the size a bit
    context_height = input_image.height // 2
    half_sky = input_image.crop((0, 0, input_image.width, context_height))

    # Prepare the image that will receive an inpainting
    gradient_extended, mask = me.create_top_mask(
        half_sky, input_image.height, extension_filling=me.ExtensionFilling.GRADIENT
    )

    # Distort on a circle
    img = image_processing.distort_image(gradient_extended)

    mask = Image.new("L", img.size, color="black")
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            half_sky.height,
            half_sky.height,
            img.height - half_sky.height,
            img.height - half_sky.height,
        ),
        fill="white",
    )

    img_with_sky = inpaint_image(
        "the sky seen from below",
        img,
        mask,
        negative_prompt="a logo, a text, birds",
        num_inference_steps=steps,
        callback_on_step_end=step_callback,
    )[0]
    extended_sky = (
        # Unroll from (4:4) to (4:~1)
        image_processing.unroll_top_image(
            img_with_sky.transpose(Image.Transpose.ROTATE_270), input_image.width
        )
        .transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    )

    # Merge the original image and the extended version to get a seamless blend
    bottom_mask = linear_gradient_mask(input_image.size, input_image.height // 5)
    blend_mask = Image.new("L", extended_sky.size, "white")
    blend_mask.paste(bottom_mask, (0, input_image.height))
    sky_blend = Image.composite(extended_sky, gradient_extended, blend_mask)

    # Return only the new part
    return sky_blend


def concatenate_images_seamless(top_image, bottom_image):
    """Vertically concatenate two images together without leaving a seam mark."""
    blend_mask = sigmoid_gradient_mask(top_image.width, top_image.height * 2)

    foreground = Image.new("RGB", (bottom_image.width, bottom_image.height * 2))
    foreground.paste(top_image)
    foreground.paste(
        top_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0, top_image.height)
    )

    background = Image.new("RGB", (bottom_image.width, bottom_image.height * 2))
    background.paste(bottom_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))
    background.paste(bottom_image, (0, bottom_image.height))

    return Image.composite(background, foreground, blend_mask)


def extend_image(base_image, steps_per_inference=50, step_callback=None):
    """
    Triple the height of an image with more sky and ground.

    The optimal dimensions for the base image are 2508x418 (1024*sqrt(6)).
    The closest dimensions divisible by 8 are 2504x416, but 2048x512 yields better image quality.

    :param PIL.Image.Image base_image: Initial image to work on.
    :param int steps_per_inference: Number of inference steps for each inpainting process.
    :param step_callback: Optional callback function to be called after each inference step.
    :type step_callback: server.task_tracker.TaskTracker | None
    :return PIL.Image.Image: The final image with more sky and ground.
    """
    img_with_sky = add_sky(
        base_image,
        min(steps_per_inference, 30),
        step_callback.incomplete_callback(30) if step_callback else None,
    )
    img_with_sky.show()

    # Add the ground
    extended_ground = add_ground(
        base_image,
        steps_per_inference,
        step_callback.incomplete_callback(30) if step_callback else None,
    )

    # Add the three pieces to the final canvas
    final_image = Image.new(base_image.mode, (base_image.width, base_image.height * 5 // 2))
    final_image.paste(base_image, (0, final_image.height - base_image.height // 2))
    final_image.paste(img_with_sky)
    final_image.paste(extended_ground, (0, img_with_sky.height))
    final_image.show()

    return final_image


def legacy_extension(base_image, prompt, num_inference_steps=50):
    """
    Extend the base image with the legacy pipeline v0.3.

    The main trade-off of this pipeline was that while it made seamless matching,
    it uses up to 16 GB of VRAM and was sometimes not compliant to sky and ground requests.

    :param PIL.Image.Image base_image: Initial image to extend.
    :param str prompt: Prompt to use to tile as a panorama.
    :param int num_inference_steps: Number of inference steps for generation.

    :return PIL.Image.Image: The extended image with a cylindrical projection.
    """
    base_image.show()
    print("Closing the sky...")
    img, mask = me.create_top_mask(base_image)
    img_with_sky = inpaint_image(
        "a sky, uniform color",
        img,
        mask,
        negative_prompt="a logo, a text, clouds, birds",
        num_inference_steps=num_inference_steps,
    )[0]
    img_with_sky.show("Image with more sky")
    img, mask = me.horizontal_tiling_mask(img_with_sky)
    print("Fixing the panorama...")
    panorama = inpaint_image(
        prompt,
        img,
        mask,
        negative_prompt="a logo, a text",
        num_inference_steps=num_inference_steps,
    )[0]
    panorama.show("panorama")
    cylindrical = cylindrical_projection(panorama)
    blended = blend_borders(cylindrical, 10)
    # horizontal_tiling(blended).show("manually tiling")

    return blended


def generate_panorama_legacy(prompt, num_inference_steps=50):
    """
    Create a panorama from a prompt.

    A panorama is an image with a deformation on the vertical axis.

    :param str prompt: The initial user prompt.
    :param int num_inference_steps: Number of inference steps for each step.
    :return PIL.Image.Image: The computed panorama.
    """
    print("Generating image...")
    base_image = generate_images(
        prompt, num_inference_steps=num_inference_steps, width=2048, height=512
    )[0]
    extended_image = legacy_extension(base_image, prompt, num_inference_steps)
    return extended_image


def generate_panorama(prompt, num_inference_steps=50, progress_tracker=None):
    """
    Create a panorama from a prompt, more complete than the legacy version.

    :param str prompt: The initial user prompt.
    :param int num_inference_steps: Number of inference steps for each step.
    :param progress_tracker: A TaskTracker to be called when the step is finished.
    :type progress_tracker: server.task_tracker.TaskTracker | None
    :return PIL.Image.Image: The computed panorama.
    """
    base_image = generate_images(
        prompt, num_inference_steps=num_inference_steps, width=2504, height=416,
        callback_on_step_end=progress_tracker.incomplete_callback(30) if progress_tracker else None
    )[0]
    base_image.show()
    # Inpaint to blend the borders
    panorama = rewrite_image_borders(base_image)
    extended_image = extend_image(panorama, num_inference_steps, progress_tracker)
    return extended_image


def __user_interaction(num_inference_steps=50, use_legacy=False):
    """A demonstration function that asks an image prompt to  the user and shows the result."""
    prompt = input("What panorama do you want? ")
    if not prompt or prompt.strip().isspace():
        prompt = "a peaceful valley"
        print("Using prompt: " + prompt)
    if use_legacy:
        img = generate_panorama_legacy(prompt, num_inference_steps)
    else:
        img = generate_panorama(prompt, num_inference_steps)
    img.show("Final panorama")


if __name__ == "__main__":
    __user_interaction(20)
