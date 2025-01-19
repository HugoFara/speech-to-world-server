"""
Image inpainting.

For general usage, see https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint

Stable diffusion xl 1.0: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
Stable diffusion 1.5 is also great for its licensing terms.
"""

import enum
import warnings

from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageFilter
import numpy as np
import torch
from skimage.restoration.inpaint import inpaint_biharmonic

from .image_processing import horizontal_carrousel


class InpaintingFilling(enum.Enum):
    """
    Inpainting filling masks.

    SAME: do not edit the input image.
    AVERAGE: replace the masked area with the average pixel value.
    MEAN_GREY: replace the masked area with a uniform grey mask.
    BIHARMONIC: bi-harmonic interpolation,
    see skimage.restoration.inpaint_biharmonic for more details
    """

    SAME = 1
    AVERAGE = 2
    MEAN_GREY = 3
    BIHARMONIC = 4
    RANDOM = 5


def make_transparent_black(image):
    """
    From an RGBA image, make the transparent pixels black.

    :param PIL.Image.Image image: Base RGBA image
    :return PIL.Image.Image: Mask image in grayscale format (L).
    """
    # Convert to grayscale format
    grayscale = image.convert("L")
    # Iterate through each pixel in the image
    for y in range(grayscale.height):
        for x in range(grayscale.width):
            # If the alpha value is less than 255 (transparent), set the pixel to black
            if image.getpixel((x, y))[-1] < 255:
                grayscale.putpixel((x, y), 0)
    return grayscale


def center_on_mask(mask_image):
    """
    Translate an image horizontally so that the mask is centered.

    :param PIL.Image.Image mask_image: Mask image where to find the mean point.

    :return: How many pixels should be translated, and if the mask goes across the image.
    :rtype: tuple[int, bool]
    """
    mask_x_pos = np.asarray(mask_image).nonzero()[1]
    mean_point = int(np.mean(mask_x_pos))
    mask_x_extend = np.min(mask_x_pos), np.max(mask_x_pos)
    if mask_x_extend[0] > 0 or mask_x_extend[1] < mask_image.width - 1:
        return mean_point, False

    dummy_mask_translation = horizontal_carrousel(
        mask_image, mask_image.width // 2
    )
    mask_x_pos = np.asarray(dummy_mask_translation).nonzero()[1]
    mean_point = int(np.mean(mask_x_pos))
    mask_x_extend = np.min(mask_x_pos), np.max(mask_x_pos)
    if mask_x_extend[0] == 0 and mask_x_extend[1] == mask_image.width - 1:
        warnings.warn("Seems like the mask is too large!")
    return mask_image.width - mean_point, True


def fill_masked_area(image, mask, inpainting_filling=InpaintingFilling.SAME):
    """Fill a masked area of the given image with a specific strategy."""
    if inpainting_filling == InpaintingFilling.SAME:
        return image
    if inpainting_filling == InpaintingFilling.AVERAGE:
        # Use the average pixel value
        image_data = np.asarray(image)
        mask_data = np.asarray(mask)
        pixels = image_data[mask_data != 0]
        mean_pixel = np.mean(pixels, axis=0).astype(np.uint8)
        area = Image.new(image.mode, image.size, color=tuple(mean_pixel))
        masked_image = image.copy()
        masked_image.paste(area, mask=mask)
        return masked_image
    if inpainting_filling == InpaintingFilling.MEAN_GREY:
        # Equalize with grey
        grey_area = Image.new(image.mode, image.size, color="grey")
        masked_image = image.copy()
        masked_image.paste(grey_area, mask=mask)
        return masked_image
    if inpainting_filling == InpaintingFilling.BIHARMONIC:
        # Bi-harmonic filling
        image_data = np.asarray(image)
        mask_data = np.asarray(mask)
        inpainted = (
            inpaint_biharmonic(image_data, mask_data, channel_axis=-1) * 255
        ).astype(np.uint8)
        return Image.fromarray(inpainted)
    if inpainting_filling == InpaintingFilling.RANDOM:
        # Adds only random values
        image_data = np.asarray(image)
        mask_data = np.asarray(mask)
        rng = np.random.default_rng(1)
        noise_data = rng.integers(
            0, 255, image_data.shape
        ) * mask_data.reshape(*mask_data.shape, 1)
        masked_image = image.copy()
        masked_image.paste(Image.fromarray(noise_data.astype(np.uint8)), mask=mask)
        return masked_image
    raise ValueError


def get_inpainting_pipeline():
    """
    This function initializes and returns a pre-trained Stable Diffusion XL inpainting pipeline.

    The pipeline is loaded from the Hugging Face model hub.
    The pipeline is set to use half-precision (float16) for faster inference and lower memory usage.

    :return: A pre-trained Stable Diffusion XL inpainting pipeline.
    """
    return StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    )


def inpaint_image(prompts, image, mask_image, negative_prompt=None, **pipe_kwargs):
    """
    Apply the prompt to do the inpainting.

    Side effect: reduce the quality of the image, even outside the mask.

    :param str or list[str] prompts: Prompts to use
    :param PIL.Image.Image image: Base image
    :param mask_image: Mask to apply. The mask is white for inpainting and black for keeping as is.
    :type mask_image: PIL.Image.Image
    :param str negative_prompt: Negative prompt to apply
    :return list[PIL.Image.Image]: Inpainted images
    """
    pipe = get_inpainting_pipeline().to("cuda")

    return pipe(
        prompt=prompts,
        image=image,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        height=image.height,
        width=image.width,
        strength=0.9,
        **pipe_kwargs,
    ).images


def force_inpainting(prompts, image, mask_image, negative_prompt=None, **pipe_kwargs):
    """
    Apply the prompts to do the inpainting when you want to be sure that the inpainting is applied.

    The inpainting will start with a random noise instead of the image,
    generating more random results.

    Side effect: reduce the quality of the image, even outside the mask.

    :param str or list[str] prompts: Prompts to use
    :param PIL.Image.Image image: Base image
    :param mask_image: Mask to apply. The mask is white for inpainting and black for keeping as is.
    :type mask_image: PIL.Image.Image
    :param str negative_prompt: Negative prompt to apply
    :return list[PIL.Image.Image]: Inpainted images
    """
    masked_image = fill_masked_area(image, mask_image, InpaintingFilling.RANDOM)
    pipe_kwargs["guidance_scale"] = 20
    pipe_kwargs["num_inference_steps"] = 25
    return inpaint_image(
        prompts, masked_image, mask_image, negative_prompt, **pipe_kwargs
    )


def image_compositing(
    initial_image, inpainted_image, mask_image, blurring_radius, horizontal_tiling=False
):
    """
    Preserve the quality of the original image by blending the original and the inpainted images.

    :param PIL.Image.Image initial_image: Initial image before any inpainting.
    :param PIL.Image.Image inpainted_image: Image after inpainting process.
    :param PIL.Image.Image mask_image: Mask image to define the area to be inpainted.
    :param int blurring_radius: Radius of the blurring filter applied to the mask.
    :param bool horizontal_tiling: If True, we apply a horizontal tiling before compositing.
    :return PIL.Image.Image final_composition: Composited image with original and inpainted parts.
    """
    if horizontal_tiling:
        image_frame = Image.new(
            initial_image.mode, (initial_image.width * 2, initial_image.height)
        )
        inpainted_image_frame = image_frame.copy()
        mask_image_frame = image_frame.copy()
        # Remark: an image of size (base_image.width + blurring_radius * 2) would be enough
        for left_padding in range(0, initial_image.width * 2, initial_image.width):
            image_frame.paste(initial_image, (left_padding, 0))
            inpainted_image_frame.paste(inpainted_image, (left_padding, 0))
            mask_image_frame.paste(mask_image, (left_padding, 0))
        blurred_mask = mask_image_frame.filter(ImageFilter.BoxBlur(blurring_radius)).convert("L")
        big_image = Image.composite(inpainted_image_frame, image_frame, blurred_mask)
        final_composition = big_image.crop(
            (initial_image.width, 0, initial_image.width * 2, initial_image.height)
        )
    else:
        blurred_mask = mask_image.filter(ImageFilter.BoxBlur(blurring_radius))
        final_composition = Image.composite(inpainted_image, initial_image, blurred_mask)
    return final_composition


def inpaint_panorama_pipeline(
        init_image, mask_image, prompt, step_callback=None, blurring_radius=40
):
    """
    Base framework for an inpainting.

    :param PIL.Image.Image init_image: Initial image to inpaint
    :param PIL.Image.Image mask_image: Mask image to use
    :param str prompt: Prompt for inpainting
    :param step_callback: Function to run at the end of each step f : step_number -> Any
    :type step_callback: Callable | None
    :param int blurring_radius: Size of the blurring radius to apply.

    :return PIL.Image.Image: The new inpainted image
    """
    left_translation, should_translate = center_on_mask(mask_image)
    # If the mask is across the borders we need to "turn" the image
    if should_translate:
        translated_image = horizontal_carrousel(init_image, left_translation)
        translated_mask = horizontal_carrousel(mask_image, left_translation)
        translated_result = force_inpainting(
            prompt,
            translated_image,
            translated_mask,
            callback_on_step_end=step_callback,
        )[0]
        new_image = horizontal_carrousel(translated_result, -left_translation)
    else:
        new_image = force_inpainting(
            prompt, init_image, mask_image, callback_on_step_end=step_callback
        )[0]
    # Apply the image on the mask only to avoid quality decrease
    composited_image = image_compositing(init_image, new_image, mask_image, blurring_radius, True)
    return composited_image


def inpainting_demo():
    """
    A demo interaction of what the model can do.

    This function demonstrates the usage of the model by prompting the user for a replacement,
    to be added in the input image.
    If the user doesn't provide a valid input, a default prompt is used.
    """
    demo_prompt = "A cat, high resolution, sitting"
    prompt = input(f"What replacement do you want? [{demo_prompt}] ")
    if not prompt or prompt.strip().isspace():
        prompt = demo_prompt
    image_path = input("What is the image path? [../sunny_mountain.png] ")
    if not image_path or image_path.strip().isspace():
        image_path = "../sunny_mountain.png"
    base_image = Image.open(image_path).convert("RGB")
    mask_path = input("What is the mask path? [mask.png] ")
    if not mask_path or mask_path.strip().isspace():
        mask_path = "mask.png"
    mask_image = Image.open(mask_path)

    print("Starting inpainting")
    inpainted_images = inpaint_image([prompt] * 4, base_image, mask_image)
    for im in inpainted_images:
        im.show()
    print("Restoring initial image quality.")
    for im in inpainted_images:
        image_compositing(base_image, im, mask_image, 5, True).show()


def __regenerate_mask():
    image = Image.open("../sunny_mountain.png")
    # Define the size of the mask (width, height)
    mask_size = image.size

    # Create a blank mask filled with zeros
    mask = torch.zeros(mask_size, dtype=torch.uint8)

    # Set some pixels to 1 to create a binary mask
    mask[
        image.width // 2 : image.width // 2 + 100,
        image.height // 2 : image.height // 2 + 100,
    ] = 255

    # Save the mask as a PNG file using Pillow
    img = Image.fromarray(mask.numpy())
    img.save("mask.png")
    return img


if __name__ == "__main__":
    inpainting_demo()
