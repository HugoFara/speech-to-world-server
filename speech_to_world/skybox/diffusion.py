"""
Simple(st) diffusion network,
based on https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0.

Generates an image after prompt.
"""
import warnings

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch


def show_images(images):
    """
    Show the first five images.

    :param list[PIL.Image.Image] images: Images.
    """
    for i in range(min(len(images), 5)):
        images[i].show()


def is_power_of_two(n):
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def get_image_generation_pipeline():
    """Load a text-to-image pipeline from Hugging Face for SDXL base."""
    return StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )


def get_image_refinement_pipeline():
    """Load an image-to-image pipeline from Hugging Face using Stable Diffusion XL."""
    return StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )


def generate_images(prompt, num_inference_steps=50, height=1024, width=None, **pipe_kwargs):
    """
    Generate an image from the given prompt, using a diffusion network.

    Note: for best results with SDXL, height * width should be equal to 1024*1024.

    :param prompt: The prompt for the image.
    :type prompt: str | tuple[str] | list[str]
    :param int num_inference_steps: Number of denoising steps
    :param int height: Image height, should be a power of two
    :param int width: Image width, if left to None it will be equal to 1024*1024 // height
    :param dict pipe_kwargs: Additional arguments to pass to the pipeline.
    :return list[PIL.Image.Image]: Generated images
    """
    if width is None:
        width = 1024 * 1024 // height
    if not is_power_of_two(height) and not is_power_of_two(width):
        warnings.warn(
            f"Specified dimensions {width} * {height} are not powers of two, proceed with care."
        )
    elif not is_power_of_two(height):
        warnings.warn(
            f"Specified image height {height} is not a power of two, you may run into issues."
        )
    elif not is_power_of_two(width):
        warnings.warn(
            f"Specified image width {width} is not a power of two, you may run into issues."
        )

    if width * height != 1024 * 1024:
        print(
            "width * height should be equal to 1024 * 1024 for better results.",
            f"Current is {width} * {height}."
        )

    pipe = get_image_generation_pipeline().to("cuda")
    # If more VRAM needed
    # pipe.enable_model_cpu_offload()

    # If computation takes a long time (Linux only)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    return pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        **pipe_kwargs
    ).images


def refine_images(prompt, init_image, num_inference_steps=15, **pipe_kwargs):
    """
    Refine a batch of images using the diffusion network.

    :param str | list[str] prompt: The prompt for the refined image.
    :param PIL.Image.Image init_image: The initial image to refine.
    :param int num_inference_steps: The number of inference steps for the refinement process.
    :param dict pipe_kwargs: Additional keyword arguments to pass to the pipeline.
    :return list[PIL.Image.Image]: A list of refined images.
    """
    pipe = get_image_refinement_pipeline().to("cuda")

    return pipe(
        prompt, image=init_image, num_inference_steps=num_inference_steps, **pipe_kwargs
    ).images


def main():
    """Main demo for the diffusion model."""
    demand = input(
        "What would you like to generate? (Empty: An astronaut riding a green horse) "
    )
    if not demand or demand.strip().isspace():
        demand = "An astronaut riding a green horse"
    batch_size = 1
    inference_steps = 50
    results = generate_images(
        [demand] * batch_size, num_inference_steps=inference_steps, height=512, width=2048,
    )
    show_images(results)
    results = refine_images([demand] * batch_size, results, inference_steps)
    show_images(results)


if __name__ == "__main__":
    main()
