"""
Inpainting using depth data as a ControlNet.

https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetInpaintPipeline
"""

import diffusers
from PIL import Image
import torch


def get_inpaint_depth_pipeline():
    """
    Initialize and return a Stable Diffusion XL ControlNet inpainting pipeline.
    The pipeline uses depth data as a control signal for inpainting.

    For details:
    https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetInpaintPipeline

    :return: A pre-configured pipeline for inpainting with depth control.
    :rtype: diffusers.pipelines.stable_diffusion.StableDiffusionPipelineOutput
    """
    controlnet = diffusers.ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
    )
    pipe = diffusers.StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    return pipe


def inpaint_depth_controlled(init_image, mask_image, control_image, prompts):
    """
    Perform depth-guided inpainting using a Stable Diffusion XL ControlNet pipeline.

    This function initializes a pre-configured pipeline for inpainting with depth control,
    and then generates images based on the given parameters.


    :param PIL.Image.Image init_image: The initial image to start inpainting from.
    :param PIL.Image.Image mask_image: The mask image indicating the areas to be inpainted.
    :param PIL.Image.Image control_image: The depth map image to guide the inpainting process.
    :param str prompts: The text prompt to guide the image generation.

    :return list[PIL.Image.Image]: A list containing the generated inpainted images.
    """
    pipe = get_inpaint_depth_pipeline()
    # pipe.to("cuda")
    pipe.enable_model_cpu_offload()  # use it instead of CUDA if you run out of VRAM
    # Generate the images
    images = pipe(
        prompts,
        num_inference_steps=50,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images
    return images


if __name__ == "__main__":
    inpaint_depth_controlled(
        Image.open("../sunny_mountain.png"),
        Image.open("../skybox/mask.png"),
        Image.open("sunny_depth_map.png"),
        "a mountain",
    )[0].show()
