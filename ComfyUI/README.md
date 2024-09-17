# ComfyUI workflows

This repository stores workflows for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). 
Please note that the custom nodes involved have to be loaded manually in ComfyUI.

## Philosophy

The node defined here are totally independent of the rest of the project, and may reimplement existing
features of the code base. The main purpose of this folder is to provide visual equivalents to the code features. 

## Main workflows

The workflows can be found in the ``workflows`` folder. It is recommended to use small workflows steps, 
as the result of the biggest workflows are quite random. 

- sdxl.json: Basic image generation using Stable Diffusion, roughly equivalent to ``../skybox/diffusion.py`` 
([HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)).
- sdxl_with_refiner.json: Improved image generation that implements the 
[refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0).
- inpainting_demo.json: some simple demo for inpainting.
- sdxl_inpainting_demo.json: a more complete inpainting demo using pure SDXL features.
- central_inpainting.json: An inpainting implementation with the standard functions,
using [sdxl inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), 
that creates a horizontal tiling.
- panorama_creator.json: an extended workflow to create a flat image as a panorama.
- text_to_skybox.json: a complete workflow to generate a skybox from a prompt.
