"""
Simple utility script that forces the download of all models.

Just load the script, and the models should get installed.
"""
from ..asr import speech_to_text
from ..environment import depth_generation, depth_inpainting
from ..skybox import diffusion, inpainting


def download_alpha_pipelines():
    """Models not used in production yet."""
    print("Starting loading models")
    print("Loading ControlNet inpainting...")
    depth_inpainting.get_inpaint_depth_pipeline()
    print("Loading depth generation...")
    depth_generation.get_depth_pipeline()

    print("Finished loading models in alpha with success!")


def download_production_pipelines():
    """Load all pipelines used in the server in order to download the associated models."""
    print("Starting loading models")
    print("Loading speech recognition...")
    speech_to_text.get_asr_model()
    print("Loading image generation...")
    diffusion.get_image_generation_pipeline()
    print("Loading image refinement...")
    diffusion.get_image_refinement_pipeline()
    print("Loading inpainting...")
    inpainting.get_inpainting_pipeline()

    print("Finished loading models with success!")


if __name__ == "__main__":
    download_production_pipelines()
