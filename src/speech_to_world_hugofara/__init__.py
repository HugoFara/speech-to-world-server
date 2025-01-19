"""
Speech-to-world, a complete environment for generating worlds from prompts.
"""

from .asr.speech_to_text import get_asr_model, do_audio_transcription

from .environment.depth_generation import (
    get_depth,
    compute_image_depth,
    get_depth_pipeline,
)
from .environment.depth_inpainting import (
    get_inpaint_depth_pipeline,
    inpaint_depth_controlled,
)
from .environment.image_segmentation import (
    segment_anything,
    get_ground_mask,
    get_skybox_mask,
)
from .environment.renderer import (
    inpaint_ground,
    inpaint_skybox,
    complete_segments,
)

from .server.run import run_server

from .skybox.diffusion import (
    get_image_generation_pipeline,
    get_image_refinement_pipeline,
    generate_images,
    refine_images
)
from .skybox.inpainting import inpaint_panorama_pipeline
from .skybox.panorama_creator import extend_image

from .utils.download_models import (
    download_production_pipelines,
    download_alpha_pipelines,
)
