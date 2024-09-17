"""
Define an outpainting node that stretches the borders of the original image.
"""
import torch


def pad_image(image, top, bottom):
    initial_height = image.shape[1]
    shape = list(image.shape)
    shape[1] = initial_height + top + bottom
    output = torch.zeros(shape)
    mask = torch.zeros(output.shape[:-1])
    output[:, top:top + initial_height] = image
    mask[:, :top] = 1
    mask[:, initial_height + top:] = 1
    return output, mask


def stretch_image(padded_image, top, bottom, border):
    output = padded_image.clone().detach()
    top_area = padded_image[:, top:top + border]
    top_area = torch.mean(top_area, 1, keepdim=True).repeat([1, top, 1, 1])
    output[:, :top] = top_area

    if bottom > 0:
        bottom_area = padded_image[:, -bottom - border:-bottom]
        bottom_area = torch.mean(bottom_area, 1, keepdim=True).repeat([1, bottom, 1, 1])
        output[:, -bottom:] = bottom_area
    return output


class ImageStretchForOutpaint:
    """
    A node that splits an image in the middle and returns the two parts in mirror.

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re-executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        The name of each output in the output tuple (Optional).
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents
        if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"`
        then it must be `foo`.
    """

    @classmethod
    def INPUT_TYPES(self):
        """
        Return a dictionary which contains config for all input fields.

        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional.
            A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        outpainting_settings = {
            "default": 0,
            "min": 0,  # Minimum value
            "max": 1024,  # Maximum value
            "step": 4,  # Slider's step
            "display": "number"  # Cosmetic only: display as "number" or "slider"
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "top": ("INT", outpainting_settings),
                "bottom": ("INT", {
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 4,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "border": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 1024,
                    "step": 1,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    # RETURN_NAMES = ("Image", "Mask", )

    FUNCTION = "main"

    # OUTPUT_NODE = False

    CATEGORY = "image"

    def main(self, image, top, bottom, border):
        padded_image, mask = pad_image(image, top, bottom)
        output = stretch_image(padded_image, top, bottom, border)
        return output, mask


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageStretchForOutpaint": ImageStretchForOutpaint
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageStretchForOutpaint": "Stretch Image for Outpainting"
}

