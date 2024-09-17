"""
Define a VerticalMiddleMask node that mask only the center of an image.
"""
import torch


class VerticalMiddleMask:
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
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_width": ("INT", {
                    "default": 16,
                    "min": 0,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 4,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                })
            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("Mask", )

    FUNCTION = "main"

    # OUTPUT_NODE = False

    CATEGORY = "mask"

    def main(self, image, mask_width):
        mask = torch.zeros(image.shape[:-1])
        image_center = image.shape[2] // 2
        mask[:, :, image_center - mask_width // 2:image_center + mask_width // 2] = 1
        return mask,


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VerticalMiddleMask": VerticalMiddleMask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VerticalMiddleMask": "VerticalMiddleMask"
}

