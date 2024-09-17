"""
Define an outpainting node that stretches the borders of the original image.
"""
import torch


def cylindrical_projection(image):
    """
    Compute a cylindrical projection from a flat image.

    The x-axis is preserved, by the y-axis will be changed.
    This is the inverse operation of a Lambert projection.

    :param torch.tensor image: Input image
    :return torch.tensor: Output image in reversed cylindrical projection
    """
    height, width = image.shape[1:3]
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

    return cylindrical_image


class ImageReverseLambert:
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
            }
        }

    RETURN_TYPES = "IMAGE",

    FUNCTION = "main"

    # OUTPUT_NODE = False

    CATEGORY = "image"

    def main(self, image):
        return cylindrical_projection(image),


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageReverseLambert": ImageReverseLambert
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageReverseLambert": "Project as Reversed Lambert"
}
