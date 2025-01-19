"""
Various utility functions for the server,
"""

import os
import io
import json

from PIL import Image


def hex_to_bytes(hex_string):
    """
    Convert a hex string to bytes.

    :param str hex_string: Hex string is in C# format, separated by dashes.
    :return bytes: Bytes object
    """
    return bytes.fromhex(hex_string.replace("-", ""))


def hex_to_pillow(hex_string):
    """
    Take a hex string and convert it to a pillow image.

    :param str hex_string: Hex string in C# format, separated by dashes.
    :return PIL.Image.Image: Decoded Pillow image.
    """
    base_image_io = io.BytesIO(hex_to_bytes(hex_string))
    return Image.open(base_image_io)


def get_image_bytes(image):
    """
    Return the bytes composing a PNG image.

    :param PIL.Image.Image image: Input image to get bytes from.
    :return bytes: Bytes object
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def get_configuration_data():
    """
    Configuration data for the server.

    :return dict: Server configuration data from a JSON file.
    """
    with open(os.path.join(os.path.dirname(__file__), "../api.json"), encoding="utf-8") as file:
        configuration_data = json.load(file)
    return configuration_data


def get_server_address():
    """Return the suggested IP and port for the server."""
    configuration_data = get_configuration_data()
    # Specify the IP address and port the server will listen on
    server_ip = configuration_data["serverIp"]
    server_port = configuration_data["serverPort"]
    return server_ip, server_port


def image_response(image):
    """
    A classical image response, image encoded in hexadecimal bytes.\

    :param PIL.Image.Image image: The image to return.
    :return dict: Response data with the key 'imageHexBytes'.
    """
    skybox_bytes = get_image_bytes(image)
    data = {"imageHexBytes": skybox_bytes.hex()}
    return data
