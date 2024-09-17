"""
Starts a Python server using sockets, able to pass data to various AI functions.
"""

import os
import time
import json
import socket
import socketserver
import threading
import warnings

import torch.cuda
from PIL import Image

from ..asr.speech_to_text import do_audio_transcription
from .utils import (
    hex_to_pillow,
    get_server_address,
    hex_to_bytes,
    image_response,
    get_configuration_data,
)
from .task_tracker import TaskTracker
from ..skybox.diffusion import generate_images, refine_images
from ..skybox.inpainting import make_transparent_black, inpaint_panorama_pipeline
from ..skybox import panorama_creator

# Max chunk size for input data
CHUNK_SIZE = 4096


def init_server(server_ip, server_port):
    """
    Initialize the server with the input configuration file.
    """
    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the IP address and port
    server_socket.bind((server_ip, server_port))

    # Listen for incoming connections
    server_socket.listen(5)
    print(f"Server is listening on {server_ip}:{server_port}")
    return server_socket


def server_data():
    """The public data about this server."""
    configuration_data = get_configuration_data()
    data = {
        "name": configuration_data["name"],
        "description": configuration_data["description"],
        "version": configuration_data["version"],
    }
    return data


def completion_report(completion, client_socket, task_id):
    """
    Completion report for a task.

    Send a completion report through the TCP connection.

    :param int completion: Task completion from 0 to 100
    :param socket.socket client_socket: TCP client socket
    :param int task_id: Identifier of the task to check completion.
    :return dict: Data sent
    """
    data = {"completion": "progress", "taskCompletion": completion, "taskId": task_id}
    response = {"status": 200, "data": json.dumps(data), "type": "completion"}
    send_response(response, client_socket)
    return data


def new_skybox_handler(prompt, advanced, progress_tracker=None):
    """
    Generate a new skybox and add the image to the output data.

    :param string prompt: The prompt for the skybox generation.
    :param bool advanced: Stop the generation at the first of the pipeline if true.
    :param TaskTracker | None progress_tracker: TaskTracker object to report each step
    """
    if advanced:
        height = 416
        image = Image.new("RGB", (2504, height * 5 // 2), "black")
        base_image = generate_images(
            prompt, num_inference_steps=50, width=2504, height=height,
            **{"callback_on_step_end": progress_tracker.callback if progress_tracker else None}
        )[0]
        image.paste(base_image, (0, height))
    else:
        image = panorama_creator.generate_panorama(prompt, progress_tracker=progress_tracker)
    response = image_response(image)
    return response


def new_skybox_local_handler(prompt, destination_path, step_callback=None):
    """
    Generate a new skybox.

    :param string prompt: The prompt for the skybox generation.
    :param string destination_path: Where to save the generated image.
    :param Callable | None step_callback: Callback function to report each step.
    :return dict: A dictionary with "skyboxFilePath" key.
    """
    images = generate_images(
        prompt, callback_on_step_end=step_callback, height=1024, width=2048
    )
    images[0].save(destination_path)
    data = {"skyboxFilePath": destination_path}
    return data


def panorama_handler(prompt, step_callback=None):
    """
    Generate a new panorma skybox (no seam line) and add the image to the output data.

    Deprecated: since pipeline v0.4, use new_skybox_handler instead.

    :param string prompt: The prompt for the skybox generation.
    :param Callable | None step_callback: Callback function to report each step.
    :return dict: A dictionary containing the image bytes in hexadecimal string.
    """
    image = generate_images(
        prompt, callback_on_step_end=step_callback, height=1024, width=2048
    )[0]
    panorama = panorama_creator.rewrite_image_borders(image)
    cylindrical = panorama_creator.cylindrical_projection(panorama)
    smoothed = panorama_creator.blend_borders(cylindrical, 10)
    response = image_response(smoothed)
    return response


def refine_skybox_handler(image_hex, prompt, step_callback=None):
    """
    Refine an image with SDXL refiner.

    :param str image_hex: Base image hexadecimal string, PNG format
    :param str prompt: Prompt to guide the refining process.
    :param Callable | None step_callback: Report function object to report each step.
    :return dict: A dictionary with the PNG image in encoded.
    """
    base = hex_to_pillow(image_hex).convert("RGB")
    image_part = base.crop((0, base.height * 2 // 5, base.width, base.height * 4 // 5))
    refined = refine_images(
        prompt, image_part, num_inference_steps=50, **{"callback_on_step_end": step_callback}
    )[0]
    base.paste(refined, (0, base.height * 2 // 5))
    response = image_response(base)
    return response


def remove_seam_handler(image_hex, _step_callback=None):
    """
    Fixes the borders of an image to make it an asymmetric tiling.

    :param str image_hex: Base image hexadecimal string, PNG format
    :param Callable | None _step_callback: Callback function to report each step.
    :return dict: A dictionary with the PNG image encoded.
    """
    image_frame = hex_to_pillow(image_hex).convert("RGB")
    image_part = image_frame.crop(
        (0, image_frame.height * 2 // 5, image_frame.width, image_frame.height * 4 // 5)
    )
    asymmetric_image = panorama_creator.rewrite_image_borders(image_part)
    image_frame.paste(asymmetric_image, (0, image_frame.height * 2 // 5))
    response = image_response(image_frame)
    return response


def extend_skybox_handler(image_hex, step_callback=None):
    """
    Expand the given image to create a larger skybox.

    :param str image_hex: Base image hexadecimal string, PNG format
    :param Callable | None step_callback: Callback function to report each step.
    :return dict: A dictionary with the PNG image encoded.
    """
    image_frame = hex_to_pillow(image_hex).convert("RGB")
    image_part = image_frame.crop(
        (0, image_frame.height * 2 // 5, image_frame.width, image_frame.height * 4 // 5)
    )
    extended = panorama_creator.extend_image(image_part, 50, step_callback=step_callback)
    response = image_response(extended)
    return response


def asr_local_handler(audio_file_path):
    """
    Return the transcription from an audio file.

    :param str audio_file_path: Audio file path
    :return dict: Text enclosed in "transcription" key
    """
    if os.path.exists(audio_file_path):
        result = do_audio_transcription(audio_file_path)
        print(result)
        data = {"transcription": result["text"]}
    else:
        print("File does not exist")
        data = {
            "transcription": f"Error: input file {audio_file_path} does not exist!",
            "message": f"Error: input file {audio_file_path} does not exist!",
        }
    return data


def asr_handler(audio_bytes):
    """
    Return the transcription from an audio.

    :param str audio_bytes: The audio as byte string, hexadecimal encoded
    :return dict: Text enclosed in "transcription" key
    """
    raw_bytes = hex_to_bytes(audio_bytes)
    result = do_audio_transcription(raw_bytes)
    print(result)
    return {"transcription": result["text"]}


def inpaint_handler(image_hex, mask_image_hex, prompt, step_callback=None):
    """
    Inpaint (draw on) an image using a prompt, and add the image to the output data.

    :param str image_hex: Hexadecimal string encoding of the image in PNG format.
    :param str mask_image_hex: Mask image bytes, PNG format
    :param str prompt: Prompt for inpainting
    :param step_callback: Function to run at the end of each step f : step_number -> Any
    :type step_callback: Callable | None

    :return dict: The new inpainted image, in standard image response format.
    """
    init_image = hex_to_pillow(image_hex).convert("RGB")
    mask_image = make_transparent_black(hex_to_pillow(mask_image_hex)).resize(init_image.size)
    new_image = inpaint_panorama_pipeline(init_image, mask_image, prompt, step_callback)
    response = image_response(new_image)
    return response


def inpaint_local_handler(
    init_image_path, mask_image_path, prompt, destination_path, step_callback=None
):
    """
    Inpaint (draw on) an image using a prompt.

    :param str init_image_path: Base image path
    :param str mask_image_path: Mask image path
    :param str prompt: Prompt for inpainting
    :param str destination_path: Destination path for the inpainted image
    :param step_callback: Function to run at the end of each step f : step_number -> Any
    :type step_callback: Callable | None

    :return dict: Path to the new image, enclosed in "inpaintedFilePath" key
    """
    init_image = Image.open(init_image_path).convert("RGB")
    mask_image = make_transparent_black(Image.open(mask_image_path)).resize(init_image.size)
    new_image = inpaint_panorama_pipeline(init_image, mask_image, prompt, step_callback)
    new_image.save(destination_path)
    data = {"inpaintedFilePath": destination_path}
    return data


def send_response(response, client_socket):
    """
    Send a response through the client socket.

    :param dict response: Response to send, a flat (not nested) dictionary.
    :param socket.socket client_socket: Socket to send the response
    """
    str_dump = json.dumps(response)
    client_socket.sendall(str_dump.encode())


def start_task(task_dict, tracker):
    """
    Start a new server task.

    :param dict task_dict: Dictionary of data about this task
    :param TaskTracker | None tracker: Object to call on step end
    :return dict: Dictionary containing the response to this task.
    """
    result = None
    print(f"Starting task: {task_dict['type']}")

    if task_dict["type"] == "new-skybox-local":
        result = new_skybox_local_handler(
            task_dict["prompt"],
            task_dict["outputFilePath"],
            step_callback=tracker.callback,
        )
    elif task_dict["type"] == "new-skybox":
        result = new_skybox_handler(
            task_dict["prompt"], bool(task_dict["quick"]), progress_tracker=tracker
        )
    elif task_dict["type"] == "panorama":
        result = panorama_handler(task_dict["prompt"], step_callback=tracker.callback)
    elif task_dict["type"] == "inpainting-local":
        result = inpaint_local_handler(
            task_dict["imagePath"],
            task_dict["maskPath"],
            task_dict["prompt"],
            task_dict["outputFilePath"],
            step_callback=tracker.callback,
        )
    elif task_dict["type"] == "inpainting":
        result = inpaint_handler(
            task_dict["imageBytes"],
            task_dict["maskBytes"],
            task_dict["prompt"],
            step_callback=tracker.callback,
        )
    elif task_dict["type"] == "refine-skybox":
        result = refine_skybox_handler(
            task_dict["imageBytes"],
            task_dict["prompt"],
            step_callback=tracker.callback,
        )
    elif task_dict["type"] == "remove-seam":
        result = remove_seam_handler(
            task_dict["imageBytes"],
            _step_callback=tracker.callback,
        )
    elif task_dict["type"] == "extend-skybox":
        result = extend_skybox_handler(
            task_dict["imageBytes"],
            step_callback=tracker,
        )
    elif task_dict["type"] == "asr-local":
        result = asr_local_handler(task_dict["audioPath"])
    elif task_dict["type"] == "asr":
        result = asr_handler(task_dict["audioBytes"])
    elif task_dict["type"] == "ping":
        result = {
            "queryTimestamp": task_dict["queryTimestamp"],
            "responseTimestamp": int(time.time() * 1000),
            "responseMilliseconds": int(time.time() * 1000)
            - task_dict["queryTimestamp"],
        }

    if result is None:
        raise NotImplementedError(
            f"The task '{task_dict['type']}' is not recognized as a valid task type."
        )

    return result


def prepare_response(json_data, tracker):
    """
    Prepare a response to be sent after a query.

    :param dict json_data: Response dictionary
    :param TaskTracker | None tracker: TaskTracker tp report completion.
    :return dict response: Response dictionary
    """
    if "reportCompletion" not in json_data or json_data["reportCompletion"] == 0:
        tracker = None
    if json_data["type"] == "info":
        answer_data = server_data()
        response = {
            "status": 200,
            "data": json.dumps(answer_data),
            "message": "Info",
            "type": "info",
        }
    elif json_data["type"] == "completion":
        response = {
            "status": 200,
            "data": json.dumps({"completion": 0}),
            "type": "completion",
        }
    elif json_data["type"] in (
        "ping",
        "new-skybox-local",
        "new-skybox",
        "panorama",
        "refine-skybox",
        "inpainting-local",
        "inpainting",
        "remove-seam",
        "extend-skybox",
        "asr-local",
        "asr",
    ):
        response = {
            "status": 200,
            "taskId": json_data["taskId"],
            "type": json_data["type"],
        }
        try:
            answer_data = start_task(json_data, tracker)
        except torch.cuda.OutOfMemoryError as err:
            response = {
                "status": 500,
                "data": json.dumps({}),
                "message": f"Out of memory: {err}",
                "type": "error",
            }
        else:
            response["data"] = json.dumps(answer_data)
    else:
        response = {
            "status": 404,
            "data": json.dumps({}),
            "message": f"Unknown type: {json_data['type']}",
            "type": "error",
        }
    return response


def safe_send(response, client_socket):
    """Safely send a response to the client, handling large responses by fragmenting them."""
    size_limit = 8196 * 16
    if len(response["data"]) > size_limit:
        n_fragments = len(response["data"])
        capsule = {}
        capsule.update(response)
        capsule["status"] = 206
        capsule["total_fragments"] = n_fragments // size_limit
        print(f"Fragmenting response in {capsule['total_fragments']} fragments.")
        for i in range(capsule["total_fragments"]):
            fragment = response["data"][i * size_limit: i * size_limit + size_limit]
            capsule["data"] = fragment
            print(len(fragment))
            capsule["index"] = i
            send_response(capsule, client_socket)
    else:
        send_response(response, client_socket)


def handle_query(data, client_socket):
    """
    Handle a query from a client.

    :param str data: Data received from client
    :param socket.socket client_socket: The client socket
    """
    if not data or data.isspace():
        print("Empty data, aborting")
        response = {"status": 400, "data": json.dumps({}), "message": "Empty data"}
    else:
        try:
            json_data = json.loads(data)
        except ValueError as error:
            print("Data is not json")
            response = {
                "status": 304,
                "data": json.dumps({}),
                "message": f"Wrong JSON: {error}",
            }
        else:
            tracker = TaskTracker(client_socket, json_data["taskId"], completion_report)
            response = prepare_response(json_data, tracker)

    send_response(response, client_socket)


def wait_for_connection(server_socket):
    """
    Wait for a connection from a client.

    :param socket.socket server_socket: Server socket
    """
    # Accept incoming connection
    client_socket, client_address = server_socket.accept()
    print(f"Client {client_address} connected.")

    # Receive data from client
    bytes_buffer = []
    while True:
        bytes_read = client_socket.recv(CHUNK_SIZE)
        bytes_buffer.append(bytes_read)
        if len(bytes_read) == 0:
            warnings.warn("Received 0 bytes from client")
        elif len(bytes_read) == 1:
            print("Received short data " + bytes_read.decode())
        elif bytes_read[-2] != b"\\" and bytes_read.endswith(b"}"):
            break
    query_string = b"".join(bytes_buffer).decode()
    try:
        handle_query(query_string, client_socket)
    except ConnectionResetError:
        print("Connection reset during transmission.")

    # Close the connection
    client_socket.close()


def handle(client_socket, client_address):
    """Read the data until termination and take action."""
    # self.request is the TCP socket connected to the client
    bytes_buffer = []
    while True:
        bytes_read = client_socket.recv(CHUNK_SIZE)
        bytes_buffer.append(bytes_read)
        if len(bytes_read) == 0:
            warnings.warn("Received 0 bytes from client")
        elif len(bytes_read) == 1:
            print("Received short data " + bytes_read.decode())
        elif bytes_read[-2] != b"\\" and bytes_read.endswith(b"}"):
            break
    query_string = b"".join(bytes_buffer).decode()
    print(f"Request from {client_address[0]}:{client_address[1]}")
    handle_query(query_string, client_socket)


class TCPHandler(socketserver.StreamRequestHandler):
    """Instantiates the server."""

    def handle(self):
        """Define our to receive data, just a wrapper for the handle function."""
        handle(self.request, self.client_address)


def run_server(forked_server=True):
    """Start the server."""
    server_ip, server_port = get_server_address()

    # Create the server
    server = socketserver.TCPServer((server_ip, server_port), TCPHandler)
    with server:
        print(f"Starting server on {server_ip}:{server_port}")
        if forked_server:
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.start()
        else:
            server.serve_forever()


if __name__ == "__main__":
    run_server(False)
