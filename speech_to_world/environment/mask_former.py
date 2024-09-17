"""
MaskFormer segmentation by Facebook.
"""

from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

# model.config.id2label
LABELS = {
    "terrain": (
        "floor-wood",
        "flower",
        "gravel",
        "river",
        "road",
        "sand",
        "sea",
        "snow",
        "stairs",
        "floor-other-merged",
        "pavement-merged",
        "mountain-merged",
        "grass-merged",
        "dirt-merged",
        "building-other-merged",
        "rock-merged",
        "rug-merged",
    ),
    "sky": ("ceiling-merged", "sky-other-merged"),
}


def mask2former_model():
    """Return the model for Mask2Former."""
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-coco-panoptic"
    )
    return model


def merge_label_list():
    """Create the id_to_fuse list."""
    label_list = [[] for _ in range(3)]
    label2id = mask2former_model().config.label2id
    for merge_list, key in zip(LABELS, label_list):
        for label in LABELS[key]:
            merge_list.append(label2id[label])
    return label_list


def panoptic_segmentation(image):
    """
    Apply a panoptic segmentation to a given image.

    :return: Batch of panoptic segmentations,
    each of which is a dict with keys "segmentation" and "segments_info".
    :rtype: list
    """
    model = mask2former_model()
    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-coco-panoptic"
    )
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Fuse terrain and sky elements: if the sky appears twice, group it in the same group
    label_ids_to_fuse = list(model.config.label2id[label] for label in LABELS["terrain"])
    label_ids_to_fuse.extend(model.config.label2id[label] for label in LABELS["sky"])

    return processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]], label_ids_to_fuse=label_ids_to_fuse
    )


def mask_former(image):
    """
    Assign mask to the pixels of an image.

    :param PIL.Image.Image image: Image to segment.
    """
    result = panoptic_segmentation(image)[0]
    # we refer to the demo notebooks for visualization
    # (see "Resources" section in the Mask2Former docs)
    predicted_panoptic_map = result["segmentation"]
    # Convert the tensor to a NumPy array
    return predicted_panoptic_map.squeeze().detach().numpy()


def get_model_labels():
    """A dictionary of id to labels associations for the model used."""
    model = mask2former_model()
    return model.config.id2label


def get_sky_ids():
    """The ids of the elements corresponding to the sky."""
    label2id = mask2former_model().config.label2id
    return tuple(label2id[label] for label in LABELS['sky'])


def main(image):
    """Demonstration usage of MaskFormer."""
    result = panoptic_segmentation(image)[0]
    segments_map = result["segmentation"].squeeze().detach()
    segments_info = result["segments_info"]
    values = torch.unique(segments_map)

    _fig, ax = plt.subplots()
    im = ax.imshow(segments_map)
    ax.imshow(image, alpha=0.5)

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    id2label = get_model_labels()
    labels_list = [id2label[seg["label_id"]] + f" ({seg['label_id']})" for seg in segments_info]
    if 0 in values:
        labels_list.insert(0, "Unknown (0)")
    patches = [
        mpl.patches.Patch(color=colors[i], label=label) for i, label in enumerate(labels_list)
    ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches)
    plt.show()


if __name__ == "__main__":
    main(Image.open("../sunny_mountain.png"))
