import io
import os
import tarfile

from PIL import Image
from tqdm import tqdm


def get_image_by_key(key: str, root: str) -> Image.Image | None:
    """Get an image by its key.

    Args:
    ----
        key (str): Image key.
        root (str): Root directory where the tar files are stored.

    """
    assert key is not None and len(key) > 5, "Key must be at least 5 characters long"

    tar_key = key[:5]
    corresponding_tar_file = os.path.join(root, f"{tar_key}.tar")

    with tarfile.open(corresponding_tar_file) as tar:
        if image_data := tar.extractfile(f"{key}.jpg"):
            image_data = image_data.read()
            image_original = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image_original = None

    return image_original


def get_images_by_key(keys: list[str], root: str) -> dict[str, Image.Image | None]:
    """Get images by their keys.

    Args:
    ----
        keys (list[str]): List of image keys.
        root (str): Root directory where the tar files are stored.

    """
    assert keys is not None and all(
        key is not None and len(key) > 5 for key in keys
    ), "All keys must be at least 5 characters long"

    # Group images by tar to perform batch read
    img2tar = {}

    for x in keys:
        prefix = x[:5]
        if prefix not in img2tar:
            img2tar[prefix] = []

        img2tar[prefix].append(x)

    # Iterate over tar files
    all_images = {}
    for tar_key, images in tqdm(img2tar.items()):
        corresponding_tar_file = os.path.join(root, f"{tar_key}.tar")

        with tarfile.open(corresponding_tar_file) as tar:
            for key in images:
                if image_data := tar.extractfile(f"{key}.jpg"):
                    image_data = image_data.read()
                    image_original = Image.open(io.BytesIO(image_data)).convert("RGB")
                else:
                    image_original = None
                all_images[key] = image_original

    return all_images
