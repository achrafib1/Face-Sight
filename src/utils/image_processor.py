import cv2
import torch
from typing import Any, Tuple


def process_image(
    image: Any, model: Any, non_max_suppression: Any
) -> Tuple[Any, Any, torch.Tensor, Tuple[int, int]]:
    """
    Processes an image for object detection.

    Parameters:
    - image: The original image.
    - model: The object detection model to use.
    - non_max_suppression: The non-maximum suppression function to apply to the model's predictions.

    Returns:
    - pred: The model's predictions after non-maximum suppression.
    - image: The resized image.
    - img: The reshaped and normalized image tensor.
    - original_size: The original size of the image.
    """

    # Save the original image size
    original_size = image.shape[:2]

    # Set the model's stride
    stride = int(model.stride.max())

    # Compute the new size of the image
    new_size = (
        image.shape[1] - image.shape[1] % stride,
        image.shape[0] - image.shape[0] % stride,
    )

    # Resize the image
    image = cv2.resize(image, new_size)

    # Convert the image to a tensor
    img = torch.from_numpy(image.transpose((2, 0, 1)))

    # Reshape and normalize the image
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run the model on the image
    pred = model(img)[0]

    # Apply non-maximum suppression to the predictions
    pred = non_max_suppression(pred)

    return pred, image, img, original_size
