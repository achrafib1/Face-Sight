from .model_loader import load_model
from .image_processor import process_image
from .box_drawer import draw_boxes


def predict(
    image,
    model,
    names,
    faces,
    scale_coords,
    non_max_suppression,
    plot_one_box,
    strategies,
    background,
    color="#56ecd5",
    image_replacement=None,
):
    """
    Predicts the bounding boxes for the detected faces in the image and applies various strategies.

    Parameters:
    - image: The original image.
    - model: The object detection model to use.
    - names: The names of the classes.
    - faces: A list to store the detected faces.
    - scale_coords: The function to rescale the coordinates to the original image size.
    - non_max_suppression: The non-maximum suppression function to apply to the model's predictions.
    - plot_one_box: The function to draw a bounding box on the image.
    - strategies: A list of strategies to apply to the image.
    - background: The desired background color in hexadecimal format.
    - color: The desired color for the faces in hexadecimal format.
    - image_replacement: The replacement image as a numpy array.

    Returns:
    - faces: The list of detected faces.
    - boxes: The list of bounding boxes for each detected face.
    - image_with_boxes: The image with the bounding boxes drawn and the strategies applied.
    """

    if image is None:
        print("Error: Could not read image")
        return None, None, None
    else:
        # Process the image
        pred, image, processed_image, original_size = process_image(
            image, model, non_max_suppression
        )

        # Draw boxes on the image
        boxes, faces, image_with_boxes = draw_boxes(
            pred,
            image,
            processed_image,
            names,
            original_size,
            faces,
            scale_coords,
            plot_one_box,
            strategies,
            background,
            color,
            image_replacement,
        )
        return faces, boxes, image_with_boxes
