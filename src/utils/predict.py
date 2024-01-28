from model_loader import load_model
from image_processor import process_image
from box_drawer import draw_boxes


def predict(image, model_path, faces):
    # Load the model
    model, names, scale_coords, non_max_suppression, plot_one_box = load_model(
        model_path
    )

    if image is None:
        print("Error: Could not read image")
        return None, None
    else:
        # Process the image
        pred, image, processed_image, original_size = process_image(
            image, model, non_max_suppression
        )

        # Draw boxes on the image
        faces, image_with_boxes = draw_boxes(
            pred,
            image,
            processed_image,
            names,
            original_size,
            faces,
            scale_coords,
            plot_one_box,
        )
        return faces, image_with_boxes
