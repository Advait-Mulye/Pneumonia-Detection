import numpy as np
import tensorflow as tf
import cv2
import os

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    # Build a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        class_output = predictions[:, class_index]

    # Extract gradients and the conv layer output
    grads = tape.gradient(class_output, conv_outputs)

    # Pool the gradients across the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weigh the conv layer outputs with the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to the heatmap and normalize it
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def save_and_overlay_gradcam(model, img_array, original_img_path, label, last_conv_layer_name="Conv_1"):
    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Read and resize original image
    original_img = cv2.imread(original_img_path)
    original_img = cv2.resize(original_img, (224, 224))

    # Resize heatmap to image size and apply colormap
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # Save the superimposed image
    filename = os.path.basename(original_img_path)
    heatmap_filename = f"heatmap_{filename}"
    heatmap_path = os.path.join("static/uploads", heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed_img)

    return heatmap_filename


