import numpy as np
import random
from PIL import Image, ImageFilter
import gradio as gr

def projected_gradient_descent(image, mask, num_iterations, step_size, epsilon, variance, blur_radius, combine):
    # Convert image to numpy array
    img_array = np.array(image).astype(float)

    # Normalize image to range [0,1]
    img_array /= 255.0

    # Define projection function to constrain pixel values to valid range
    def project(x):
        return np.clip(x, 0.0, 1.0)

    # Initialize perturbation vector to zeros
    perturbation = np.zeros_like(img_array)

    # Perform gradient descent for specified number of iterations
    for i in range(num_iterations):
        # Compute gradient of loss function
        gradient = compute_gradient(img_array + perturbation)

        # Update perturbation vector
        perturbation += step_size * gradient

        # Project perturbation vector onto valid range
        perturbation = project(perturbation)

        # Apply perturbation to image
        perturbed_image = img_array + perturbation

        # Project perturbed image onto valid range
        perturbed_image = project(perturbed_image)

        # Check if perturbed image is within epsilon of original image
        if np.linalg.norm(perturbed_image - img_array) <= epsilon:
            break

    # Convert perturbed image back to PIL Image format
    perturbed_image *= 255.0
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_image)

    input_size = perturbed_image.size
    mask = mask.resize(input_size)
    
    perturbed_image = add_masked_noise(perturbed_image, mask, variance)
    perturbed_image = perturbed_image.filter(ImageFilter.GaussianBlur(blur_radius))
    perturbed_image = perturbed_image.filter(ImageFilter.UnsharpMask(radius=blur_radius, percent=200, threshold=2))

    # Create a new image with the same size as the first image
    combined_img = Image.new('RGBA', input_size)

    # Paste the first image onto the new image
    combined_img.paste(perturbed_image, (0, 0))

    # Paste the second image onto the new image with 10% transparency
    combined_img.paste(mask, (0, 0), mask=mask.split()[2].point(lambda x: x*combine))

    
    return combined_img

def compute_gradient(image):
    # Compute gradient of loss function with respect to image
    # This is where you would define your own loss function
    # and compute its gradient using automatic differentiation
    # in a deep learning framework like TensorFlow or PyTorch
    # For the sake of simplicity, we will just use the identity
    # function as our loss function and return the image itself
    return image

def add_masked_noise(image, mask, variance):
    """
    Add masked Gaussian noise to a PIL image.

    Args:
    image (PIL.Image): The input image.
    mask (PIL.Image): The mask image that will be used to mask the noise.
    variance (float): The variance of the Gaussian distribution.

    Returns:
    PIL.Image: The noisy image.
    """
    # Convert the input image and mask to NumPy arrays
    img_array = np.array(image)
    mask_array = np.array(mask)

    # Create a noise array with the same shape as the input image
    noise = np.random.normal(scale=variance, size=img_array.shape)

    # Mask the noise array using the mask image
    masked_noise = np.where(mask_array > 128, noise, 0)

    # Add the masked noise to the input image
    noisy_img_array = np.clip(img_array + masked_noise, 0, 255).astype(np.uint8)

    # Convert the NumPy array back to a PIL image
    noisy_image = Image.fromarray(noisy_img_array)

    return noisy_image

# Example usage
#img = Image.open("example_image.jpg")
#perturbed_img = projected_gradient_descent(img, num_iterations=10, step_size=0.001, epsilon=0.1)
#perturbed_img.show()

app = gr.Interface(
    fn=projected_gradient_descent,
    inputs=[gr.Image(type="pil"), gr.Image(type="pil"), gr.Slider(0, 20, step=1), gr.Slider(0, 0.1, step=0.0001), gr.Slider(0, 10, step=0.1), gr.Slider(0, 100, step=1), gr.Slider(0, 5, step=0.1), gr.Slider(0, 1, step=0.01)],
    outputs=[gr.Image(type="pil")],
    title="Projected Gradient Descent"
)
if __name__ == "__main__":
    app.launch()
