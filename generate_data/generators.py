import cv2
import numpy as np

# ===================== Decorators =====================

# Define Decorators to get a parametrized function


def decorator_shading(gen_shading):
    def gen_function_noise(parameters):
        def gen_shading_withparameters(im):
            return gen_shading(im, parameters)

        return gen_shading_withparameters

    return gen_function_noise


def decorator_noise(gen_noise):
    def gen_function_noise(parameters):
        def gen_noise_withparameters(im):
            return gen_noise(im, parameters)

        return gen_noise_withparameters

    return gen_function_noise


def decorator_im(gen_im):
    def gen_function(parameters):
        def gen_im_withparameters():
            return gen_im(parameters)

        return gen_im_withparameters

    return gen_function


# ===================== Generators Noise =====================

# Here you call gen_noise_uniform(parameters) to get a parametrized generators of noise


@decorator_noise
def gen_noise_uniform(image_arr, parameters):
    min_val = parameters["min_val"]
    max_val = parameters["max_val"]

    a, b = np.random.uniform(min_val, max_val, 2)

    noise = np.random.normal(0, np.sqrt(a * image_arr + b), image_arr.shape)
    noisy_image = image_arr + noise

    return noisy_image


@decorator_noise
def gen_noise_beta(image_arr, parameters):
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    min_val = parameters["min_val"]
    max_val = parameters["max_val"]

    result = (np.random.beta(alpha, beta, 2)) * (max_val - min_val) + min_val
    a, b = result[0], result[1]

    noise = np.random.normal(0, np.sqrt(a * image_arr + b), image_arr.shape)

    noisy_image = image_arr + noise

    return noisy_image


# ===================== Generators Images =====================

# Same process : get generators of images


# Generate uniform patchs : parameters : side and channels
@decorator_im
def gen_patch_uniform(parameters):
    side = parameters["side"]
    channels = parameters["channels"]
    intensity = np.random.uniform(low=0, high=1, size=channels)

    im_arr = np.full((side, side, channels), 1.0)

    return np.einsum("ijk,k->ijk", im_arr, intensity)


# Generate an image with random shapes in it
@decorator_im
def generate_random_shapes(parameters):
    # parameters
    side = parameters["side"]
    channels = parameters["channels"]
    num_shapes = parameters["num_shapes"]

    image_size = (side, side, channels)

    # background intensity
    back = np.random.uniform(0, 1)

    # Generate background
    img = np.full((side, side, channels), back) * 255.0
    img = img.astype(np.uint8)

    for _ in range(num_shapes):
        shape_type = np.random.choice(["circle", "rectangle", "polygon"])

        # Random positions and sizes
        x1, y1 = (
            np.random.randint(0, image_size[0] // 2),
            np.random.randint(0, image_size[1] // 2),
        )
        x2, y2 = (
            np.random.randint(image_size[0] // 2, image_size[0]),
            np.random.randint(image_size[1] // 2, image_size[1]),
        )
        grayscale_value = (np.random.randint(0, 256),)  # Grayscale color (0-255)

        if shape_type == "circle":
            # Calculate center and radius for the circle
            center = (
                np.random.randint(0, image_size[0]),
                np.random.randint(0, image_size[1]),
            )
            radius = np.random.randint(10, image_size[0] // 4)
            cv2.circle(img, center, radius, color=grayscale_value, thickness=-1)

        elif shape_type == "rectangle":
            # Draw a rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color=grayscale_value, thickness=-1)

        elif shape_type == "polygon":
            # Generate random polygon with 3-6 vertices
            num_vertices = np.random.randint(3, 6)
            vertices = np.array(
                [
                    [
                        (
                            np.random.randint(0, image_size[0]),
                            np.random.randint(0, image_size[1]),
                        )
                        for _ in range(num_vertices)
                    ]
                ],
                dtype=np.int32,
            )
            vertices = vertices.reshape((-1, 1, 2))  # Reshape to (num_points, 1, 2)
            cv2.fillPoly(img, [vertices], color=grayscale_value)

    img = img.astype("float") / 255
    return img


# Generate a sequence of alternate bands in an image
# You can get transpose images to get get horizontal bands
@decorator_im
def gen_freq(parameters):
    side = parameters["side"]
    freq = parameters["frequence"]

    low = np.random.uniform(0, 1)
    high = np.random.uniform(0, 1)

    array = np.zeros((side, side))
    indexes = np.arange(side)
    dup = np.tile(indexes, (200, 1))
    dup = dup // freq
    array[(dup % 2) == 0] = low
    array[(dup % 2) == 1] = high

    return array.reshape((side, side, 1))


# ===================== shading generator ===================


@decorator_shading
def illuminate_image(image, parameters):
    """
    Illuminate a 2D image with a light source in 3D space.

    Parameters:
        image (numpy.ndarray): 2D array with values in range [0, 1].
        light_position (tuple): (x, y, z) position of the light source in 3D.

    Returns:
        numpy.ndarray: Illuminated image.
    """

    h, w = image.shape[0], image.shape[1]
    image = image.reshape((h, w))
    # Ensure the image is 2D
    if image.ndim != 2:
        raise ValueError("Image must be a 2D array.")

    light_z = parameters["elevation"]
    x_range_min = parameters["x_range"]["min"]
    x_range_max = parameters["x_range"]["max"]

    x_range_min = parameters["x_range"]["min"]
    x_range_max = parameters["x_range"]["max"]

    y_range_min = parameters["y_range"]["min"]
    y_range_max = parameters["y_range"]["max"]

    light_x = np.random.uniform(x_range_min, x_range_max)
    light_y = np.random.uniform(y_range_min, y_range_max)

    # Image dimensions

    # Compute gradients (approximates the surface normals)
    grad_y, grad_x = np.gradient(image)

    # Normalize the gradients to get unit normals
    normal_magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1)
    normals = np.stack(
        (
            -grad_x / normal_magnitude,
            -grad_y / normal_magnitude,
            np.ones_like(image) / normal_magnitude,
        ),
        axis=-1,
    )

    # Create a grid of pixel coordinates (assume z=0 for the image plane)
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    z_coords = np.zeros_like(image)

    # Light direction vector

    light_x = light_x * h
    light_y = light_y * w
    light_z = light_z * np.min(image.shape)

    light_vector = np.array(
        [light_x - x_coords, light_y - y_coords, light_z - z_coords]
    )
    light_vector = light_vector * 0.01
    light_vector_magnitude = np.sqrt(np.sum(light_vector**2, axis=0))

    # falloff = 1 / (1 + (light_vector_magnitude/w/1000)**2)
    falloff = 1

    light_direction = light_vector / light_vector_magnitude
    # Fix axis alignment
    light_direction = np.moveaxis(light_direction, 0, -1)  # Shape (h, w, 3)

    # Compute the dot product of the light direction and the surface normal
    illumination = np.sum(normals * light_direction, axis=-1)
    illumination = np.clip(illumination, 0, 1)  # Ensure non-negative illumination
    illumination = illumination * falloff

    # Apply the illumination to the image
    illuminated_image = image * illumination

    return illuminated_image
