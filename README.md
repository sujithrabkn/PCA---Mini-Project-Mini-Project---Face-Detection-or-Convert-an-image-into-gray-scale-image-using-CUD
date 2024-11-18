# PCA-Mini-Project-Convert-an-image-into-gray-scale-image-using-CUDA

## AIM:

The aim of this program is to convert a color image into a grayscale image using CUDA GPU programming for efficient parallel processing.

## REQIREMENTS:

Python,Libraries(NumPy,OpenCV,Numba,Matplotlib),CUDA-enabled GPU,Google Colab 

## ALGORITHM:

1. Set up the environment
2. Install Required Libraries
3. Import Required Libraries
4. Define the CUDA Kernel for Grayscale Conversion
5. Load the Image
6. Convert the Image to RGB Format
7. Prepare the Output Array
8. Allocate Memory on the GPU
9. Set Up Grid and Block Dimensions
10. Launch the CUDA Kernel
11. Copy the Result Back to Host Memory
12. Display the Grayscale Image
13. Display the Original Image
14. Run the Program

## PROGRAM:

```
Name: SUJITHRA B K N
Regno: 212222230153
```

```
import numpy as np
import cv2
from numba import cuda
from matplotlib import pyplot as plt

# CUDA kernel for grayscale conversion
@cuda.jit
def rgb_to_grayscale_kernel(img, out_img):
    height, width, _ = img.shape
    x, y = cuda.grid(2)
    if x < width and y < height:
        r = img[y, x, 0]
        g = img[y, x, 1]
        b = img[y, x, 2]
        # Standard formula for converting RGB to grayscale
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        out_img[y, x] = gray

# Load an image from a local file
image_path = '/content/anime.jpg'  # Replace with your local file path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Image not found. Please check the path.")

# Convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create an output image array (grayscale image)
output_image = np.zeros_like(image[:, :, 0], dtype=np.float32)

# Allocate memory on the GPU
d_img = cuda.to_device(image)
d_out_img = cuda.to_device(output_image)

# Set up the grid and block dimensions
threads_per_block = (16, 16)  # Block size (16x16 threads)
blocks_per_grid = (int(np.ceil(image.shape[1] / threads_per_block[0])),
                   int(np.ceil(image.shape[0] / threads_per_block[1])))  # Grid size

# Launch the kernel to convert the image to grayscale
rgb_to_grayscale_kernel[blocks_per_grid, threads_per_block](d_img, d_out_img)

# Copy the result back to host memory
d_out_img.copy_to_host(output_image)

# Display both the original and grayscale images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the grayscale image
axes[1].imshow(output_image, cmap='gray')
axes[1].set_title('Grayscale Image')
axes[1].axis('off')

# Show the plot
plt.show()

```

## OUTPUT:
![download](https://github.com/user-attachments/assets/f04c29ae-24f1-4cc7-9085-9b488e5732f0)

## RESULT:

The conversion of an input image into a grayscale image using CUDA GPU programming for parallel processing ia implemented successfully.
