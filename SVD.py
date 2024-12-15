import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread

# Function to compress an image
def CompressImage(image):
    #image = image.astype(float)
    # Perform SVD on the image
    U, S, VT = np.linalg.svd(image, full_matrices=False)

    # Compute cumulative energy to decide r
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.argmax(energy >= 0.99) + 1  # Rank that retains 95% energy

    # Keep only the first r components
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    VTr = VT[:r, :]
    
    # Reconstruct the rank-r approximation
    compressed = Ur @ Sr @ VTr
    
    return compressed, r

# Function to compress a color image
def CompressColorImage(image):
    if image.shape[-1] != 3:
        raise ValueError("Expected a color image with 3 channels (RGB).")

    red_image = image[:, :, 0]
    green_image = image[:, :, 1]
    blue_image = image[:, :, 2]
    
    # Get ranks for each channel
    _, r_red = CompressImage(red_image)
    _, r_green = CompressImage(green_image)
    _, r_blue = CompressImage(blue_image)

    # Use the maximum rank for consistency
    r = max(r_red, r_green, r_blue)
    print(f"Using maximum rank across channels: R={r}")

    # Re-compress each channel using the maximum rank
    red_comp = compress_channels(red_image, r)
    green_comp = compress_channels(green_image, r)
    blue_comp = compress_channels(blue_image, r)

    # Create a temporary array to hold the new found channels
    comp_image = np.zeros_like(image, dtype=np.float32)
    comp_image[:, :, 0] = red_comp
    comp_image[:, :, 1] = green_comp
    comp_image[:, :, 2] = blue_comp

    return np.clip(comp_image, 0, 255).astype(np.uint8), red_comp, green_comp, blue_comp, r

# Function to compress individual channels
def compress_channels(image, rank):
    # Perform SVD
    U, S, VT = np.linalg.svd(image.astype(float), full_matrices=False)
    # Keep only rank (r) elements
    Ur, Sr, VTr = U[:, :rank], np.diag(S[:rank]), VT[:rank, :]
    # Perform dot product to get your reconstructed channel
    compressed_channel =  Ur @ Sr @ VTr#np.dot(Ur, np.dot(Sr, VTr))
    return compressed_channel

color = False
if color == True:
    A = imread("flower.jpg")
    A_r, red, green, blue, r = CompressColorImage(A)
else:
    image = imread("flower.jpg")
    A = rgb2gray(image)
    A_r, r = CompressImage(A)

#A = A.astype(np.float32)
#A_r = A_r.astype(np.float32)
print(f'Original Image dtype: {A.dtype}, shape: {A.shape}')
print(f'Compressed Image dtype: {A_r.dtype}, shape: {A_r.shape}')

# Find the difference between the reduced and original image
difference = A - A_r
# Calculate Frobenius norm manually
error = np.sqrt(np.sum(difference ** 2))
relative_error = error / np.sqrt(np.sum(A ** 2))

print(f"Frobenius Norm (Error): {error}")
print(f"Relative Approximation Error: {relative_error:.4f}")

print(f"Rank: {r}, Approximation Error: {relative_error:.4f}") 

plt.figure(figsize=(15, 4))

if color == True:
    # Approximated image
    plt.subplot(1, 5, 2)
    plt.imshow(red, cmap="Reds")
    plt.title(f'Red Channel')
    plt.axis('off')

    # Approximated image
    plt.subplot(1, 5, 3)
    plt.imshow(green, cmap="Greens")
    plt.title(f'Green Channel')
    plt.axis('off')

    # Approximated image
    plt.subplot(1, 5, 4)
    plt.imshow(blue, cmap="Blues")
    plt.title(f'Blue Channel')
    plt.axis('off')

    # Original image
    plt.subplot(1, 5, 1)
    plt.imshow(A/255)
    plt.title('Original Image')
    plt.axis('off')

    # Approximated image
    plt.subplot(1, 5, 5)
    plt.imshow(A_r/255)
    plt.title(f'Rank-{r} Approximation')
    plt.axis('off')

else:
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(A/255, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Approximated image
    plt.subplot(1, 2, 2)
    plt.imshow(A_r/255, cmap='gray')
    plt.title(f'Rank-{r} Approximation')
    plt.axis('off')

plt.tight_layout()
plt.show()