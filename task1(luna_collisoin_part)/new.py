import cv2
import numpy as np

# Load the left and right images
left_image = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

def image_match(left_img, right_img, grid, max_disparity):
    # Get the height and width of the left image
    h, w = left_img.shape
    
    # Initialize a matrix to store the disparity map
    matrix = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate half of the grid size
    size1 = grid // 2

    # Iterate through every pixel in the image
    for y in range(size1, h - size1):
        for x in range(size1, w - size1):
            # Initialize variables to store the best disparity and minimum difference
            best_disparity = 0
            min_difference = float('inf')

            # Extract the grid around the pixel from the left image
            left_grid = left_img[y - size1:y + size1 + 1, x - size1:x + size1 + 1]

            # Iterate over the range of disparities
            for disparity in range(max_disparity):
                # Check for boundary conditions
                if x - disparity < size1:
                    break

                # Extract the corresponding grid from the right image
                right_grid = right_img[y - size1:y + size1 + 1, x - disparity - size1:x - disparity + size1 + 1]
                
                # Calculate the absolute difference between the grids
                diff = np.sum(np.abs(left_grid - right_grid))

                # Update the best disparity if the difference is smaller
                if diff < min_difference:
                    min_difference = diff
                    best_disparity = disparity

            # Store the best disparity in the matrix
            matrix[y, x] = best_disparity * (255 // max_disparity)

    return matrix

max_disparity = 16  # Maximum disparity range
grid = 20  # Size of the block for matching

# Perform stereo matching to obtain the disparity map
matrix = image_match(left_image, right_image, grid, max_disparity)

# Apply a color map to visualize the disparity map
depth_map = cv2.applyColorMap(matrix, cv2.COLORMAP_JET)

# Display the depth map
cv2.imshow('Depth_map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
