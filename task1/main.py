import cv2 as cv
import numpy as np

def cosec(theta):
    return float(1 / np.sin(theta))

def sec(theta):
    return float(1 / np.cos(theta))


def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filter(img):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64)
    sobel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], dtype=np.float64)
    img_x = cv.filter2D(img, -1, sobel_x)
    img_y = cv.filter2D(img, -1, sobel_y)
    sobel_combined = np.sqrt(img_x**2 + img_y**2) # Calculating gradient
    theta = np.arctan2(img_y, img_x)
    edges = sobel_combined.astype(np.uint8) # Normalizing the values of pixels
    # theta = np.arctan2(sobel_y, sobel_x)
    return (edges, theta)

def non_max_suppression(img, D): # Here D is the angle matrix calculated using theta in above sobel_filter function
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.float64) # Initialize matrix of same size as image with zeroes 
    angle = D * 180. / np.pi # Convert in degrees
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
                # q = 255
                # r = 255 # These are the pixel values to be given to black pixels for suppressing the edges
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
    
    return Z
rho =9
theta1=0.261
threshold=220
def hough_lines(edges: np.ndarray, threshold: float, min_theta: float, max_theta: float) -> np.ndarray:
    # Initialize the counter matrix in polar coordinates
    #*****************
    diagonal = np.sqrt(edges.shape[0]**2 + edges.shape[1]**2)
    #*****************
    # Compute the values for the thetas and the rhos
    theta_angles = np.arange(min_theta, max_theta, theta1)
    rho_values = np.arange(-diagonal, diagonal, rho)
    # Compute the dimension of the accumulator matrix
    num_thetas = len(theta_angles)
    num_rhos = len(rho_values)
    accumulator = np.zeros([num_rhos, num_thetas])
     # Pre-compute sin and cos
    sins = np.sin(theta_angles)
    coss = np.cos(theta_angles)
    
    # Consider edges only
    ys, xs = np.where(edges > 0)
    
    for x,y in zip(xs,ys):
        for t in range(num_thetas):
            # compute the rhos for the given point for each theta
            current_rho = x * coss[t] + y * sins[t]
            # for each rho, compute the closest rho among the rho_values below it
            # the index corresponding to that rho is the one we will increase
            rho_pos = np.where(current_rho > rho_values)[0][-1]
            #rho_pos = np.argmin(np.abs(current_rho - rho_values))
            accumulator[rho_pos, t] += 1
    final_rho_index, final_theta_index = np.where(accumulator > threshold)
    final_rho = rho_values[final_rho_index]    
    final_theta = theta_angles[final_theta_index]
    polar_coordinates = np.vstack([final_rho, final_theta]).T
    return polar_coordinates


    
img = cv.imread("table.png")
img=cv.resize(img,(600,800))
# cv.resize(img, (0, 0), fx = 0.2, fy = 0.2)
grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_ary = np.array(grayscale_img, dtype=np.float64) # Image has been converted to np array with pixel values as floating point numbers
blurred_image = cv.filter2D(img_ary, -1, gaussian_kernel(5, sigma=1.3))
edges, theta = sobel_filter(blurred_image)
#print(img.shape)
suppressed_img = non_max_suppression(edges, theta)
suppressed_img=suppressed_img.astype(np.uint8)
#cv.imshow('edge detection', suppressed_img.astype(np.uint8))
_, binary_image = cv.threshold(suppressed_img,35, 255, cv.THRESH_BINARY)
lines=hough_lines(binary_image,threshold,-np.pi/2,np.pi/2)
for x in lines:
    rho=x[0]
    theta=x[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0=rho*a
    y0=rho*b
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv.imshow('Binary image',binary_image)
cv.imshow('Original image',img)
cv.waitKey(0)  
cv.destroyAllWindows() 