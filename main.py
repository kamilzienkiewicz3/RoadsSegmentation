import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import hough_line, hough_line_peaks


def load_images(filepath):
    img_bgr = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_rgb, img_gray




def get_rgb_mask(img_rgb):
    lower_rgb = np.array([160, 160, 0])   
    upper_rgb = np.array([200, 200, 200])
    mask_rgb_raw = cv2.inRange(img_rgb, lower_rgb, upper_rgb)
    mask_rgb_inverted = cv2.bitwise_not(mask_rgb_raw)
    
    kernel_open_rgb = np.ones((10, 10), np.uint8)
    mask_rgb_open = cv2.morphologyEx(mask_rgb_inverted, cv2.MORPH_OPEN, kernel_open_rgb)
    
    kernel_close_rgb = np.ones((12, 12), np.uint8)
    mask_rgb_final = cv2.morphologyEx(mask_rgb_open, cv2.MORPH_CLOSE, kernel_close_rgb)
    
    return mask_rgb_final




def get_hsv_mask(img_bgr):
    img_blurred = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    img_hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    
    lower_hsv = np.array([0, 0, 40]) 
    upper_hsv = np.array([180, 60, 220])
    mask_hsv_raw = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    
    kernel_open_hsv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_hsv_open = cv2.morphologyEx(mask_hsv_raw, cv2.MORPH_OPEN, kernel_open_hsv)
    
    kernel_close_hsv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_hsv_final = cv2.morphologyEx(mask_hsv_open, cv2.MORPH_CLOSE, kernel_close_hsv)
    
    return mask_hsv_final



def get_adaptive_mask(img_gray):
    img_blurred = cv2.GaussianBlur(img_gray, (11, 11), 0)
    
    mask_adaptive_raw = cv2.adaptiveThreshold(
        img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 4
    )
    
    kernel_open_adv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_adv_open = cv2.morphologyEx(mask_adaptive_raw, cv2.MORPH_OPEN, kernel_open_adv)
    
    kernel_close_adv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_adv_final = cv2.morphologyEx(mask_adv_open, cv2.MORPH_CLOSE, kernel_close_adv)
    
    return mask_adv_final



def get_canny_edges(img_gray):
    img_blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)
    
    edges = cv2.Canny(img_blurred, 100, 200)
    
    return edges

def get_hough_lines(edges):
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    peaks = hough_line_peaks(h, theta, d)
    
    return peaks

def display_hough_results(img_rgb, edges, hough_peaks):
    plt.figure(figsize=(16, 8))
    
    ax0 = plt.subplot(1, 2, 1)
    ax0.imshow(img_rgb)
    ax0.set_title("Wykryte linie na RGB")
    ax0.axis('off')
    
    accumulators, angles, dists = hough_peaks
    
    for _, angle, dist in zip(accumulators, angles, dists):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax0.axline((x0, y0), slope=np.tan(angle + np.pi / 2), color="red", linewidth=3)
    
    ax1 = plt.subplot(1, 2, 2)
    ax1.imshow(edges, cmap='gray')
    ax1.set_title("Krawędzie Canny")
    ax1.axis('off')
    
    plt.tight_layout()
    plt.show()


def save_masks(mask_rgb, mask_hsv, mask_adv):
    cv2.imwrite("maska_RGB.png", mask_rgb)   
    cv2.imwrite("maska_HSV.png", mask_hsv)
    cv2.imwrite("maska_adaptive.png", mask_adv)


def display_results(mask_rgb, mask_hsv, mask_adv):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask_rgb, cmap='gray')
    plt.title("Maska RGB")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_hsv, cmap='gray')
    plt.title("Maska HSV")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask_adv, cmap='gray')
    plt.title("Maska Adaptacyjna")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def main():
    img_bgr, img_rgb, img_gray = load_images("StandardResolution.tiff")
    
    mask_rgb = get_rgb_mask(img_rgb)
    mask_hsv = get_hsv_mask(img_bgr)
    mask_adv = get_adaptive_mask(img_gray)
    edges = get_canny_edges(img_gray)
    hough_peaks = get_hough_lines(edges)
    display_hough_results(img_rgb, edges, hough_peaks)
    
    save_masks(mask_rgb, mask_hsv, mask_adv)
    display_results(mask_rgb, mask_hsv, mask_adv)

if __name__ == "__main__":
    main()