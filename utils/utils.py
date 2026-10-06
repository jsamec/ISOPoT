import cv2
import numpy as np
from skimage import color
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
from scipy.spatial.transform import Rotation as R
import warnings
from scipy.ndimage import maximum_filter

import math
from skimage.transform import ProjectiveTransform

import numpy as np
import math
from skimage.transform import ProjectiveTransform


def compute_angle(src, dst, weights=None):
    a1 = np.arctan2(src[:, 1], src[:, 0])
    a2 = np.arctan2(dst[:, 1], dst[:, 0])
    angles = a2 - a1
    return np.average(angles, weights=weights)

def compute_angle_lse(src, dst):
    Sxy = np.sum(src[:,0]*dst[:,1] - src[:,1]*dst[:,0])   # numerator
    Sxx = np.sum(src[:,0]*dst[:,0] + src[:,1]*dst[:,1])   # denominator
    return np.arctan2(Sxy, Sxx)

class BottomCenterRotationTransform(ProjectiveTransform):
    """Transformation that rotates around the bottom-center point and translates only in Y."""

    def __init__(self, rotation=0.0, translation_y=0.0, center=(610, 610)):
        super().__init__()
        cx, cy = center
        cos_t = math.cos(rotation)
        sin_t = math.sin(rotation)

        # Step 1: translate center to origin
        T1 = np.array([
            [1, 0, -cx],
            [0, 1, -cy],
            [0, 0, 1]
        ])

        # Step 2: rotate
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0, 0, 1]
        ])

        # Step 3: Y translation
        T2 = np.array([
            [1, 0, 0],
            [0, 1, translation_y],
            [0, 0, 1]
        ])
        
        # Step 4: translate center back to original position
        T3 = np.array([
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1]
        ])

        # Combine transformations: T2 * R * T1
        self.params = T3 @ T2 @ R @ T1

    def estimate(self, src, dst, src_values, dst_values, center=(610, 610)):
        """Estimate rotation and Y-translation given corresponding points."""
        src = np.asarray(src)
        dst = np.asarray(dst)
        cx, cy = center
        
        # Center points around bottom-center
        src_centered = src - [cx, cy]
        dst_centered = dst - [cx, cy]

        src_values = src_values.astype(float)
        dst_values = dst_values.astype(float)
        luminance_diff = np.abs(src_values - dst_values)
        weights = (1 / (1 + luminance_diff)) ** 2
        
        # TODO: Alternatively, use Kabsch algorithm to find rotation matrix
        rotation = compute_angle(src_centered, dst_centered, weights=weights)
                
        # Compute Y translation (difference in mean Y after undoing rotation)
        R = np.array([
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation),  math.cos(rotation)]
        ])
        src_rot = src_centered @ R.T
        
        translation_y = np.average(dst_centered[:, 1] - src_rot[:, 1], weights=weights)

        # Update params
        self.__init__(rotation=rotation, translation_y=translation_y, center=(cx, cy))
        return True
    
    def residuals(self, src, dst, src_values, dst_values):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the Euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N,) array
            Residual for coordinate.

        """
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))# + np.sqrt(((src_values - dst_values) ** 2)) Add intensity difference?



def nms_general(arr, size=7):
    """
    Non-maximum suppression with generic neighborhood size.

    arr  : input 2D array
    size : window size, e.g. 3, 5, 7

    Returns new array with non-maximum elements suppressed to zero.
    """
    max_filt = maximum_filter(arr, size=size, mode="nearest")
    return (arr == max_filt) * arr


def sobel_keypoints(image, percentile=0.75, mask = None, size = 7, sigma=None) -> np.ndarray:
    
    if sigma is not None:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    if mask is not None:
        gradient_magnitude = gradient_magnitude * mask
        
    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)  # Normalize to [0, 1]
    
    gradient_magnitude = nms_general(gradient_magnitude, size=size)

    #get the 75th percentile of the gradient magnitude of those pixels above zero
    thresh_value = np.percentile(gradient_magnitude[gradient_magnitude > 0], percentile * 100)
    keypoints = np.column_stack(np.where(gradient_magnitude >= thresh_value))
    
    return np.fliplr(keypoints)  # Return as (x, y) coordinates

def estimate_transform(src_points: np.ndarray,
                       dst_points: np.ndarray,
                       src_image: np.ndarray = None,
                       dst_image: np.ndarray = None,
                       transform_type: str = "euclidean",
                       threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the transformation matrix from source to destination points.
    Args:
        src_points (np.ndarray): Source points of shape (M, 2) (only visible ones).
        dst_points (np.ndarray): Destination points of shape (M, 2).
        transform_type (str): Type of transformation to estimate ('euclidean' or 'affine').
        threshold (float): RANSAC residual threshold for outlier rejection.
    Returns:
        tuple[np.ndarray, np.ndarray]: (transform_matrix, inliers)
            - transform_matrix: shape (2, 3)
            - inliers: boolean array of shape (M,)
    """
    inliers = np.zeros(len(src_points), dtype=bool)
    if len(src_points) < 3 or len(dst_points) < 3:
        # Not enough points
        return np.eye(2, 3), inliers

    try:
        if transform_type == "euclidean":
            model, inliers_mask = ransac(
                (src_points, dst_points),
                EuclideanTransform,
                min_samples=4,
                residual_threshold=threshold,
                max_trials=1000,
            )
            transform_matrix = model.params[:2, :3]
            inliers = inliers_mask
        elif transform_type == "euclidean++":
            if src_image is None or dst_image is None:
                raise ValueError("Source and destination images must be provided for 'euclidean++' transform.")
            
            src_values = src_image[src_points[:, 1].astype(int), src_points[:, 0].astype(int)]
            dst_values = dst_image[dst_points[:, 1].astype(int), dst_points[:, 0].astype(int)]
            
            model, inliers_mask = ransac(
                (src_points, dst_points, src_values, dst_values),
                BottomCenterRotationTransform,
                min_samples=4,
                residual_threshold=threshold,
                max_trials=5000,
            )
            transform_matrix = model.params[:2, :3]
            inliers = inliers_mask    
        
        elif transform_type == "affine":
            transform_matrix, inliers_mask = cv2.estimateAffine2D(src_points, dst_points)
            if inliers_mask is not None:
                inliers = inliers_mask.ravel().astype(bool)
            else:
                inliers = np.zeros(len(src_points), dtype=bool)
        else:
            raise ValueError("Unsupported transform type. Use 'euclidean' or 'affine'.")
    except Exception:
        transform_matrix = np.eye(2, 3)

    return transform_matrix, inliers

def find_transformation_between_predictions(
    pred1: np.ndarray,
    pred2: np.ndarray,
    visible1: np.ndarray,
    visible2: np.ndarray,
    src_image: np.ndarray = None,
    dst_image: np.ndarray = None,
    threshold: float = 5.0,
    transform_type: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the transformation matrix between two sets of predicted points.
    Returns:
        (transform_matrix, inliers_full)
        - transform_matrix: (2, 3)
        - inliers_full: boolean array of shape (N,)
    """
    N = len(pred1)
    visible_mask = visible1 & visible2
    pts1 = pred1[visible_mask]
    pts2 = pred2[visible_mask]

    if pts1.shape[0] < 3 or pts2.shape[0] < 3:
        warnings.warn("Not enough visible points to estimate transformation.")
        return np.eye(2, 3), np.zeros(N, dtype=bool)

    transform_matrix, inliers_visible = estimate_transform(
        pts1, pts2, src_image, dst_image, 
        transform_type=transform_type, threshold=threshold
    )

    # Expand to full-size inlier mask
    inliers_full = np.zeros(N, dtype=bool)
    inliers_full[visible_mask] = inliers_visible

    return transform_matrix, inliers_full

def estimate_motion(T: np.ndarray, resolution: float, image_shape: tuple) -> tuple:
    """Estimate the motion from the transformation matrix.
    Args:
        T (np.ndarray): Transformation matrix of shape (2, 3).
    Returns:
        tuple: Estimated motion as (dx[cm], dy[cm], yaw[rad]).
    """

    # The transformation is calculated from the origin point
    # of the sonar beams on the image (bottom of image at the center)
    origin = [image_shape[1] // 2, image_shape[0]]
    transformed_origin = np.dot(T, np.array([origin[0], origin[1], 1]))
    delta_origin = (
        -np.array(
            [transformed_origin[1] - origin[1], transformed_origin[0] - origin[0]]
        ) * resolution
    )

    rotation = np.arctan2(T[1, 0], T[0, 0])
    #rotation = np.deg2rad(rotation)
    return -delta_origin[0], -delta_origin[1], rotation


def get_relative_pose(ref_pose, target_pose):
    """
    Transforms the target pose (x, y, yaw) into the local frame of the reference pose.

    Parameters:
        ref_pose (tuple): Reference pose (x0, y0, yaw0)
        target_pose (tuple): Target/global pose (x1, y1, yaw1)

    Returns:
        tuple: (dx_local, dy_local, dyaw) in the reference frame
    """
    x0, y0, yaw0 = ref_pose
    x1, y1, yaw1 = target_pose

    # Translation in global frame
    dx_global = x1 - x0
    dy_global = y1 - y0

    # Rotate into local frame
    dx_local = np.cos(yaw0) * dx_global + np.sin(yaw0) * dy_global
    dy_local = -np.sin(yaw0) * dx_global + np.cos(yaw0) * dy_global

    # Relative rotation
    dyaw = yaw1 - yaw0

    return [dx_local, dy_local, dyaw]


def get_pose(pose_data):
    pose_translation = pose_data["transform"]["translation"]
    pose_rotation = pose_data["transform"]["rotation"]
    x1, y1, _ = pose_translation["x"], pose_translation["y"], pose_translation["z"]
    qx1, qy1, qz1, qw1 = pose_rotation["x"], pose_rotation["y"], pose_rotation["z"], pose_rotation["w"]
    
    yaw = R.from_quat([qx1, qy1, qz1, qw1]).as_euler("xyz")[2]
    
    return (x1, y1, yaw)

def get_full_pose(pose_data):
    pose_translation = pose_data["transform"]["translation"]
    pose_rotation = pose_data["transform"]["rotation"]
    x1, y1, z1 = pose_translation["x"], pose_translation["y"], pose_translation["z"]
    qx1, qy1, qz1, qw1 = pose_rotation["x"], pose_rotation["y"], pose_rotation["z"], pose_rotation["w"]
    
    return (x1, y1, z1, qx1, qy1, qz1, qw1)

def filter_image_by_frequency(img, bands_to_keep=("low", "mid"), cutoffs=(30, 80, 200)):
    """
    Removes selected frequency bands from an image using a 2D FFT.
    
    Parameters
    ----------
    img : ndarray
        Input image (RGB or grayscale).
    bands_to_keep : tuple of str
        Which frequency bands to keep. Options: ('low', 'mid', 'high').
    cutoffs : tuple of (low_cut, mid_cut, high_cut)
        Radii defining the frequency bands in pixels.
    
    Returns
    -------
    filtered_img : ndarray
        The reconstructed image with only the chosen frequency bands kept.
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        img = color.rgb2gray(img)
    
    # FFT
    fshift = np.fft.fftshift(np.fft.fft2(img))
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    def band_mask(low, high):
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        return (distance >= low) & (distance <= high)
    
    # Define frequency bands
    low_cut, mid_cut, high_cut = cutoffs
    bands = {
        "low": band_mask(0, low_cut),
        "mid": band_mask(low_cut, mid_cut),
        "high": band_mask(mid_cut, high_cut)
    }
    
    # Combine selected bands
    combined_mask = np.zeros_like(fshift, dtype=bool)
    for b in bands_to_keep:
        combined_mask |= bands[b]
    
    # Apply combined mask and inverse FFT
    f_filtered = fshift * combined_mask
    filtered_img = np.fft.ifft2(np.fft.ifftshift(f_filtered)).real
    return filtered_img