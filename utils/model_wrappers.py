"""
This module contains wrappers for various models to allow for a unified interface for point tracking.
Each wrapper takes in raw images read from the dataset and outputs tracked points in a consistent format.
"""


import torch
import numpy as np
from PIL import Image

from models.tapnet_utils import transforms

from models.tapnext.tapnext_torch import TAPNext
from models.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

class TAP:
    def __init__(self, device="cuda", inference_resolution=(256, 256)):
        """Initializes the TAP tracker.
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu').
            inference_resolution (tuple): Resolution for inference (h, w).
        """
        self.device = device
        self.inference_resolution = inference_resolution

    def flip_points(self, points):
        """Flips the points from (x, y) to (y, x) format.

        Args:
            points (np.ndarray): Array of points in (x, y) format.

        Returns:
            np.ndarray: Points in (y, x) format.
        """
        return points[:, [1, 0]]

    def resize_points(self, points, original_resolution, t = 0):
        """Resizes points to the inference resolution.

        Args:
            points (np.ndarray): Array of points in (x, y) format.
            original_resolution (tuple): Original image size (h, w).

        Returns:
            np.ndarray: Resized points in (y, x) format.
        """
        points_resized = []
        for i in range(len(points)):
            points_resized.append(
                [
                    t,
                    points[i][0] * self.inference_resolution[1]
                    / original_resolution[0],
                    points[i][1] * self.inference_resolution[0]
                    / original_resolution[1],
                ]
            )
        return np.array(points_resized)

    def reshape_image(self, image):
        """Reshapes the image to the required input format for the model.

        Args:
            image (np.ndarray): Input image in HWC format.

        Returns:
            np.ndarray: Reshaped image in the format expected by the model.
        """

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        image = Image.fromarray(image)
        image = image.resize(self.inference_resolution, Image.LANCZOS)
        image = np.array(image)

        return image

    def preprocess_frames(self, frames):
        """Preprocess frames to model inputs.

        Args:
            frames: [num_frames, height, width, 3], [0, 255], np.uint8

        Returns:
            frames: [num_frames, height, width, 3], [-1, 1], np.float32
        """
        frames = frames.float()
        frames = frames / 255 * 2 - 1
        return frames

    def track(self, images, query_points):
        """Track points across a sequence of images.

        Args:
            images (list of np.ndarray): List of images in HWC format.
            query_points (np.ndarray): Points to track in (x, y) format.

        Returns:
            np.ndarray: Tracked points in (y, x) format.
        """
        raise NotImplementedError(
                        "This method should be implemented by subclasses."
                        )


class TAPNextWrapper(TAP):
    def __init__(
        self,
        device="cuda",
        inference_resolution=(256, 256),
        ckpt_path="weights/bootstapnext_ckpt.npz",
        fine_tuned_weights = None
    ):
        """
        Initializes the TAPNext tracker.

        Args:
            initial_points (np.ndarray): Array of starting points (e.g., shape (N, 2)).
            initial_image (np.ndarray): The initial image/frame.
        """
        super().__init__(device, inference_resolution)

        model = TAPNext(image_size=inference_resolution)
        
        if ckpt_path.endswith(".npz"):
            self.model = restore_model_from_jax_checkpoint(model, ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict({k.replace('tapnext.', ''): v for k, v in ckpt['state_dict'].items()})
            self.model = model
        
        if fine_tuned_weights is not None:
            self.model.load_state_dict(
                torch.load(fine_tuned_weights)
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def track(self, images, query_points):
        """Track points across a sequence of images.

        Args:
            images (list of np.ndarray): List of images in HWC format.
            query_points (np.ndarray): Points to track in (x, y) format.

        Returns:
            np.ndarray: Tracked points in (y, x) format.
        """

        image_shape = images[0].shape
        number_of_images = len(images)

        images = [self.reshape_image(image) for image in images]
        images = [
            self.preprocess_frames(
                torch.tensor(image, dtype=torch.float32, device=self.device)
            )
            for image in images
        ]
        images = torch.stack(images, dim=0).unsqueeze(0)

        points = self.flip_points(query_points)
        points = self.resize_points(points, (image_shape[0], image_shape[1]))
        points = torch.tensor(points, dtype=torch.float32, device=self.device)

        pred_tracks, track_logits, visible_logits, point_tokens, tracking_state = self.model(
            video=images[:, :1],
            query_points=points.unsqueeze(0),
        )

        predictions = [
            {
                "tracks": pred_tracks.permute(0, 2, 1, 3),
                "visibles": visible_logits.squeeze(0) > 0,
            }
        ]

        with torch.no_grad():
            for i in range(1, number_of_images):
                pred_tracks, point_tokens, visible_logits, point_tokens, tracking_state = self.model(
                    video=images[:, i:i+1], state=tracking_state
                )

                pred_tracks = pred_tracks.permute(0, 2, 1, 3)
                visible_logits = visible_logits.squeeze(0)

                predictions.append(
                    {
                        "tracks": pred_tracks,
                        "visibles": visible_logits > 0,
                    }
                )

        tracks = (
            torch.cat([x["tracks"][0] for x in predictions], dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        visibles = (
            torch.cat([x["visibles"][0] for x in predictions], dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        tracks = transforms.convert_grid_coordinates(
            tracks, self.inference_resolution, (image_shape[0], image_shape[1])
        )
        tracks = tracks[..., [1, 0]]

        return tracks, visibles

