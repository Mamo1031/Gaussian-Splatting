"""Training logic

Implements training loop, loss calculation, and optimization for Gaussian Splatting.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import cv2
from pathlib import Path

from .gaussian_model import GaussianModel
from .renderer import render, Camera


class Trainer:
    """Training class for Gaussian Splatting"""

    def __init__(
        self,
        gaussian_model: GaussianModel,
        cameras: List[Camera],
        images: List[np.ndarray],
        output_dir: str,
        iterations: int = 30000,
        lr: float = 0.01,
        densification_interval: int = 100,
        opacity_reset_interval: int = 3000,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15000,
        densification_grad_threshold: float = 0.0002,
        min_opacity: float = 0.005,
        max_radii2D: int = 1,
    ):
        """Initialize training class.

        Args:
            gaussian_model: GaussianModel instance
            cameras: List of cameras
            images: List of corresponding images
            output_dir: Output directory
            iterations: Number of training iterations
            lr: Learning rate
            densification_interval: Densification interval
            opacity_reset_interval: Opacity reset interval
            densify_from_iter: Iteration to start densification
            densify_until_iter: Iteration to end densification
            densification_grad_threshold: Densification gradient threshold
            min_opacity: Minimum opacity
            max_radii2D: Maximum 2D radius
        """
        self.gaussian_model = gaussian_model
        self.cameras = cameras
        self.images = images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.iterations = iterations
        self.lr = lr
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_grad_threshold = densification_grad_threshold
        self.min_opacity = min_opacity
        self.max_radii2D = max_radii2D

        # Setup optimizer
        self.optimizer = optim.Adam(
            gaussian_model.get_parameters(),
            lr=lr,
            eps=1e-15
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[iterations // 2, iterations * 3 // 4],
            gamma=0.33
        )

        # Background color (white)
        self.bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).cuda()

        # Pipeline settings (simplified)
        self.pipe = type('obj', (object,), {
            'convert_SHs_python': False,
            'compute_cov3D_python': False,
            'debug': False,
        })()

    def compute_loss(
        self,
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute loss.

        Args:
            rendered_image: Rendered image [C, H, W]
            gt_image: Ground truth image [C, H, W]

        Returns:
            (Total loss, loss detail dictionary)
        """
        # L1 loss
        l1_loss = torch.nn.functional.l1_loss(rendered_image, gt_image)

        # SSIM loss (simplified: L1 only)
        loss = l1_loss

        loss_dict = {
            "loss": loss.item(),
            "l1_loss": l1_loss.item(),
        }

        return loss, loss_dict

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: int,
    ):
        """Densify and prune Gaussians.

        Args:
            max_grad: Maximum gradient
            min_opacity: Minimum opacity
            extent: Extent
            max_screen_size: Maximum screen size
        """
        # Simplified: implementation omitted
        # In actual implementation, split and remove Gaussians based on gradients
        pass

    def train(self):
        """Run training."""
        print("Training started...")

        # Convert images to tensors
        gt_images = []
        for img in self.images:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).float().cuda()
                if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                gt_images.append(img_tensor)
            else:
                gt_images.append(img)

        # Training loop
        progress_bar = tqdm(range(self.iterations), desc="Training")
        for iteration in progress_bar:
            # Randomly select camera
            camera_idx = np.random.randint(0, len(self.cameras))
            camera = self.cameras[camera_idx]
            gt_image = gt_images[camera_idx]

            # Render
            rendered_image, rendered_depth = render(
                viewpoint_camera=camera,
                pc=self.gaussian_model,
                pipe=self.pipe,
                bg_color=self.bg_color,
            )

            # Compute loss
            loss, loss_dict = self.compute_loss(rendered_image, gt_image)

            # Backward pass
            loss.backward()

            # Optimize
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update learning rate
            self.scheduler.step()

            # Densify and prune (at regular intervals)
            if iteration % self.densification_interval == 0 and \
               self.densify_from_iter <= iteration < self.densify_until_iter:
                # Simplified: densification implementation omitted
                pass

            # Log output
            if iteration % 100 == 0:
                progress_bar.set_postfix(loss_dict)

            # Save intermediate results
            if iteration % 1000 == 0 and iteration > 0:
                self.save_checkpoint(iteration)
                self.save_render(iteration, camera, rendered_image, gt_image)

        # Save final results
        self.save_checkpoint(self.iterations)
        self.gaussian_model.save_ply(str(self.output_dir / "final_model.ply"))
        print(f"Training completed. Model saved to {self.output_dir / 'final_model.ply'}")

    def save_checkpoint(self, iteration: int):
        """Save checkpoint.

        Args:
            iteration: Iteration number
        """
        checkpoint_path = self.output_dir / f"checkpoint_{iteration}.pth"
        torch.save({
            'iteration': iteration,
            'model_state_dict': {
                'xyz': self.gaussian_model._xyz.state_dict(),
                'features_dc': self.gaussian_model._features_dc.state_dict(),
                'features_rest': self.gaussian_model._features_rest.state_dict(),
                'opacity': self.gaussian_model._opacity.state_dict(),
                'scaling': self.gaussian_model._scaling.state_dict(),
                'rotation': self.gaussian_model._rotation.state_dict(),
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def save_render(
        self,
        iteration: int,
        camera: Camera,
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
    ):
        """Save rendering results.

        Args:
            iteration: Iteration number
            camera: Camera
            rendered_image: Rendered image
            gt_image: Ground truth image
        """
        # Move images to CPU and convert to numpy
        rendered_np = rendered_image.detach().cpu().permute(1, 2, 0).numpy()
        rendered_np = np.clip(rendered_np, 0, 1)
        rendered_np = (rendered_np * 255).astype(np.uint8)

        gt_np = gt_image.detach().cpu().permute(1, 2, 0).numpy() if gt_image.dim() == 3 else gt_image
        if gt_np.max() <= 1.0:
            gt_np = (gt_np * 255).astype(np.uint8)

        # Save images
        render_path = self.output_dir / f"render_{iteration:06d}.png"
        gt_path = self.output_dir / f"gt_{iteration:06d}.png"

        cv2.imwrite(str(render_path), cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(gt_path), cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))

