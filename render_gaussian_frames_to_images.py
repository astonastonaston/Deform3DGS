import os
import argparse
import numpy as np
import open3d as o3d
import imageio


def load_gaussian_npz(npz_path, opacity_threshold=0.01):
    data = np.load(npz_path)
    mean = data["means"]          # [N, 3]
    covs = data["covariances"]    # [N, 3, 3]
    opacity = data["opacity"].squeeze()  # [N]
    color = data.get("colors", None)     # [N, 3]

    mask = opacity > opacity_threshold
    mean = mean[mask]
    covs = covs[mask]
    opacity = opacity[mask]
    if color is not None:
        color = color[mask]
    else:
        # fallback grayscale
        color = np.clip(opacity / (opacity.max() + 1e-6), 0, 1).reshape(-1, 1).repeat(3, axis=1)

    return mean, covs, color


def create_gaussian_ellipsoids(means, covs, colors, sphere_resolution=10):
    """
    Returns a list of Open3D mesh ellipsoids at the given means with shape defined by covariances.
    """
    ellipsoids = []
    for i in range(means.shape[0]):
        eigvals, eigvecs = np.linalg.eigh(covs[i])
        scales = 2.0 * np.sqrt(np.maximum(eigvals, 1e-8))  # 95% confidence ellipse scale

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=sphere_resolution)
        sphere.paint_uniform_color(colors[i])
        sphere.scale(1.0, center=np.zeros(3))

        # Apply transformation from unit sphere to Gaussian ellipsoid
        transform = np.eye(4)
        transform[:3, :3] = eigvecs @ np.diag(scales)
        transform[:3, 3] = means[i]
        sphere.transform(transform)
        ellipsoids.append(sphere)
    return ellipsoids


def render_scene_to_image(geometries, output_path, width=640, height=480, zoom=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geometries:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    vis.destroy_window()


def visualize_gaussians(npz_dir, output_dir, mode="frame", frame_id=0, opacity_threshold=0.01, make_video=False, video_path="gaussian_video.mp4", fps=24):
    files = sorted([f for f in os.listdir(npz_dir) if f.endswith(".npz")])
    os.makedirs(output_dir, exist_ok=True)

    images = []

    if mode == "frame":
        f = files[frame_id]
        means, covs, colors = load_gaussian_npz(os.path.join(npz_dir, f), opacity_threshold)
        ellipsoids = create_gaussian_ellipsoids(means, covs, colors)
        render_scene_to_image(ellipsoids, os.path.join(output_dir, f"frame_{frame_id:04d}.png"))

    elif mode == "video":
        for idx, f in enumerate(files):
            print(f"Rendering frame {idx}")
            means, covs, colors = load_gaussian_npz(os.path.join(npz_dir, f), opacity_threshold)
            ellipsoids = create_gaussian_ellipsoids(means, covs, colors)
            img_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
            render_scene_to_image(ellipsoids, img_path)
            if make_video:
                images.append(imageio.imread(img_path))

        if make_video:
            imageio.mimwrite(video_path, images, fps=fps)
            print(f"Saved video to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Gaussian ellipsoids from exported frames.")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory with .npz files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for rendered frames.")
    parser.add_argument("--mode", choices=["frame", "video"], default="frame", help="Render a single frame or full video.")
    parser.add_argument("--frame_id", type=int, default=0, help="Frame ID for 'frame' mode.")
    parser.add_argument("--opacity_threshold", type=float, default=0.01, help="Opacity filtering threshold.")
    parser.add_argument("--make_video", action="store_true", help="Whether to compile rendered images into a video.")
    parser.add_argument("--video_path", type=str, default="gaussian_video.mp4", help="Output path for video.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for the output video.")
    args = parser.parse_args()

    visualize_gaussians(
        npz_dir=args.npz_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        frame_id=args.frame_id,
        opacity_threshold=args.opacity_threshold,
        make_video=args.make_video,
        video_path=args.video_path,
        fps=args.fps,
    )
