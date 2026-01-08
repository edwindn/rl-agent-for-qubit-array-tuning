"""Utility functions for inference - saving GIFs and distance plots locally."""

import glob
from pathlib import Path


def save_gifs_locally(iteration_num, config, disable_cleanup=False):
    """Process saved images into GIFs and save locally."""
    import shutil
    from PIL import Image

    gif_save_dir = Path(config['gif_config']['save_dir'])
    inference_dir = Path(__file__).parent

    if not gif_save_dir.exists():
        print("No image dir found for gif creation")
        return

    try:
        print(f"Processing GIFs for iteration {iteration_num}...")

        # Find all agent subdirectories
        agent_dirs = [d for d in gif_save_dir.iterdir() if d.is_dir()]

        if not agent_dirs:
            print("No agent directories found for GIF creation")
            return

        gifs_saved = 0
        # Process each agent's images separately
        for agent_dir in agent_dirs:
            agent_id = agent_dir.name
            image_files = sorted(agent_dir.glob("step_*.png"))

            if len(image_files) < 2:
                print(f"Not enough images for {agent_id} (need at least 2)")
                continue

            # Load images
            images = []
            for img_file in image_files:
                img = Image.open(img_file)
                images.append(img)

            # Save as GIF in inference directory
            gif_output_path = inference_dir / f"{agent_id}_iteration_{iteration_num}.gif"
            images[0].save(
                gif_output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / config['gif_config'].get('fps', 0.5)),
                loop=0
            )
            print(f"Saved GIF: {gif_output_path}")
            gifs_saved += 1

        # Clean up temporary files unless disabled
        if not disable_cleanup:
            shutil.rmtree(gif_save_dir, ignore_errors=True)
        print(f"Processed and saved {gifs_saved} agent GIFs")

    except Exception as e:
        print(f"Error processing GIFs: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error unless disabled
        if not disable_cleanup:
            import shutil
            shutil.rmtree(gif_save_dir, ignore_errors=True)


def save_distance_plots(distance_data_dir, iteration=1):
    """Create and save distance plots locally."""
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.use('Agg')

    inference_dir = Path(__file__).parent

    try:
        distance_data_path = Path(distance_data_dir)

        # Get all agent folders
        plunger_folders = sorted([f for f in distance_data_path.iterdir() if f.is_dir() and f.name.startswith("plunger_")])
        barrier_folders = sorted([f for f in distance_data_path.iterdir() if f.is_dir() and f.name.startswith("barrier_")])

        # Plot plunger distances
        if plunger_folders:
            fig, ax = plt.subplots(figsize=(10, 6))

            for agent_folder in plunger_folders:
                # Get all .npy files and find the one with highest count
                npy_files = glob.glob(str(agent_folder / "*.npy"))
                if npy_files:
                    # Extract counts and find max
                    max_count = 0
                    latest_file = None
                    for filepath in npy_files:
                        filename = Path(filepath).stem
                        count_str = filename.split('_')[0]
                        count = int(count_str)
                        if count > max_count:
                            max_count = count
                            latest_file = filepath

                    if latest_file is not None:
                        # Load and plot the data
                        distances = np.load(latest_file)
                        steps = np.arange(1, len(distances) + 1)
                        ax.plot(steps, distances, label=agent_folder.name, alpha=0.7)

            ax.set_xlabel("Episode Step")
            ax.set_ylabel("Distance from Ground Truth")
            ax.set_title(f"Plunger Agent Distances (Iteration {iteration})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plunger_plot_path = inference_dir / f"plunger_distances_iteration_{iteration}.png"
            fig.savefig(plunger_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved plunger distances plot: {plunger_plot_path}")

        # Plot barrier distances
        if barrier_folders:
            fig, ax = plt.subplots(figsize=(10, 6))

            for agent_folder in barrier_folders:
                # Get all .npy files and find the one with highest count
                npy_files = glob.glob(str(agent_folder / "*.npy"))
                if npy_files:
                    # Extract counts and find max
                    max_count = 0
                    latest_file = None
                    for filepath in npy_files:
                        filename = Path(filepath).stem
                        count_str = filename.split('_')[0]
                        count = int(count_str)
                        if count > max_count:
                            max_count = count
                            latest_file = filepath

                    if latest_file is not None:
                        # Load and plot the data
                        distances = np.load(latest_file)
                        steps = np.arange(1, len(distances) + 1)
                        ax.plot(steps, distances, label=agent_folder.name, alpha=0.7)

            ax.set_xlabel("Episode Step")
            ax.set_ylabel("Distance from Ground Truth")
            ax.set_title(f"Barrier Agent Distances (Iteration {iteration})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            barrier_plot_path = inference_dir / f"barrier_distances_iteration_{iteration}.png"
            fig.savefig(barrier_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved barrier distances plot: {barrier_plot_path}")

    except Exception as e:
        print(f"Error plotting distance data: {e}")
        import traceback
        traceback.print_exc()