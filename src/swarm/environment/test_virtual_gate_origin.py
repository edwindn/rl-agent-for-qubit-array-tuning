"""
Test script to instantiate QArray base class and generate a scan
with all voltages set to their ground truth values.
"""
import numpy as np
import matplotlib.pyplot as plt
from qarray_base_class import QarrayBaseClass


def main():
    # Configuration
    num_dots = 4
    use_barriers = True

    # Instantiate QArray base class
    print("Instantiating QArray base class...")
    qarray = QarrayBaseClass(
        num_dots=num_dots,
        use_barriers=use_barriers,
        obs_voltage_min=-10.0,
        obs_voltage_max=10.0,
        obs_image_size=100,
        vary_peak_width=False,
    )

    # Calculate ground truth values
    print("Calculating ground truth values...")
    # Initialize with zeros to get ground truth in virtual space
    initial_gate_voltages = np.zeros(num_dots)
    initial_barrier_voltages = np.zeros(num_dots - 1) if use_barriers else None

    vg_gt, vb_gt, sensor_gt = qarray.calculate_ground_truth(
        initial_gate_voltages,
        initial_barrier_voltages
    )

    print(f"Ground truth gate voltages: {vg_gt}")
    print(f"Ground truth barrier voltages: {vb_gt}")
    print(f"Ground truth sensor voltage: {sensor_gt}")

    # Generate first scan with ground truth values
    print("Generating first scan with ground truth voltages...")
    obs1 = qarray._get_obs(vg_gt, vb_gt, sensor_gt)
    image1 = obs1["image"][:, :, 0]

    # Update virtual gate origin to +10 volts
    print("Updating virtual gate origin to +10V...")
    v0_offset = np.ones(num_dots + 1) * 10.0
    qarray._update_virtual_gate_origin(v0_offset)

    # Extract virtual gate matrix and compute adjustment
    # The transformation is: Vphys = VGM @ Vvirtual + v0
    # To keep Vphys constant when v0 changes:
    # VGM @ Vvirtual_new + v0_new = VGM @ Vvirtual_old + v0_old
    # Vvirtual_new = Vvirtual_old + inv(VGM) @ (v0_old - v0_new)
    # Since v0_old = 0 and v0_new = offset:
    # Vvirtual_new = Vvirtual_old - inv(VGM) @ offset

    vgm = qarray.model.gate_voltage_composer.virtual_gate_matrix
    vgm_inv = np.linalg.inv(vgm)
    adjustment = vgm_inv @ v0_offset

    # Adjust plunger voltages: subtract inv(VGM) @ offset from virtual voltages
    vg_adjusted = vg_gt - adjustment[:-1]  # Exclude sensor voltage
    sensor_adjusted = sensor_gt - adjustment[-1]

    print(f"inv(VGM) @ offset = {adjustment}")
    print(f"Adjusted gate voltages: {vg_adjusted}")
    print(f"Adjusted sensor voltage: {sensor_adjusted}")

    # Generate second scan with adjusted voltages and offset origin
    print("Generating second scan with adjusted voltages and offset origin...")
    obs2 = qarray._get_obs(vg_adjusted, vb_gt, sensor_adjusted)
    image2 = obs2["image"][:, :, 0]

    # Save both scans side by side
    print("Saving scans to testing_offset.png...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # First scan
    im1 = axes[0].imshow(
        image1,
        extent=[
            qarray.obs_voltage_min,
            qarray.obs_voltage_max,
            qarray.obs_voltage_min,
            qarray.obs_voltage_max,
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("$\\Delta$PL (V)")
    axes[0].set_ylabel("$\\Delta$PR (V)")
    axes[0].set_title("Original (v0 = 0V)")
    axes[0].axis("equal")
    plt.colorbar(im1, ax=axes[0], label="Charge Sensor Signal")

    # Second scan
    im2 = axes[1].imshow(
        image2,
        extent=[
            qarray.obs_voltage_min,
            qarray.obs_voltage_max,
            qarray.obs_voltage_min,
            qarray.obs_voltage_max,
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("$\\Delta$PL (V)")
    axes[1].set_ylabel("$\\Delta$PR (V)")
    axes[1].set_title("After v0 = +10V offset")
    axes[1].axis("equal")
    plt.colorbar(im2, ax=axes[1], label="Charge Sensor Signal")

    plt.tight_layout()
    plt.savefig("testing_offset.png", dpi=150)
    plt.close()

    print("Done! Image saved to testing_offset.png")


if __name__ == "__main__":
    main()
