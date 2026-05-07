"""
Streamlit GUI for exploring qarray quantum device simulation parameters.
Clean interface for adjusting parameters and visualizing charge stability diagrams.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from image_generator import QarrayImageGenerator


def load_config_ranges():
    """Load parameter ranges from qarray_config.yaml"""
    config_path = os.path.join(
        os.path.dirname(__file__),  # gui directory
        "..",  # environment directory
        "qarray_config.yaml"
    )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config['simulator']['model']


def create_extended_slider(label, config_key, config_ranges, default_value, step=0.01, extend_factor=4.0, **kwargs):
    """
    Create a slider with extended range and visual indicators for recommended range.

    Args:
        label: Slider label
        config_key: Key in config_ranges dict
        config_ranges: Dict with parameter ranges from config
        default_value: Default slider value
        step: Step size for slider
        extend_factor: How much to extend beyond config range (multiplier)
        **kwargs: Additional arguments for st.slider
    """
    if config_key in config_ranges:
        config_range = config_ranges[config_key]
        config_min = config_range['min']
        config_max = config_range['max']

        # Calculate extended range
        range_width = config_max - config_min
        extension = range_width * (extend_factor - 1) / 2

        extended_min = config_min - extension
        extended_max = config_max + extension

        # Create help text showing config range
        help_text = f"Config range: {config_min} - {config_max}"

        # Create the slider with extended range
        value = st.slider(
            label,
            min_value=extended_min,
            max_value=extended_max,
            value=default_value,
            step=step,
            help=help_text,
            **kwargs
        )

        # Add visual indicator if current value is outside config range
        if value < config_min or value > config_max:
            st.caption(f"⚠️ Outside config range ({config_min} - {config_max})")
        else:
            st.caption(f"✅ Within config range")

        return value
    else:
        # Fallback to regular slider if config not found
        return st.slider(label, value=default_value, step=step, **kwargs)


# Set page config
st.set_page_config(
    page_title="Qarray Parameter Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚛️ Quantum Device Parameter Explorer")
st.markdown("Explore how different parameters affect quantum dot charge stability diagrams")

# Regenerate button at the top
regenerate_clicked = st.button("🔄 Regenerate Image", type="primary", help="Click to update the charge stability diagram with current parameters")

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'current_voltages' not in st.session_state:
    st.session_state.current_voltages = None
if 'device_config' not in st.session_state:
    st.session_state.device_config = {'num_dots': 2, 'use_barriers': False}
if 'cached_image' not in st.session_state:
    st.session_state.cached_image = None
if 'cached_image_info' not in st.session_state:
    st.session_state.cached_image_info = None
if 'model_settings' not in st.session_state:
    st.session_state.model_settings = {'use_variable_peak_width': False}

# Load config ranges for parameter indicators
try:
    config_ranges = load_config_ranges()
except Exception as e:
    st.error(f"Could not load config ranges: {e}")
    config_ranges = {}

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Device Configuration")

    # Basic device settings
    num_dots = st.selectbox("Number of Dots", [2, 3, 4], index=0)
    use_barriers = st.checkbox("Use Barriers", value=False)

    # Image settings
    st.subheader("Image Settings")
    voltage_window = st.slider("Voltage Window (V)", 0.5, 3.0, 1.5, 0.1)
    image_size = st.selectbox("Image Resolution", [64, 128, 256], index=1)

    # Check if device configuration changed
    config_changed = (
        st.session_state.device_config['num_dots'] != num_dots or
        st.session_state.device_config['use_barriers'] != use_barriers
    )

    # Initialize generator button
    if st.button("🎲 Initialize/Resample Device", type="primary") or config_changed:
        st.session_state.generator = QarrayImageGenerator(
            num_dots=num_dots,
            use_barriers=use_barriers,
            obs_voltage_min=-voltage_window,
            obs_voltage_max=voltage_window,
            obs_image_size=image_size
        )

        # Update device config
        st.session_state.device_config = {'num_dots': num_dots, 'use_barriers': use_barriers}

        # Set initial voltages from ground truth
        ground_truth = st.session_state.generator.get_ground_truth_voltages()
        st.session_state.current_voltages = {
            'plungers': ground_truth['plunger_voltages'].copy(),
            'barriers': ground_truth['barrier_voltages'].copy() if use_barriers else np.zeros(num_dots-1)
        }

        # Clear cached image when device is reinitialized
        st.session_state.cached_image = None
        st.session_state.cached_image_info = None

        if config_changed:
            st.success("Device configuration updated!")
            st.rerun()  # Force rerun to update the voltage sliders
        else:
            st.success("Device initialized with random parameters!")

# Main content area
if st.session_state.generator is None:
    st.info("👈 Please initialize a device using the sidebar controls")
    st.stop()

# Create two columns: controls and visualization
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎛️ Parameter Controls")

    # Parameter override controls
    st.subheader("Model Parameters")
    param_overrides = {}

    # Get current device configuration for consistent usage
    current_num_dots = st.session_state.device_config['num_dots']
    current_use_barriers = st.session_state.device_config['use_barriers']

    # Core Physical Parameters
    with st.expander("Core Physical Parameters", expanded=True):
        temperature = create_extended_slider(
            "Temperature (mK)",
            "T",
            config_ranges,
            125.0,
            step=5.0,
            key="temp_slider"
        )
        param_overrides['T'] = temperature

        peak_width = create_extended_slider(
            "Coulomb Peak Width",
            "coulomb_peak_width",
            config_ranges,
            0.35,
            step=0.01,
            key="peak_width_slider"
        )
        param_overrides['coulomb_peak_width'] = peak_width

        noise_amp = create_extended_slider(
            "White Noise Amplitude",
            "white_noise_amplitude",
            config_ranges,
            0.001,
            step=0.0001,
            key="noise_slider"
        )
        param_overrides['white_noise_amplitude'] = noise_amp

        if current_use_barriers:
            tunnel_coupling = create_extended_slider(
                "Tunnel Coupling",
                "tc",
                config_ranges,
                0.15,
                step=0.01,
                key="tc_slider"
            )
            param_overrides['tc'] = tunnel_coupling

    # Noise Parameters
    with st.expander("Telegraph Noise Parameters"):
        p01 = create_extended_slider(
            "Telegraph P01",
            "telegraph_noise_parameters.p01",
            config_ranges.get('telegraph_noise_parameters', {}),
            0.005,
            step=0.001,
            key="p01_slider"
        )
        param_overrides['telegraph_noise_parameters.p01'] = p01

        p10_factor = create_extended_slider(
            "Telegraph P10 Factor",
            "p10_factor",
            config_ranges.get('telegraph_noise_parameters', {}),
            50.0,
            step=1.0,
            key="p10_slider"
        )
        param_overrides['telegraph_noise_parameters.p10_factor'] = p10_factor

        telegraph_amp = create_extended_slider(
            "Telegraph Amplitude",
            "amplitude",
            config_ranges.get('telegraph_noise_parameters', {}),
            0.006,
            step=0.001,
            key="tele_amp_slider"
        )
        param_overrides['telegraph_noise_parameters.amplitude'] = telegraph_amp

    # Capacitance Matrices - Interactive Matrix Editors
    if st.session_state.generator:
        # Get current matrices to display/edit
        current_model = st.session_state.generator.array

        # Dot-to-Dot Capacitance Matrix (Cdd)
        with st.expander("Dot-to-Dot Capacitance Matrix (Cdd)", expanded=False):
            st.write("**Current Cdd Matrix:**")
            cdd_matrix = current_model.model.Cdd.copy()

            # Create editable matrix
            for i in range(current_num_dots):
                cols = st.columns(current_num_dots)
                for j in range(current_num_dots):
                    with cols[j]:
                        if i == j:
                            st.text(f"Dot{i+1}→Dot{j+1}: 0.0")  # Diagonal is always 0
                        else:
                            new_val = st.number_input(
                                f"Dot{i+1}→Dot{j+1}",
                                value=float(cdd_matrix[i,j]),
                                step=0.001,
                                format="%.3f",
                                key=f"cdd_{i}_{j}_{current_num_dots}"
                            )
                            cdd_matrix[i,j] = new_val
                            cdd_matrix[j,i] = new_val  # Keep symmetric

            param_overrides['Cdd'] = cdd_matrix

        # Gate-to-Dot Capacitance Matrix (Cgd)
        with st.expander("Gate-to-Dot Capacitance Matrix (Cgd)", expanded=False):
            st.write("**Current Cgd Matrix:**")
            cgd_matrix = current_model.model.Cgd.copy()

            num_gates = current_num_dots + 1  # plunger gates + sensor gate

            # Create editable matrix
            for i in range(current_num_dots):  # dots
                cols = st.columns(num_gates)
                for j in range(num_gates):  # gates
                    with cols[j]:
                        gate_name = f"P{j+1}" if j < current_num_dots else "Sensor"
                        new_val = st.number_input(
                            f"Dot{i+1}→{gate_name}",
                            value=float(cgd_matrix[i,j]),
                            step=0.001,
                            format="%.3f",
                            key=f"cgd_{i}_{j}_{current_num_dots}"
                        )
                        cgd_matrix[i,j] = new_val

            param_overrides['Cgd'] = cgd_matrix

        # Sensor Capacitances
        with st.expander("Sensor Capacitances", expanded=False):
            # Cds
            st.write("**Dot-to-Sensor (Cds):**")
            cds_matrix = current_model.model.Cds.copy()
            cols = st.columns(current_num_dots)
            for i in range(current_num_dots):
                with cols[i]:
                    new_val = st.number_input(
                        f"Dot{i+1}→Sensor",
                        value=float(cds_matrix[0,i]),
                        step=0.001,
                        format="%.3f",
                        key=f"cds_{i}_{current_num_dots}"
                    )
                    cds_matrix[0,i] = new_val
            param_overrides['Cds'] = cds_matrix

            # Cgs
            st.write("**Gate-to-Sensor (Cgs):**")
            cgs_matrix = current_model.model.Cgs.copy()
            cols = st.columns(num_gates)
            for j in range(num_gates):
                with cols[j]:
                    gate_name = f"P{j+1}" if j < current_num_dots else "Sensor"
                    new_val = st.number_input(
                        f"{gate_name}→Sensor",
                        value=float(cgs_matrix[0,j]),
                        step=0.001,
                        format="%.3f",
                        key=f"cgs_{j}_{current_num_dots}"
                    )
                    cgs_matrix[0,j] = new_val
            param_overrides['Cgs'] = cgs_matrix

        # Barrier matrices (if enabled)
        if current_use_barriers:
            num_barriers = current_num_dots - 1

            # Barrier Model Parameters
            with st.expander("Barrier Model Parameters", expanded=False):
                barrier_tc_base = create_extended_slider(
                    "Barrier TC Base",
                    "tc_base",
                    config_ranges.get('barrier_model', {}),
                    0.15,
                    step=0.01,
                    key="barrier_tc_slider"
                )
                param_overrides['barrier_model_parameters.tc_base'] = barrier_tc_base

                barrier_alpha = create_extended_slider(
                    "Barrier Alpha",
                    "alpha_per_barrier",  # This is the config key name
                    config_ranges.get('barrier_model', {}),
                    1.4,
                    step=0.1,
                    key="barrier_alpha_slider"
                )
                param_overrides['barrier_model_parameters.alpha'] = barrier_alpha

    # Voltage Dependent Parameters
    with st.expander("Voltage Dependent Parameters", expanded=False):
        st.subheader("Voltage-Dependent Capacitance")

        # Voltage Capacitance Model Type
        vc_model_type = st.selectbox(
            "Capacitance Model Type",
            options=["null", "linear"],
            index=1 if config_ranges.get('voltage_capacitance_model', {}).get('type', 'linear') == 'linear' else 0,
            key="vc_model_type",
            help="Enable/disable voltage-dependent capacitance effects"
        )
        param_overrides['voltage_capacitance_model.type'] = vc_model_type

        if vc_model_type != "null":
            # Voltage Capacitance Parameters (only show if model is enabled)
            vc_alpha = create_extended_slider(
                "Voltage Cap Alpha (Cdd dependence)",
                "alpha",
                config_ranges.get('voltage_capacitance_model', {}),
                0.1,
                step=0.01,
                key="vc_alpha_slider"
            )
            param_overrides['voltage_capacitance_parameters.alpha'] = vc_alpha

            vc_beta = create_extended_slider(
                "Voltage Cap Beta (Cgd dependence)",
                "beta",
                config_ranges.get('voltage_capacitance_model', {}),
                0.1,
                step=0.01,
                key="vc_beta_slider"
            )
            param_overrides['voltage_capacitance_parameters.beta'] = vc_beta

        st.subheader("Variable Peak Width Model")

        # Enable Variable Peak Width
        use_variable_peak_width = st.checkbox(
            "Enable Variable Peak Width",
            value=False,
            key="use_vpw_checkbox",
            help="Enable voltage-dependent Coulomb peak width"
        )

        if use_variable_peak_width:
            # Variable Peak Width Alpha
            vpw_alpha = create_extended_slider(
                "Peak Width Alpha",
                "alpha",
                config_ranges.get('variable_peak_width_model', {}),
                0.01,
                step=0.001,
                key="vpw_alpha_slider"
            )
            param_overrides['variable_peak_width_model.alpha'] = vpw_alpha

    # Check if model settings changed and reinitialize if needed
    model_settings_changed = (
        st.session_state.model_settings['use_variable_peak_width'] != use_variable_peak_width
    )

    if model_settings_changed and st.session_state.generator:
        # Update model settings and reinitialize generator
        st.session_state.model_settings['use_variable_peak_width'] = use_variable_peak_width

        # Get current device config
        current_device_config = st.session_state.device_config

        st.session_state.generator = QarrayImageGenerator(
            num_dots=current_device_config['num_dots'],
            use_barriers=current_device_config['use_barriers'],
            obs_voltage_min=st.session_state.generator.obs_voltage_min,
            obs_voltage_max=st.session_state.generator.obs_voltage_max,
            obs_image_size=st.session_state.generator.obs_image_size,
            use_variable_peak_width=use_variable_peak_width
        )

        # Clear cached image when model settings change
        st.session_state.cached_image = None
        st.session_state.cached_image_info = None

        # Update ground truth voltages
        ground_truth = st.session_state.generator.get_ground_truth_voltages()
        st.session_state.current_voltages = {
            'plungers': ground_truth['plunger_voltages'].copy(),
            'barriers': ground_truth['barrier_voltages'].copy() if current_device_config['use_barriers'] else np.zeros(current_device_config['num_dots']-1)
        }

        st.success("Model settings updated! Generator reinitialized.")

    # Apply parameter updates immediately - but preserve user voltage settings
    if param_overrides and st.session_state.generator:
        st.session_state.generator.reset_array(param_overrides)
        # DON'T overwrite user voltage settings when just updating parameters
        # The ground truth voltages are available via the "Go to Ground Truth" button

    # Voltage controls
    st.subheader("Voltage Controls")


    # Plunger voltages
    st.write("**Plunger Gate Voltages**")
    plunger_voltages = []
    for i in range(current_num_dots):
        default_voltage = -1.0
        if (st.session_state.current_voltages and
            st.session_state.current_voltages['plungers'] is not None and
            i < len(st.session_state.current_voltages['plungers'])):
            default_voltage = float(st.session_state.current_voltages['plungers'][i])

        voltage = st.slider(
            f"Plunger {i+1} (V)",
            -3.0, 1.0,
            default_voltage,
            0.1,
            key=f"plunger_{i}_{current_num_dots}"  # Include num_dots in key to force refresh
        )
        plunger_voltages.append(voltage)

    # Update session state with current user slider values
    if st.session_state.current_voltages is None:
        st.session_state.current_voltages = {}
    st.session_state.current_voltages['plungers'] = plunger_voltages.copy()

    # Barrier voltages (if enabled)
    barrier_voltages = []
    if current_use_barriers:
        st.write("**Barrier Voltages**")
        for i in range(current_num_dots - 1):
            default_voltage = 5.0
            if (st.session_state.current_voltages and
                st.session_state.current_voltages['barriers'] is not None and
                i < len(st.session_state.current_voltages['barriers'])):
                default_voltage = float(st.session_state.current_voltages['barriers'][i])

            voltage = st.slider(
                f"Barrier {i+1} (V)",
                0.0, 15.0,
                default_voltage,
                0.5,
                key=f"barrier_{i}_{current_num_dots}_{current_use_barriers}"  # Include config in key
            )
            barrier_voltages.append(voltage)

    # Update session state with barrier voltages
    st.session_state.current_voltages['barriers'] = barrier_voltages.copy() if barrier_voltages else np.zeros(current_num_dots-1)

    # Quick voltage buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎯 Go to Ground Truth"):
            ground_truth = st.session_state.generator.get_ground_truth_voltages()
            st.session_state.current_voltages = {
                'plungers': ground_truth['plunger_voltages'].copy(),
                'barriers': ground_truth['barrier_voltages'].copy() if use_barriers else np.zeros(num_dots-1)
            }
            st.rerun()

    with col_b:
        if st.button("🎲 Random Voltages"):
            random_voltages = st.session_state.generator.sample_random_voltages()
            st.session_state.current_voltages = {
                'plungers': random_voltages['plunger_voltages'].copy(),
                'barriers': random_voltages['barrier_voltages'].copy()
            }
            st.rerun()

with col2:
    st.header("📊 Charge Stability Diagram")

    # Gate pair selection
    available_pairs = st.session_state.generator.get_available_gate_pairs()
    pair_options = [f"{pair[2]}" for pair in available_pairs]
    selected_pair_idx = st.selectbox("Select Gate Pair", range(len(pair_options)), format_func=lambda x: pair_options[x])

    # Generate image only when button is clicked or if no cached image exists
    plunger_array = np.array(plunger_voltages)
    barrier_array = np.array(barrier_voltages) if barrier_voltages else None

    # Check if we need to generate a new image
    should_generate = (regenerate_clicked or
                      st.session_state.cached_image is None or
                      st.session_state.cached_image_info is None)

    if should_generate:
        try:
            with st.spinner("Generating charge stability diagram..."):
                # Generate the image
                image = st.session_state.generator.generate_image(
                    gate_voltages=plunger_array,
                    barrier_voltages=barrier_array,
                    gate_pair_index=selected_pair_idx
                )

                # Cache the image and its generation info
                st.session_state.cached_image = image
                st.session_state.cached_image_info = {
                    'plunger_voltages': plunger_array.copy(),
                    'barrier_voltages': barrier_array.copy() if barrier_array is not None else None,
                    'gate_pair_idx': selected_pair_idx,
                    'voltage_window': voltage_window
                }
        except Exception as e:
            st.error(f"Error generating image: {e}")
            st.exception(e)
            st.session_state.cached_image = None
            st.session_state.cached_image_info = None

    # Display the cached image (if available)
    if st.session_state.cached_image is not None and st.session_state.cached_image_info is not None:
        try:
            image = st.session_state.cached_image
            cached_info = st.session_state.cached_image_info

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))

            im = ax.imshow(
                image,
                extent=[-cached_info['voltage_window'], cached_info['voltage_window'],
                       -cached_info['voltage_window'], cached_info['voltage_window']],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )

            gate_pair = available_pairs[cached_info['gate_pair_idx']]
            ax.set_xlabel(f'Gate {gate_pair[0]+1} Voltage (V)')
            ax.set_ylabel(f'Gate {gate_pair[1]+1} Voltage (V)')
            ax.set_title(f'Charge Stability Diagram: {gate_pair[2]}')

            plt.colorbar(im, ax=ax, label='Signal')
            plt.tight_layout()

            # Display in streamlit
            st.pyplot(fig)
            plt.close(fig)


        except Exception as e:
            st.error(f"Error displaying image: {e}")
            st.exception(e)
    else:
        st.info("Click the 'Regenerate Image' button above to generate a charge stability diagram with your current parameters.")

# Footer with model info
with st.expander("Model Information"):
    if st.session_state.generator:
        model_info = st.session_state.generator.get_model_info()
        st.json(model_info)