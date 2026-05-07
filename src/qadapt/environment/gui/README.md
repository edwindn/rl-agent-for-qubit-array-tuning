# Qarray Parameter Explorer GUI

Interactive Streamlit web GUI for exploring quantum device simulation parameters.

## Features

- **Device Configuration**: Configure number of dots, barrier usage, voltage windows
- **Parameter Controls**: Adjust core simulation parameters with sliders
- **Voltage Controls**: Set individual plunger and barrier voltages
- **Real-time Visualization**: Generate charge stability diagrams instantly
- **Gate Pair Selection**: Choose which adjacent gate pair to visualize
- **Random Sampling**: Resample device parameters or voltages randomly
- **Ground Truth Display**: Compare current settings with calculated ground truth

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the GUI:**
   ```bash
   python run_gui.py
   ```

3. **Open your browser** to `http://localhost:8501`

## GUI Layout

- **Sidebar**: Device configuration and parameter controls
- **Main Panel**: Charge stability diagram visualization and voltage controls
- **Parameter Sections**:
  - Core Parameters: Temperature, peak width, noise
  - Coupling Parameters: Dot-dot and gate-dot coupling strengths
  - Barrier Parameters: Tunnel coupling and barrier settings (when enabled)
  - Voltage Controls: Individual gate and barrier voltage sliders

## Quick Start

1. Click "Initialize/Resample Device" to create a quantum device
2. Adjust parameter sliders and click "Apply Parameters" to see effects
3. Use voltage sliders to explore different operating points
4. Try "Go to Ground Truth" to see optimal voltages
5. Use "Random Voltages" for quick exploration

## Architecture

- `streamlit_app.py`: Main GUI application
- `image_generator.py`: Clean wrapper around QarrayBaseClass
- `run_gui.py`: Launch script
- Leverages existing QarrayBaseClass parameter handling infrastructure