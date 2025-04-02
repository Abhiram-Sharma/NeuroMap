import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation
from skimage import filters, measure, morphology, segmentation
from skimage.morphology import dilation, ball  # Add this line or modify the existing import
import plotly.graph_objects as go
from io import BytesIO
import os

def read_dicom_series(files):
    """Read and stack DICOM files into a 3D numpy array with progress feedback."""
    slices = []
    progress = st.progress(0)
    for i, file in enumerate(files):
        try:
            ds = pydicom.dcmread(file)
            slices.append(ds)
        except Exception as e:
            st.error(f"Error reading DICOM file: {e}")
            return None
        progress.progress((i + 1) / len(files))
    slices.sort(key=lambda x: float(x.get('SliceLocation', x.InstanceNumber)))
    try:
        shape = slices[0].pixel_array.shape
        volume = np.stack([s.pixel_array for s in slices], axis=0)
        return volume.astype(np.float32)
    except Exception as e:
        st.error(f"Error stacking DICOM slices: {e}")
        return None

def brain_extraction(volume):
    """Extract brain tissue using thresholding and morphological operations."""
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min())
    thresh = filters.threshold_otsu(volume_norm)
    brain_mask = volume_norm > thresh
    brain_mask = morphology.binary_closing(brain_mask, morphology.ball(3))
    brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
    labels = measure.label(brain_mask)
    regions = measure.regionprops(labels)
    if regions:
        largest = max(regions, key=lambda r: r.area)
        brain_mask = labels == largest.label
    return brain_mask

def segment_vessels(volume):
    """Segment blood vessels using a vesselness filter with safety margin."""
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min())
    vessels = filters.frangi(volume_norm, sigmas=range(1, 3), black_ridges=False)
    thresh = filters.threshold_otsu(vessels)
    vessel_mask = vessels > thresh * 1.2  # Increased sensitivity
    safety_margin = 2
    vessel_mask_dilated = dilation(vessel_mask, ball(safety_margin))
    return vessel_mask_dilated

def find_entry_points(brain_mask):
    """Find entry points on the upper brain surface."""
    boundary = segmentation.find_boundaries(brain_mask, mode='outer')
    boundary_coords = np.where(boundary)
    upper_half = boundary_coords[0] < brain_mask.shape[0] // 2
    boundary_points = list(zip(boundary_coords[0][upper_half], 
                              boundary_coords[1][upper_half], 
                              boundary_coords[2][upper_half]))
    if len(boundary_points) > 100:
        indices = np.random.choice(len(boundary_points), 100, replace=False)
        entry_points = [boundary_points[i] for i in indices]
    else:
        entry_points = boundary_points
    return entry_points

def check_trajectory(entry, target, obstacles):
    """Check if a straight-line trajectory avoids obstacles."""
    num_points = 100
    t = np.linspace(0, 1, num_points)
    line_x = np.round(entry[0] + t * (target[0] - entry[0])).astype(int)
    line_y = np.round(entry[1] + t * (target[1] - entry[1])).astype(int)
    line_z = np.round(entry[2] + t * (target[2] - entry[2])).astype(int)
    shape = obstacles.shape
    valid = (line_x >= 0) & (line_x < shape[0]) & \
            (line_y >= 0) & (line_y < shape[1]) & \
            (line_z >= 0) & (line_z < shape[2])
    line_x, line_y, line_z = line_x[valid], line_y[valid], line_z[valid]
    for x, y, z in zip(line_x, line_y, line_z):
        if obstacles[x, y, z]:
            return False
    return True

def plan_trajectories(targets, entry_points, obstacles):
    """Plan straight-line trajectories, selecting the shortest valid path."""
    trajectories = []
    for target in targets:
        valid_entries = []
        for entry in entry_points:
            if check_trajectory(entry, target, obstacles):
                distance = np.linalg.norm(np.array(entry) - np.array(target))
                valid_entries.append((entry, distance))
        if valid_entries:
            best_entry = min(valid_entries, key=lambda x: x[1])[0]
            trajectories.append((best_entry, target))
        else:
            st.warning(f"No valid trajectory found for target {target}")
    return trajectories

def visualize_2d(volume, trajectories, targets, entries):
    """Generate 2D slice visualizations with trajectories overlaid."""
    figs = []
    views = ['axial', 'sagittal', 'coronal']
    slice_indices = [volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2]
    for view, idx in zip(views, slice_indices):
        fig, ax = plt.subplots()
        if view == 'axial':
            slice_img = volume[idx, :, :]
            ax.set_xlabel('Y')
            ax.set_ylabel('X')
        elif view == 'sagittal':
            slice_img = volume[:, idx, :]
            ax.set_xlabel('Z')
            ax.set_ylabel('X')
        else:  # coronal
            slice_img = volume[:, :, idx]
            ax.set_xlabel('Y')
            ax.set_ylabel('X')
        ax.imshow(slice_img, cmap='gray')
        for entry, target in trajectories:
            if view == 'axial' and abs(entry[0] - idx) < 5:
                ax.plot([entry[2], target[2]], [entry[1], target[1]], 'r-', lw=2)
            elif view == 'sagittal' and abs(entry[1] - idx) < 5:
                ax.plot([entry[2], target[2]], [entry[0], target[0]], 'r-', lw=2)
            elif view == 'coronal' and abs(entry[2] - idx) < 5:
                ax.plot([entry[1], target[1]], [entry[0], target[0]], 'r-', lw=2)
        for entry in entries:
            if view == 'axial' and abs(entry[0] - idx) < 5:
                ax.plot(entry[2], entry[1], 'go', label='Entry' if 'Entry' not in ax.get_legend_handles_labels()[1] else "")
            elif view == 'sagittal' and abs(entry[1] - idx) < 5:
                ax.plot(entry[2], entry[0], 'go')
            elif view == 'coronal' and abs(entry[2] - idx) < 5:
                ax.plot(entry[1], entry[0], 'go')
        for target in targets:
            if view == 'axial' and abs(target[0] - idx) < 5:
                ax.plot(target[2], target[1], 'bo', label='Target' if 'Target' not in ax.get_legend_handles_labels()[1] else "")
            elif view == 'sagittal' and abs(target[1] - idx) < 5:
                ax.plot(target[2], target[0], 'bo')
            elif view == 'coronal' and abs(target[2] - idx) < 5:
                ax.plot(target[1], target[0], 'bo')
        ax.legend()
        ax.set_title(f"{view.capitalize()} View")
        figs.append(fig)
    return figs

def visualize_3d(trajectories, targets, entries):
    """Generate a 3D plot of trajectories."""
    fig = go.Figure()
    for entry, target in trajectories:
        fig.add_trace(go.Scatter3d(
            x=[entry[2], target[2]], y=[entry[1], target[1]], z=[entry[0], target[0]],
            mode='lines', line=dict(color='red', width=4), name='Trajectory'
        ))
    fig.add_trace(go.Scatter3d(
        x=[e[2] for e in entries], y=[e[1] for e in entries], z=[e[0] for e in entries],
        mode='markers', marker=dict(size=5, color='green'), name='Entry Points'
    ))
    fig.add_trace(go.Scatter3d(
        x=[t[2] for t in targets], y=[t[1] for t in targets], z=[t[0] for t in targets],
        mode='markers', marker=dict(size=5, color='blue'), name='Target Points'
    ))
    fig.update_layout(scene=dict(xaxis_title='Z', yaxis_title='Y', zaxis_title='X'),
                      title='3D Trajectory Visualization')
    return fig

def main():
    """Main function to run the Streamlit app."""
    st.title("Automated SEEG Trajectory Planning")
    uploaded_files = st.file_uploader("Upload DICOM Files (approx. 250 images)",
                                     accept_multiple_files=True, type=['dcm'])
    if uploaded_files and len(uploaded_files) > 0:
        with st.spinner("Reading DICOM files..."):
            volume = read_dicom_series(uploaded_files)
        if volume is None:
            return
        with st.spinner("Extracting brain tissue..."):
            brain_mask = brain_extraction(volume)
        with st.spinner("Segmenting blood vessels..."):
            obstacles = segment_vessels(volume)
        
        # Interactive target definition
        st.subheader("Define Target Points")
        num_targets = st.number_input("Number of targets", min_value=1, max_value=5, value=2)
        targets = []
        for i in range(num_targets):
            st.write(f"Target {i+1}")
            x = st.number_input(f"X coordinate (0 to {volume.shape[0]-1})", 
                               min_value=0, max_value=volume.shape[0]-1, key=f"x{i}")
            y = st.number_input(f"Y coordinate (0 to {volume.shape[1]-1})", 
                               min_value=0, max_value=volume.shape[1]-1, key=f"y{i}")
            z = st.number_input(f"Z coordinate (0 to {volume.shape[2]-1})", 
                               min_value=0, max_value=volume.shape[2]-1, key=f"z{i}")
            targets.append((x, y, z))
        
        with st.spinner("Finding entry points..."):
            entry_points = find_entry_points(brain_mask)
        with st.spinner("Planning trajectories..."):
            trajectories = plan_trajectories(targets, entry_points, obstacles)
        
        if trajectories:
            st.success(f"Planned {len(trajectories)} trajectories successfully!")
            st.subheader("2D Visualizations")
            figs_2d = visualize_2d(volume, trajectories, targets, [t[0] for t in trajectories])
            for fig in figs_2d:
                st.pyplot(fig)
                plt.close(fig)
            st.subheader("3D Visualization")
            fig_3d = visualize_3d(trajectories, targets, [t[0] for t in trajectories])
            st.plotly_chart(fig_3d)
        else:
            st.error("Could not find valid trajectories.")

if __name__ == "__main__":
    main()
