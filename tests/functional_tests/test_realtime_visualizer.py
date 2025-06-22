"""
Test script for the real-time SpikingQRS visualizer using PyQtGraph.

This script demonstrates how to use the new plot_chunk_realtime method
for real-time visualization of SpikingQRS beat detection.
"""
import sys
import os
import numpy as np
import time
import threading

# Add project root to path to allow importing project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from plotting.spiking_visualizer import SpikingVisualizer
from utils.snippets import generate_challenging_ecg, generate_spike_input

def test_realtime_visualization():
    """Test the real-time visualization with synthetic data."""
    print("Testing SpikingQRS Real-time Visualizer...")
    
    # Initialize visualizer
    sampling_rate = 500
    visualizer = SpikingVisualizer(sampling_rate)
    
    # Check if PyQtGraph is available
    if not hasattr(visualizer, 'realtime_win') or visualizer.realtime_win is None:
        print("PyQtGraph not available. Please install with: pip install pyqtgraph")
        return
    
    # Start real-time visualization
    visualizer.start_realtime_visualization()
    
    # Generate synthetic data
    print("Generating challenging synthetic ECG data...")
    ecg_signal = generate_challenging_ecg(duration=30, sampling_rate=sampling_rate)
    
    # Simulate processing in chunks
    chunk_size = 1000
    overlap = 100
    
    print("Starting real-time processing simulation...")
    
    def processing_simulation():
        for start_idx in range(0, len(ecg_signal) - chunk_size, chunk_size - overlap):
            end_idx = start_idx + chunk_size
            
            # Extract chunk
            chunk_ecg = ecg_signal[start_idx:end_idx]
            
            # Simulate filtered signal (simple moving average)
            chunk_filtered = np.convolve(chunk_ecg, np.ones(10)/10, mode='same')
            chunk_filtered = chunk_filtered ** 2  # Square for detection
            
            # Generate spike input from filtered signal
            spk_tensor, _ = generate_spike_input(chunk_filtered, num_steps=len(chunk_filtered))
            chunk_spikes = spk_tensor.numpy().T
            
            # Simulate detected peaks (simple threshold-based detection)
            threshold = np.mean(chunk_filtered) + 2 * np.std(chunk_filtered)
            detected_peaks = np.where(chunk_filtered > threshold)[0]
            
            # Simulate spiking periods (bursts)
            spiking_periods = []
            accepted_peaks_with_periods = []
            if len(detected_peaks) > 0:
                # Create some spiking periods around detected peaks
                for peak in detected_peaks[:3]:  # Limit to first 3 peaks
                    start_period = max(0, peak - 50)
                    end_period = min(len(chunk_ecg), peak + 50)
                    spiking_periods.append((start_period, end_period))
                    
                    # Create accepted_peaks_with_periods format, which is required by the visualizer
                    accepted_peaks_with_periods.append({
                        "detection": peak,
                        "spiking_period": (start_period, end_period)
                    })
            
            # Update real-time visualization
            visualizer.plot_chunk_realtime(
                start_index=start_idx,
                end_index=end_idx,
                spikes_layer1=chunk_spikes,
                spikes_layer2=chunk_spikes,  # Use same for simplicity
                accepted_peaks_with_periods=accepted_peaks_with_periods,
                spiking_periods=spiking_periods,
                ecg_signal=ecg_signal,
                chunk_squared_signal=chunk_filtered
            )
            
            # Simulate processing time
            time.sleep(0.1)  # 100ms delay between chunks
    
    # Run processing in a separate thread
    processing_thread = threading.Thread(target=processing_simulation, daemon=True)
    processing_thread.start()
    
    print("Real-time visualization is running...")
    print("Close the visualization window to stop.")
    
    # Start the Qt event loop. This will block until the window is closed.
    if visualizer.realtime_app:
        print("Starting PyQtGraph event loop...")
        visualizer.realtime_app.exec_()
        print("\nVisualization window closed. Test finished.")
    else:
        print("Could not start event loop: QApplication instance not found.")

if __name__ == "__main__":
    test_realtime_visualization() 