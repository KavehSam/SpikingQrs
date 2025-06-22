import numpy as np
from spiking_qrs.raster_fex import extract_features_for_burst

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    PYTQTGRAPH_AVAILABLE = True
except ImportError:
    PYTQTGRAPH_AVAILABLE = False
    print("PyQtGraph not available. Install with: pip install pyqtgraph")

__all__ = ["SpikingVisualizer", "PYTQTGRAPH_AVAILABLE"]


class SpikingVisualizer:
    """
    Visualization class for spiking neural network activity and ECG beat detection.
    """

    def __init__(self, sampling_rate, font_size=14, num_neurons=None):
        self.sampling_rate = sampling_rate
        self.font_size = font_size
        self.num_neurons = num_neurons

        # PyQtGraph real-time visualization components
        self.realtime_app = None
        self.realtime_win = None
        self.realtime_plots = {}
        self.realtime_curves = {}
        self.realtime_images = {}
        self.realtime_scatters = {}
        self.realtime_timer = None
        self.data_queues = {}
        self.bw_lut = None
        self.realtime_legends = {}

        # Initialize PyQtGraph if available
        if PYTQTGRAPH_AVAILABLE:
            self._init_realtime_components()

    def _init_realtime_components(self):
        """Initialize PyQtGraph components for real-time visualization."""
        # Set a light theme for consistency with Matplotlib
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.realtime_app = pg.mkQApp("SpikingQRS Real-time Monitor")
        self.realtime_win = pg.GraphicsLayoutWidget(
            show=False, title="SpikingQRS Real-time Monitor"
        )
        self.realtime_win.resize(1200, 800)

        # Define an inverted lookup table (black spikes on white background)
        self.bw_lut = np.array([[255, 255, 255, 255], [0, 0, 0, 255]], dtype=np.uint8)

        ## Create plots
        # 1) ECG Signal Plot
        self.realtime_plots["ecg"] = self.realtime_win.addPlot(
            row=0, col=0, title="Raw ECG Signal"
        )
        self.realtime_plots["ecg"].setLabel("left", "Millivolts")
        self.realtime_plots["ecg"].setLabel("bottom", "Time (s)")
        self.realtime_curves["ecg"] = self.realtime_plots["ecg"].plot(
            pen="b", name="ECG"
        )
        self.realtime_scatters["peaks"] = self.realtime_plots["ecg"].plot(
            pen=None, symbol="o", symbolBrush="g", symbolSize=8, name="R-peaks"
        )

        # 2) Filtered & Squared Signal Plot
        self.realtime_win.nextRow()
        self.realtime_plots["filtered"] = self.realtime_win.addPlot(
            row=1, col=0, title="Filtered & Squared ECG Signal"
        )
        self.realtime_plots["filtered"].setLabel("left", "Millivolts")
        self.realtime_plots["filtered"].setLabel("bottom", "Time (s)")
        self.realtime_plots["filtered"].setXLink(self.realtime_plots["ecg"])
        self.realtime_curves["filtered"] = self.realtime_plots["filtered"].plot(
            pen="b", name="Filtered"
        )
        self.realtime_scatters["detected_peaks"] = self.realtime_plots["filtered"].plot(
            pen=None, symbol="o", symbolBrush="g", symbolSize=8, name="Accepted R-Peaks"
        )
        self.realtime_scatters["rejected_peaks"] = self.realtime_plots["filtered"].plot(
            pen=None,
            symbol="o",
            symbolBrush="y",
            symbolSize=6,
            name="Rejected Candidates",
        )

        # Add legend manually for the filtered signal plot
        self.realtime_legends["filtered"] = self.realtime_plots["filtered"].addLegend()
        self.realtime_legends["filtered"].addItem(
            self.realtime_curves["filtered"], "Filtered"
        )
        self.realtime_legends["filtered"].addItem(
            self.realtime_scatters["detected_peaks"], "Accepted R-Peaks"
        )
        self.realtime_legends["filtered"].addItem(
            self.realtime_scatters["rejected_peaks"], "Rejected Candidates"
        )

        # 3) hidden-layer spike raster plot
        self.realtime_win.nextRow()
        self.realtime_plots["spikes"] = self.realtime_win.addPlot(
            row=2, col=0, title="Hiden-layer Spike Raster Plot"
        )
        self.realtime_plots["spikes"].setLabel("left", "Neuron Index")
        self.realtime_plots["spikes"].setLabel("bottom", "Time (s)")
        self.realtime_plots["spikes"].setXLink(self.realtime_plots["ecg"])

        self.realtime_scatters["spikes"] = pg.ScatterPlotItem(
            pen=None, symbol="s", size=2, brush=pg.mkBrush(0, 0, 0, 255)
        )
        self.realtime_plots["spikes"].addItem(self.realtime_scatters["spikes"])

        if self.num_neurons is not None:
            self.realtime_plots["spikes"].setYRange(0, self.num_neurons)

        # Add legend for the ECG plot
        self.realtime_plots["ecg"].addLegend()

        # Initialize data queues
        self.data_queues = {
            "ecg": [],
            "filtered": [],
            "spikes": [],
            "peaks": [],
            "detected_peaks": [],
            "spiking_periods": [],
        }

        # Update timer
        self.realtime_timer = pg.QtCore.QTimer()
        self.realtime_timer.timeout.connect(self._update_realtime_plots)
        self.realtime_timer.start(50)  # 20 FPS

    def _update_realtime_plots(self):
        """Update real-time plots with new data."""
        if not self.data_queues["ecg"]:
            return

        # Get latest data
        data = self.data_queues["ecg"].pop(0)
        start_index = data.get("start_index", 0)

        # Update ECG plot
        if "ecg_signal" in data:
            num_samples = len(data["ecg_signal"])
            time_axis_s = (start_index + np.arange(num_samples)) / self.sampling_rate
            self.realtime_curves["ecg"].setData(time_axis_s, data["ecg_signal"])

        # Update filtered signal plot
        if "filtered_signal" in data:
            num_samples = len(data["filtered_signal"])
            time_axis_s = (start_index + np.arange(num_samples)) / self.sampling_rate
            self.realtime_curves["filtered"].setData(
                time_axis_s, data["filtered_signal"]
            )

        # Update spike raster
        if "spikes" in data:
            spikes_matrix = data["spikes"]

            # Find the (time, neuron) coordinates of each spike
            neuron_indices, time_indices_in_chunk = np.where(spikes_matrix)

            if len(neuron_indices) > 0:
                # Convert chunk time indices to absolute time in seconds
                absolute_time_indices = start_index + time_indices_in_chunk
                time_coords_s = absolute_time_indices / self.sampling_rate

                # Set the data for the scatter plot
                self.realtime_scatters["spikes"].setData(
                    x=time_coords_s, y=neuron_indices
                )
            else:
                # Clear the plot if there are no spikes
                self.realtime_scatters["spikes"].setData([], [])

            # Add colored time windows for accepted and rejected spiking periods
            # Clear existing regions
            for item in self.realtime_plots["spikes"].items[:]:
                if isinstance(item, pg.LinearRegionItem):
                    self.realtime_plots["spikes"].removeItem(item)

            # Add green regions for accepted spiking periods
            if (
                "accepted_peaks_with_periods" in data
                and data["accepted_peaks_with_periods"]
            ):
                for peak_info in data["accepted_peaks_with_periods"]:
                    spiking_period = peak_info["spiking_period"]
                    start_time = (start_index + spiking_period[0]) / self.sampling_rate
                    end_time = (start_index + spiking_period[1]) / self.sampling_rate

                    region = pg.LinearRegionItem(
                        values=[start_time, end_time],
                        brush=pg.mkBrush(0, 255, 0, 50),  # Green with transparency
                    )
                    self.realtime_plots["spikes"].addItem(region)

            # Add yellow regions for rejected spiking periods
            if "rejected_candidates" in data and data["rejected_candidates"]:
                for rejected in data["rejected_candidates"]:
                    spiking_period = rejected["spiking_period"]
                    start_time = (start_index + spiking_period[0]) / self.sampling_rate
                    end_time = (start_index + spiking_period[1]) / self.sampling_rate

                    region = pg.LinearRegionItem(
                        values=[start_time, end_time],
                        brush=pg.mkBrush(255, 255, 0, 50),  # Yellow with transparency
                    )
                    self.realtime_plots["spikes"].addItem(region)

        # Update detected peaks
        if "detected_peaks" in data and len(data["detected_peaks"]) > 0:
            detected_indices_in_chunk = np.array(data["detected_peaks"])

            # Get amplitudes from the filtered signal at the peak locations
            if "filtered_signal" in data:
                valid_indices_in_chunk = [
                    p
                    for p in detected_indices_in_chunk
                    if 0 <= p < len(data["filtered_signal"])
                ]
                if valid_indices_in_chunk:
                    detected_amplitudes = data["filtered_signal"][
                        valid_indices_in_chunk
                    ]

                    # Convert valid chunk indices to absolute time for plotting
                    valid_absolute_indices = start_index + np.array(
                        valid_indices_in_chunk
                    )
                    valid_times_s = valid_absolute_indices / self.sampling_rate

                    # Plot accepted peaks in green
                    self.realtime_scatters["detected_peaks"].setData(
                        x=valid_times_s, y=detected_amplitudes
                    )
                else:
                    self.realtime_scatters["detected_peaks"].setData([], [])
            else:
                self.realtime_scatters["detected_peaks"].setData([], [])
        elif "detected_peaks" in data:
            self.realtime_scatters["detected_peaks"].setData([], [])

        # Update rejected candidates (plot in yellow)
        if "rejected_candidates" in data and len(data["rejected_candidates"]) > 0:
            rejected_times_s = []
            rejected_amplitudes = []

            for rejected in data["rejected_candidates"]:
                chunk_index = rejected["detection"]
                if 0 <= chunk_index < len(data["filtered_signal"]):
                    absolute_time = start_index + chunk_index
                    rejected_times_s.append(absolute_time / self.sampling_rate)
                    rejected_amplitudes.append(data["filtered_signal"][chunk_index])

            self.realtime_scatters["rejected_peaks"].setData(
                x=rejected_times_s, y=rejected_amplitudes
            )
        else:
            self.realtime_scatters["rejected_peaks"].setData([], [])

    def start_realtime_visualization(self):
        """Start the real-time visualization window."""
        if not PYTQTGRAPH_AVAILABLE:
            print("PyQtGraph not available. Cannot start real-time visualization.")
            return

        self.realtime_win.show()
        print("Real-time visualization started. Close the window to stop.")

    def stop_realtime_visualization(self):
        """Stop the real-time visualization updates and wait for the user to close the window."""
        if self.realtime_timer:
            self.realtime_timer.stop()

        if self.realtime_app and self.realtime_win:
            print("Processing finished. Close the visualization window to exit.")
            # Start the event loop. This will block until the window is closed by the user.
            self.realtime_app.exec_()

    def update_realtime_data(
        self,
        ecg_signal=None,
        filtered_signal=None,
        spikes=None,
        peaks=None,
        detected_peaks=None,
        spiking_periods=None,
        start_index=None,
        rejected_candidates=None,
        accepted_peaks_with_periods=None,
    ):
        """
        Update real-time visualization with new data.

        Args:
            ecg_signal: Raw ECG signal array
            filtered_signal: Filtered and squared ECG signal array
            spikes: Spike raster matrix (neurons x time)
            peaks: True R-peak indices
            detected_peaks: Detected R-peak indices
            spiking_periods: List of (start, end) tuples for spiking periods
            start_index: The starting index of the chunk in the original signal
            rejected_candidates: Rejected candidate indices
            accepted_peaks_with_periods: Accepted peaks with spiking period information
        """
        if not PYTQTGRAPH_AVAILABLE:
            return

        # Prepare data for plotting
        data = {}

        if ecg_signal is not None:
            data["ecg_signal"] = ecg_signal

        if filtered_signal is not None:
            data["filtered_signal"] = filtered_signal

        if spikes is not None:
            data["spikes"] = spikes

        if peaks is not None:
            data["peaks"] = peaks
            if filtered_signal is not None and len(peaks) > 0:
                valid_peaks = [p for p in peaks if 0 <= p < len(filtered_signal)]
                data["peak_amplitudes"] = (
                    filtered_signal[valid_peaks] if valid_peaks else np.ones(len(peaks))
                )

        if detected_peaks is not None:
            data["detected_peaks"] = detected_peaks
            if filtered_signal is not None and len(detected_peaks) > 0:
                valid_detected = [
                    p for p in detected_peaks if 0 <= p < len(filtered_signal)
                ]
                data["detected_amplitudes"] = (
                    filtered_signal[valid_detected]
                    if valid_detected
                    else np.ones(len(detected_peaks))
                )

        if spiking_periods is not None:
            data["spiking_periods"] = spiking_periods

        if start_index is not None:
            data["start_index"] = start_index

        if rejected_candidates is not None:
            data["rejected_candidates"] = rejected_candidates

        if accepted_peaks_with_periods is not None:
            data["accepted_peaks_with_periods"] = accepted_peaks_with_periods

        # Update data queues (limited to 10 frames)
        self.data_queues["ecg"].append(data)
        if len(self.data_queues["ecg"]) > 10:
            self.data_queues["ecg"].pop(0)

    def plot_chunk_realtime(
        self,
        start_index,
        end_index,
        spikes_layer1,
        spikes_layer2,
        accepted_peaks_with_periods,
        spiking_periods,
        ecg_signal,
        chunk_squared_signal,
        rejected_candidates=None,
    ):
        """
        Real-time variant of plot_chunk_v3 using PyQtGraph.
        Updates the visualization with new chunk data.
        """
        if not PYTQTGRAPH_AVAILABLE:
            print("PyQtGraph not available. Cannot plot real-time data.")
            return

        # Extract the ECG chunk from the full signal
        chunk_ecg = ecg_signal[start_index:end_index]

        # Extract detection indices from accepted_peaks_with_periods
        detected_peaks = (
            [peak["detection"] for peak in accepted_peaks_with_periods]
            if accepted_peaks_with_periods
            else []
        )

        self.update_realtime_data(
            ecg_signal=chunk_ecg,
            filtered_signal=chunk_squared_signal,
            spikes=spikes_layer1,
            detected_peaks=detected_peaks,
            spiking_periods=spiking_periods,
            accepted_peaks_with_periods=accepted_peaks_with_periods,
            rejected_candidates=rejected_candidates,
            start_index=start_index,
        )

    def _plot_burst_annotations(self, axes, spikes_layer1, spiking_periods):
        burst_features = [
            extract_features_for_burst(spikes_layer1, start, end, time_step=1)
            for start, end in spiking_periods
        ]
        for feature in burst_features:
            is_accepted = feature["burst_type"] == "accepted"
            start = feature["start_time"]
            end = feature["end_time"]
            axes.axvspan(start, end, color="green" if is_accepted else "yellow", alpha=0.3)
