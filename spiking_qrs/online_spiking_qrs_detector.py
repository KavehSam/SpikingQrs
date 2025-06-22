import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import multiprocessing as mp
from typing import Optional, TYPE_CHECKING

from snn_models.adaptive_homesatic_two_layers import AdaptiveHomeostasisTwoLayers
from spiking_qrs.preprocessor import preprocess_ecg_signal
from spiking_qrs.raster_fex import *
from utils.snippets import generate_spike_input

if TYPE_CHECKING:
    from plotting.spiking_visualizer import SpikingVisualizer


class OnlineSpikingQrsDetector:
    def __init__(
        self,
        sampling_rate: float,
        net: AdaptiveHomeostasisTwoLayers,
        chunk_size: int = 60,
        overlap: int = 4,
        plotting: bool = False,
        visualizer: "Optional[SpikingVisualizer]" = None,
        return_spikes: bool = False,
        is_ecg_raw: bool = True,
        record_name: str | None = None,
    ) -> None:
        self.net = net
        self.sampling_rate = sampling_rate
        self.chunk_size = int(chunk_size * self.sampling_rate)
        self.overlap = int(overlap * self.sampling_rate)
        self.plotting = plotting
        self.visualizer = visualizer
        self.return_spikes = return_spikes
        self.do_log = False
        self.writer: Optional[SummaryWriter] = None
        self.is_ecg_raw = is_ecg_raw
        self.sl = self.nl = None
        self.record_name = record_name
        self.last_r_peak_fex: dict = {
            "Burst Duration": 0,
            "Total Spikes": 0,
            "Spike Rate": 0,
            "Neuron Participation": 0,
            "ISI CV": 0,
            "Mean ISI": 0,
        }
        self.all_spikes_layer_1: list = []
        self.all_spikes_layer_2: list = []

        # Processing state
        self.ecg_signal: Optional[np.ndarray] = None
        self.squared_signal: Optional[np.ndarray] = None
        self.filtered_signal: Optional[np.ndarray] = None
        self.r_peaks: list = []
        
        # Tracking rejected candidates for analysis
        self.rejected_candidates: list = []

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        self.do_log = True
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def preprocess(self):
        self.filtered_signal, self.squared_signal, self.smoothed_signal = (
            preprocess_ecg_signal(self.ecg_signal, self.sampling_rate)
        )
        self.smoothed_signal = None
        if not self.plotting:
            self.filtered_signal = None

    def detect_beats_multi(self, ecg_signal):
        # Multi-lead beat detection is not implemented in this version
        raise NotImplementedError(
            "Multi-lead beat detection is excluded in this version"
        )

    def detect_beat_single_lead(
        self, ecg_signal: np.ndarray
    ) -> tuple[np.ndarray, list, list]:
        self.ecg_signal = ecg_signal
        self.r_peaks = []
        self.rejected_candidates = []
        self.last_r_peak_fex = {
            "Burst Duration": 0,
            "Total Spikes": 0,
            "Spike Rate": 0,
            "Neuron Participation": 0,
            "ISI CV": 0,
            "Mean ISI": 0,
        }

        self.preprocess()

        # Start real-time visualization if enabled
        if self.plotting and self.visualizer:
            self.visualizer.start_realtime_visualization()

        if self.squared_signal is None:
            return np.array([]), [], []

        chunk_indices = range(
            self.chunk_size, len(self.squared_signal) + 1, self.chunk_size
        )

        for chunk_start_ind in chunk_indices:
            (
                adjusted_chunk_r_peaks,
                spikes_layer1,
                spikes_layer2,
            ) = self._detect_r_peaks_chunk(chunk_start_ind)

            if adjusted_chunk_r_peaks is None:
                continue

            self.r_peaks += adjusted_chunk_r_peaks
            # remove duplicates from self.r_peaks
            self.r_peaks = list(np.unique(self.r_peaks))

            if self.return_spikes:
                self.all_spikes_layer_1.append(spikes_layer1)
                self.all_spikes_layer_2.append(spikes_layer2)

            if self.do_log and self.writer:
                if spikes_layer1 is not None:
                    self.writer.add_histogram(
                        "Spikes Layer 1",
                        np.array(spikes_layer1),
                        global_step=chunk_start_ind,
                    )
                if spikes_layer2 is not None:
                    self.writer.add_histogram(
                        "Spikes Layer 2",
                        np.array(spikes_layer2),
                        global_step=chunk_start_ind,
                    )
                self.writer.add_scalar(
                    "Chunk Start Index", chunk_start_ind, global_step=chunk_start_ind
                )
            
            # Process GUI events to keep the window responsive
            if self.plotting and self.visualizer and self.visualizer.realtime_app:
                self.visualizer.realtime_app.processEvents()

            # Release memory
            spikes_layer1, spikes_layer2 = (
                None,
                None,
            )
            # torch.cuda.empty_cache()

        if self.do_log and self.writer is not None:
            self.writer.close()

        # Stop real-time visualization if enabled
        if self.plotting and self.visualizer:
            self.visualizer.stop_realtime_visualization()

        return (
            np.unique(np.array(self.r_peaks)),
            self.all_spikes_layer_1,
            self.all_spikes_layer_2,
        )

    def _detect_r_peaks_chunk(self, chunk_start_ind):
        if self.squared_signal is None:
            return None, None, None

        end_index = min(len(self.squared_signal), chunk_start_ind)
        start_index = max(0, end_index - self.chunk_size - self.overlap)
        if end_index - start_index < self.chunk_size:
            return None, None, None

        chunk_squared = self.squared_signal[start_index:end_index]
        num_samples = end_index - start_index

        input_array, _ = generate_spike_input(
            torch.from_numpy(chunk_squared), self.net.input_size
        )

        input_array = input_array.T.unsqueeze(1).clone().detach()

        self.net.set_num_time_steps(num_samples)
        self.net.reset_network()

        # Simulate the SNN
        spk1_rec_chunk, spk2_rec_chunk = self.net(input_array.unsqueeze(1))
        self.net.update_weights(spk1_rec_chunk, spk2_rec_chunk)

        spikes_layer1 = spk1_rec_chunk.squeeze().detach().numpy().T
        spikes_layer2 = spk2_rec_chunk.squeeze().detach().numpy().T

        # Detect burst spiking periods
        spiking_periods, _, _ = detect_spiking_periods(
            spikes_layer1,
            threshold_factor=3,
            smoothing=5,
            min_duration=int(self.sampling_rate * 0.03),
        )
        blocks = np.zeros(num_samples, dtype=np.int8)
        for _period in spiking_periods:
            blocks[_period[0] : _period[1]] = 1

        # Extract features for each spiking period
        spiking_periods_fex = [
            extract_features_for_burst(spikes_layer1, start, end, time_step=1)
            for start, end in spiking_periods
        ]

        chunk_r_peaks, rejected_candidates, accepted_peaks_with_periods = self.r_peak_qualification(
            spiking_periods, chunk_squared, start_index, spiking_periods_fex
        )

        # Make a copy for plotting before converting to absolute indices for global tracking
        plot_rejected_candidates = [rc.copy() for rc in rejected_candidates]

        # Accumulate rejected candidates with absolute indices
        for rejected in rejected_candidates:
            rejected['detection'] += start_index
            self.rejected_candidates.append(rejected)

        adjusted_chunk_r_peaks = [r_peak + start_index for r_peak in chunk_r_peaks] if chunk_r_peaks else []

        if self.plotting and self.visualizer:
            if self.ecg_signal is not None:
                self.visualizer.plot_chunk_realtime(
                    start_index,
                    end_index,
                    spikes_layer1,
                    spikes_layer2,
                    accepted_peaks_with_periods,
                    spiking_periods,
                    self.ecg_signal,
                    chunk_squared,
                    plot_rejected_candidates,
                )
            # The static plotting mode has been removed.

        return (adjusted_chunk_r_peaks, spikes_layer1, spikes_layer2)

    def r_peak_qualification(
        self, spiking_periods, chunk_squared, start_index, spiking_periods_fex
    ):
        rr_distance_threshold = int(0.25 * self.sampling_rate)  # 200 or 250ms

        chunk_max_amplitude = np.max(chunk_squared[-self.chunk_size :])
        amplitude_threshold = 0.5 * chunk_max_amplitude

        chunk_r_peaks = (
            [self.r_peaks[-1] - start_index] if len(self.r_peaks) > 0 else []
        )
        
        last_r_point_amp = 0.0
        if self.squared_signal is not None and len(self.r_peaks) > 0:
            last_r_point_amp = self.squared_signal[self.r_peaks[-1]]
        else:
            last_r_point_amp = amplitude_threshold

        start = end = 0

        r_peak_candidates = []  # Store (index, amplitude, fex, spiking_period) for pruning later
        avg_amp = 0
        fex_avg = {
            "Burst Duration": 0.0,
            "Total Spikes": 0.0,
            "Spike Rate": 0.0,
            "Neuron Participation": 0.0,
            "ISI CV": 0.0,
            "Mean ISI": 0.0,
        }

        for (start, end), fex in zip(spiking_periods, spiking_periods_fex):
            detection = np.argmax(abs(chunk_squared[start : end + 1])) + start
            amplitude = chunk_squared[detection]
            avg_amp += amplitude
            for key in fex_avg.keys():
                fex_avg[key] += fex[key]

            # Store detection with spiking period for later pruning
            r_peak_candidates.append((detection, amplitude, fex, (start, end)))

        if len(r_peak_candidates):
            avg_amp /= len(r_peak_candidates)
            amplitude_threshold = min(avg_amp, amplitude_threshold)

            for key in fex_avg.keys():
                fex_avg[key] /= len(r_peak_candidates)

        # Track rejected candidates
        rejected_candidates = []
        accepted_peaks_with_periods = []

        # Pruning based on most frequent amplitude (mode or median fallback)
        if r_peak_candidates:
            # Filter R-peak candidates based on amplitude threshold
            pruned_r_peaks = []
            for detection, amplitude, fex, spiking_period in r_peak_candidates:
                if (
                    (fex["Burst Duration"] >= fex_avg["Burst Duration"] / 2)
                    and (amplitude >= 0.01)
                ):
                    pruned_r_peaks.append((detection, amplitude, fex, spiking_period))
                else:
                    # Track rejected candidates with rejection reason
                    rejection_reason = []
                    if fex["Burst Duration"] < fex_avg["Burst Duration"] / 2:
                        rejection_reason.append("low_burst_duration")
                    if amplitude < 0.01:
                        rejection_reason.append("low_amplitude")
                    rejected_candidates.append({
                        'detection': detection,
                        'amplitude': amplitude,
                        'fex': fex,
                        'spiking_period': spiking_period,
                        'rejection_reason': rejection_reason
                    })

            # Handle RR interval and update
            for i, (detection, amplitude, fex, spiking_period) in enumerate(pruned_r_peaks):
                if len(chunk_r_peaks) > 0:
                    ## If there are previous R-peaks either in this chunk or before it, check RR interval
                    if detection - chunk_r_peaks[-1] > rr_distance_threshold:
                        is_fp = False
                        if self.last_r_peak_fex:
                           is_fp = is_burst_block_fp(self.last_r_peak_fex, fex)

                        if amplitude < 0.1 * last_r_point_amp:
                            if is_fp or is_burst_block_outlier(fex_avg, fex):
                                # Track rejected candidates due to burst block
                                rejection_reason = []
                                if is_fp:
                                    rejection_reason.append("burst_block_fp")
                                if is_burst_block_outlier(fex_avg, fex):
                                    rejection_reason.append("burst_block_outlier")
                                rejected_candidates.append({
                                    'detection': detection,
                                    'amplitude': amplitude,
                                    'fex': fex,
                                    'spiking_period': spiking_period,
                                    'rejection_reason': rejection_reason
                                })
                                continue

                        chunk_r_peaks.append(detection)
                        last_r_point_amp = amplitude
                        self.last_r_peak_fex = fex
                        accepted_peaks_with_periods.append({
                            'detection': detection,
                            'amplitude': amplitude,
                            'fex': fex,
                            'spiking_period': spiking_period
                        })

                    elif ((detection - chunk_r_peaks[-1]) > 0) and (
                        (detection - chunk_r_peaks[-1]) < rr_distance_threshold
                    ):
                        # happens after last R-peak but RR interval is less than threshold
                        if amplitude > last_r_point_amp and last_r_point_amp != 0:
                            last_r_point_amp = amplitude
                            self.last_r_peak_fex = fex
                            chunk_r_peaks[-1] = detection
                            # Update the last accepted peak with new information
                            if accepted_peaks_with_periods:
                                accepted_peaks_with_periods[-1] = {
                                    'detection': detection,
                                    'amplitude': amplitude,
                                    'fex': fex,
                                    'spiking_period': spiking_period
                                }
                            if len(chunk_r_peaks) == 1:
                                # it is first detection in the chunk
                                if len(self.r_peaks) > 0:
                                    # also replaces the last R-peak in the global list
                                    self.r_peaks[-1] = detection + start_index

                    else:  # happens before last R-peak
                        if (chunk_r_peaks[-1] - detection) < rr_distance_threshold:
                            # if the detection is before the last R-peak but within the RR interval distance threshold
                            if amplitude > last_r_point_amp:
                                # replace the last R-peak with the new detection
                                last_r_point_amp = amplitude
                                self.last_r_peak_fex = fex
                                if len(chunk_r_peaks) > 1:
                                    chunk_r_peaks[-1] = detection
                                else:
                                    self.r_peaks[-1] = detection + start_index
                                # Update the last accepted peak with new information
                                if accepted_peaks_with_periods:
                                    accepted_peaks_with_periods[-1] = {
                                        'detection': detection,
                                        'amplitude': amplitude,
                                        'fex': fex,
                                        'spiking_period': spiking_period
                                    }
                        else:
                            # if the detection is before the last R-peak but outside the RR interval distance threshold
                            if fex["Burst Duration"] > fex_avg["Burst Duration"]:
                                # check if detection + start_index is not in self.r_peaks
                                if (detection + start_index) not in self.r_peaks:
                                    # check if there is any value in self.r_peaks which its distance to detection + start_index is less than rr_distance_threshold
                                    closeest_r_peak = [
                                        (indx, r_peak)
                                        for indx, r_peak in enumerate(self.r_peaks)
                                        if abs((detection + start_index) - r_peak)
                                        < rr_distance_threshold
                                    ]
                                    if len(closeest_r_peak):
                                        if (
                                            self.squared_signal is not None and
                                            amplitude
                                            > self.squared_signal[closeest_r_peak[0][1]]
                                        ):
                                            # if the amplitude of the detection is greater than the amplitude of the closest R-peak
                                            # then replace the index of that with detection + start_index
                                            self.r_peaks[closeest_r_peak[0][0]] = (
                                                detection + start_index
                                            )
                                    else:
                                        if not is_burst_block_outlier(fex_avg, fex):
                                            # find the index of last smaller value than detection + start_index in self.r_peaks
                                            if len(self.r_peaks) > 1 and self.squared_signal is not None:
                                                indx = np.where(
                                                    np.array(self.r_peaks)
                                                    < (detection + start_index)
                                                )[0]
                                                # insert detection + start_index after that index
                                                if len(indx):
                                                    self.r_peaks.insert(
                                                        indx[-1] + 1,
                                                        detection + start_index,
                                                    )
                                        else:
                                            # Track rejected candidates due to burst block outlier
                                            rejected_candidates.append({
                                                'detection': detection,
                                                'amplitude': amplitude,
                                                'fex': fex,
                                                'spiking_period': spiking_period,
                                                'rejection_reason': ["burst_block_outlier"]
                                            })
                else:
                    # first detection
                    if amplitude > last_r_point_amp:
                        chunk_r_peaks.append(detection)
                        last_r_point_amp = amplitude
                        self.last_r_peak_fex = fex
                        accepted_peaks_with_periods.append({
                            'detection': detection,
                            'amplitude': amplitude,
                            'fex': fex,
                            'spiking_period': spiking_period
                        })

        return chunk_r_peaks, rejected_candidates, accepted_peaks_with_periods
