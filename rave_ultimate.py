#!/usr/bin/env python3
"""
🎛️ ULTIMATE AUDIO PROCESSOR 3.0
=================================
Zawiera 10+ metod odszumiania + RAVE AI:

PODSTAWOWE:
- Noise Gate (próg tłumienia)
- Band-pass Filter (80Hz-15kHz)
- High-pass Filter (usuwa rumble)
- Low-pass Filter (usuwa hiss)

ZAAWANSOWANE:
- Spectral Subtraction (profil szumu)
- Wiener Filter (adaptacyjny)
- Median Filter (impulse noise)
- Kalman Filter (tracking)

AI-POWERED:
- DeepFilterNet (DNN denoising)
- RAVE (style transfer)

EKSPERYMENTALNE:
- Multi-band Gate (3 pasma)
- Adaptive Spectral Gate
"""

import sys
import os
import subprocess
import numpy as np
import gi
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import butter, lfilter, wiener, medfilt
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d, PchipInterpolator
import threading
import json

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib

import torch
import sounddevice as sd

# Patch dla DeepFilter
original_query = sd.query_devices
def safe_query_devices(device=None, kind=None):
    try:
        if device is None and kind is None:
            return original_query()
        res = original_query(device, kind)
        return res if isinstance(res, dict) else res[0]
    except:
        return {'name': 'Default', 'default_samplerate': 48000, 'max_input_channels': 2}
sd.query_devices = safe_query_devices

# Import DeepFilter
try:
    from df.enhance import init_df
    from df.modules import get_device
    DF_AVAILABLE = True
except Exception as e:
    DF_AVAILABLE = False

Gst.init(None)

# ==========================================
# 🔧 SETUP
# ==========================================
def create_virtual_sink(sink_name="autoeq_sink"):
    try:
        res = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True)
        if sink_name in res.stdout: return
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={sink_name}", 
             "sink_properties=device.description='UltimateAudioProcessor'"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"✅ Virtual sink: {sink_name}")
    except: pass

# ==========================================
# 🎚️ NOISE REDUCTION ALGORITHMS
# ==========================================

class SimpleNoiseGate:
    """Podstawowy gate z hysteresis"""
    def __init__(self, threshold=-40, attack=0.01, release=0.1, fs=48000):
        self.threshold_db = threshold
        self.threshold_lin = 10 ** (threshold / 20)
        self.attack_samples = int(attack * fs)
        self.release_samples = int(release * fs)
        self.envelope = 0
        self.is_open = False
        
    def process(self, audio):
        result = np.zeros_like(audio)
        for i in range(len(audio)):
            frame = audio[i]
            level = np.sqrt(np.mean(frame**2))
            
            # Envelope follower
            if level > self.envelope:
                self.envelope += (level - self.envelope) / self.attack_samples
            else:
                self.envelope += (level - self.envelope) / self.release_samples
            
            # Gate logic z hysteresis
            if self.envelope > self.threshold_lin * 1.2:
                self.is_open = True
            elif self.envelope < self.threshold_lin * 0.8:
                self.is_open = False
            
            result[i] = frame if self.is_open else frame * 0.1
            
        return result

class MultiBandGate:
    """Gate działający osobno dla 3 pasm częstotliwości"""
    def __init__(self, fs=48000):
        self.fs = fs
        # Low: 20-250Hz, Mid: 250-4kHz, High: 4k-20kHz
        self.filters = self._create_filters()
        self.gates = {
            'low': SimpleNoiseGate(threshold=-50, fs=fs),
            'mid': SimpleNoiseGate(threshold=-45, fs=fs),
            'high': SimpleNoiseGate(threshold=-40, fs=fs)
        }
        
    def _create_filters(self):
        nyq = self.fs / 2
        filters = {}
        
        # Lowpass dla low (250Hz)
        b_low, a_low = butter(4, 250 / nyq, btype='low')
        filters['low'] = (b_low, a_low)
        
        # Bandpass dla mid (250-4000Hz)
        b_mid, a_mid = butter(4, [250 / nyq, 4000 / nyq], btype='band')
        filters['mid'] = (b_mid, a_mid)
        
        # Highpass dla high (4000Hz)
        b_high, a_high = butter(4, 4000 / nyq, btype='high')
        filters['high'] = (b_high, a_high)
        
        return filters
    
    def process(self, audio):
        bands = {}
        for name, (b, a) in self.filters.items():
            if audio.ndim == 1:
                bands[name] = lfilter(b, a, audio)
            else:
                bands[name] = np.column_stack([
                    lfilter(b, a, audio[:, ch]) for ch in range(audio.shape[1])
                ])
        
        # Zastosuj gate do każdego pasma
        processed_bands = {
            name: self.gates[name].process(band) 
            for name, band in bands.items()
        }
        
        # Sumuj
        return sum(processed_bands.values())

class HighPassFilter:
    """Usuwa ultra-niskie częstotliwości (rumble)"""
    def __init__(self, cutoff=80, fs=48000, order=6):
        nyq = fs / 2
        self.b, self.a = butter(order, cutoff / nyq, btype='high')
        
    def process(self, audio):
        if audio.ndim == 1:
            return lfilter(self.b, self.a, audio)
        return np.column_stack([
            lfilter(self.b, self.a, audio[:, ch]) 
            for ch in range(audio.shape[1])
        ])

class LowPassFilter:
    """Usuwa ultra-wysokie częstotliwości (hiss)"""
    def __init__(self, cutoff=15000, fs=48000, order=6):
        nyq = fs / 2
        self.b, self.a = butter(order, cutoff / nyq, btype='low')
        
    def process(self, audio):
        if audio.ndim == 1:
            return lfilter(self.b, self.a, audio)
        return np.column_stack([
            lfilter(self.b, self.a, audio[:, ch]) 
            for ch in range(audio.shape[1])
        ])

class SpectralSubtraction:
    """Odejmowanie spektralne z over-subtraction"""
    def __init__(self, noise_factor=1.5, over_sub=1.0):
        self.noise_profile = None
        self.noise_factor = noise_factor
        self.over_sub = over_sub
        self.learning = True
        self.frames_collected = 0
        
    def learn_noise(self, audio, max_frames=30):
        if self.frames_collected < max_frames:
            spectrum = np.abs(rfft(audio, axis=0))
            if self.noise_profile is None:
                self.noise_profile = spectrum
            else:
                alpha = 0.95
                self.noise_profile = alpha * self.noise_profile + (1 - alpha) * spectrum
            self.frames_collected += 1
            if self.frames_collected >= max_frames:
                self.learning = False
                print("✅ Spectral profile learned")
    
    def process(self, audio):
        if self.learning:
            self.learn_noise(audio)
            return audio
            
        if self.noise_profile is None:
            return audio
        
        spectrum = rfft(audio, axis=0)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Over-subtraction
        clean_mag = magnitude - self.noise_factor * self.over_sub * self.noise_profile
        
        # Spectral floor
        clean_mag = np.maximum(clean_mag, 0.05 * magnitude)
        
        clean_spectrum = clean_mag * np.exp(1j * phase)
        return irfft(clean_spectrum, n=len(audio), axis=0)

class AdaptiveSpectralGate:
    """Gate w domenie częstotliwości - każdy bin osobno"""
    def __init__(self, threshold_factor=1.5):
        self.noise_floor = None
        self.threshold_factor = threshold_factor
        self.learning = True
        self.frames_collected = 0
        
    def learn_noise(self, audio, max_frames=20):
        if self.frames_collected < max_frames:
            spectrum = np.abs(rfft(audio, axis=0))
            if self.noise_floor is None:
                self.noise_floor = spectrum
            else:
                self.noise_floor = np.minimum(self.noise_floor, spectrum)
            self.frames_collected += 1
            if self.frames_collected >= max_frames:
                self.learning = False
                print("✅ Spectral gate calibrated")
    
    def process(self, audio):
        if self.learning:
            self.learn_noise(audio)
            return audio
            
        if self.noise_floor is None:
            return audio
        
        spectrum = rfft(audio, axis=0)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Gate dla każdego binu
        threshold = self.noise_floor * self.threshold_factor
        mask = magnitude > threshold
        
        # Smooth mask
        gated_mag = np.where(mask, magnitude, magnitude * 0.1)
        
        gated_spectrum = gated_mag * np.exp(1j * phase)
        return irfft(gated_spectrum, n=len(audio), axis=0)

class MedianFilter:
    """Median filter - dobry na impulse noise (kliknięcia)"""
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        
    def process(self, audio):
        if audio.ndim == 1:
            return medfilt(audio, kernel_size=self.kernel_size)
        
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] = medfilt(audio[:, ch], kernel_size=self.kernel_size)
        return result

class KalmanFilter:
    """Simplified Kalman filter for audio tracking"""
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.P = 1.0  # Estimate error
        self.x = 0.0  # State estimate
        
    def process(self, audio):
        result = np.zeros_like(audio)
        
        if audio.ndim == 1:
            for i in range(len(audio)):
                # Predict
                self.P = self.P + self.Q
                
                # Update
                K = self.P / (self.P + self.R)
                self.x = self.x + K * (audio[i] - self.x)
                self.P = (1 - K) * self.P
                
                result[i] = self.x
        else:
            for ch in range(audio.shape[1]):
                x = 0.0
                P = 1.0
                for i in range(len(audio)):
                    P = P + self.Q
                    K = P / (P + self.R)
                    x = x + K * (audio[i, ch] - x)
                    P = (1 - K) * P
                    result[i, ch] = x
                    
        return result

class WienerFilter:
    """Wiener filter"""
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        
    def process(self, audio):
        if audio.ndim == 1:
            return wiener(audio, noise=self.noise_level)
        
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] = wiener(audio[:, ch], noise=self.noise_level)
        return result

class DeclickInpainter:
    """Wykrywa i naprawia trzaski przez inpainting"""
    def __init__(self, fs=48000):
        self.fs = fs
        self.threshold = 0.7  # Próg amplitudy dla peak detection
        self.derivative_threshold = 0.3
        self.mask_expansion = 5  # Próbki wokół defektu
        
    def detect_clicks(self, audio):
        """Wykrywa trzaski przez analizę amplitudy i pochodnej"""
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
        else:
            mono = audio
        
        defects = []
        
        # 1. Peak detection (amplitude spikes)
        peaks = np.where(np.abs(mono) > self.threshold)[0]
        defects.extend(peaks)
        
        # 2. Derivative analysis (nagłe skoki)
        # ODWRÓCONA LOGIKA: wyższy threshold = więcej wykrywa
        diff = np.diff(mono, prepend=mono[0])
        diff2 = np.diff(diff, prepend=diff[0])
        abs_diff2 = np.abs(diff2)
        sensitivity_factor = 1.0 / max(self.derivative_threshold, 0.05)
        threshold_val = sensitivity_factor * np.percentile(abs_diff2, 99)
        spikes = np.where(abs_diff2 > threshold_val)[0]
        defects.extend(spikes)
        
        return np.unique(defects)
    
    def expand_mask(self, defects, length):
        """Rozszerza maskę o sąsiednie próbki"""
        mask = np.zeros(length, dtype=bool)
        for idx in defects:
            start = max(0, idx - self.mask_expansion)
            end = min(length, idx + self.mask_expansion + 1)
            mask[start:end] = True
        return mask
    
    def interpolate_cubic(self, audio, mask):
        """Interpolacja kubiczna PCHIP"""
        result = audio.copy()
        good_idx = np.where(~mask)[0]
        bad_idx = np.where(mask)[0]
        
        if len(good_idx) < 4 or len(bad_idx) == 0:
            return result
        
        try:
            interpolator = PchipInterpolator(good_idx, audio[good_idx])
            result[bad_idx] = interpolator(bad_idx)
        except:
            # Fallback do linear
            if len(good_idx) >= 2:
                interpolator = interp1d(good_idx, audio[good_idx], 
                                       kind='linear', fill_value='extrapolate')
                result[bad_idx] = interpolator(bad_idx)
        
        return result
    
    def unsharp_mask(self, audio, amount=0.3, sigma=1.5):
        """Deblur przez unsharp masking"""
        from scipy.ndimage import gaussian_filter1d
        blurred = gaussian_filter1d(audio, sigma=sigma)
        detail = audio - blurred
        return audio + amount * detail
    
    def process(self, audio):
        """Kompletny pipeline: detect → mask → interpolate → deblur"""
        original_shape = audio.shape
        
        # Mono processing
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
            is_stereo = True
        else:
            mono = audio
            is_stereo = False
        
        # Detect
        defects = self.detect_clicks(mono)
        
        if len(defects) == 0:
            return audio
        
        # Mask
        mask = self.expand_mask(defects, len(mono))
        
        # Interpolate
        repaired = self.interpolate_cubic(mono, mask)
        
        # Deblur
        repaired = self.unsharp_mask(repaired, amount=0.3, sigma=1.5)
        
        # Soft limiting
        repaired = np.clip(repaired, -0.98, 0.98)
        
        # Stereo conversion
        if is_stereo:
            result = np.column_stack([repaired, repaired])
        else:
            result = repaired
        
        return result

class DeepFilterProcessor:
    """DeepFilterNet AI denoising"""
    def __init__(self):
        self.model = None
        self.df_state = None
        self.available = DF_AVAILABLE
        
    def load_model(self):
        if not self.available: return False
        try:
            self.model, self.df_state, _ = init_df()
            self.model.eval()
            print("✅ DeepFilterNet loaded")
            return True
        except Exception as e:
            print(f"❌ DeepFilter error: {e}")
            return False
    
    def process(self, audio):
        if self.model is None: return audio
        try:
            with torch.no_grad():
                if audio.ndim == 2:
                    mono = audio.mean(axis=1)
                else:
                    mono = audio
                    
                tensor = torch.from_numpy(mono).float().to(get_device()).unsqueeze(0).unsqueeze(0)
                enhanced = self.model(tensor, self.df_state)
                result_mono = enhanced.squeeze().cpu().numpy()
                
                if audio.ndim == 2:
                    result = np.column_stack([result_mono, result_mono])
                else:
                    result = result_mono
                    
                return result[:len(audio)]
        except:
            return audio

# ==========================================
# 🧠 MAIN PROCESSOR
# ==========================================
class UltimateProcessor:
    def __init__(self, model_path="vintage.ts"):
        self.model_path = model_path
        self.rave_model = None
        self.stream = None
        self.active = False
        self.status_callback = None
        self.fs = 48000
        self.bs = 2048
        self.input_gain = 3.5
        
        # Initialize all processors
        self.processors = {
            # Inpainting
            'declick': DeclickInpainter(fs=self.fs),
            
            # Basic
            'simple_gate': SimpleNoiseGate(threshold=-45, fs=self.fs),
            'highpass': HighPassFilter(cutoff=80, fs=self.fs),
            'lowpass': LowPassFilter(cutoff=15000, fs=self.fs),
            
            # Advanced
            'spectral_sub': SpectralSubtraction(noise_factor=1.3),
            'spectral_gate': AdaptiveSpectralGate(threshold_factor=1.8),
            'wiener': WienerFilter(noise_level=0.008),
            'median': MedianFilter(kernel_size=3),
            'kalman': KalmanFilter(process_noise=0.01, measurement_noise=0.15),
            
            # Multi-band
            'multiband_gate': MultiBandGate(fs=self.fs),
            
            # AI
            'deepfilter': DeepFilterProcessor(),
        }
        
        # Active methods
        self.active_methods = set()
        
        # Processing order (matters!)
        self.processing_order = [
            'declick',  # Remove clicks FIRST!
            'highpass',
            'lowpass', 
            'simple_gate',
            'multiband_gate',
            'median',
            'spectral_gate',
            'spectral_sub',
            'wiener',
            'kalman',
            'deepfilter',
            'rave'
        ]

    def load_rave_model(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️ RAVE not found: {self.model_path}")
            return False
        try:
            self.rave_model = torch.jit.load(self.model_path)
            self.rave_model.eval()
            print("✅ RAVE loaded")
            return True
        except Exception as e:
            print(f"❌ RAVE error: {e}")
            return False

    def _update_status(self, msg):
        if self.status_callback:
            GLib.idle_add(self.status_callback, msg)

    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Main processing loop"""
        # VU Meter
        volume = np.linalg.norm(indata) * 10
        if volume > 0.2:
            bars = int(min(volume, 40))
            methods_str = f"{len(self.active_methods)} active"
            sys.stdout.write(f"\r🎵 [{'█' * bars:<40}] {volume:.1f} | {methods_str}")
            sys.stdout.flush()

        # Pre-gain
        audio = indata * self.input_gain
        
        # Apply active methods in order
        for method in self.processing_order:
            if method not in self.active_methods:
                continue
                
            try:
                if method == 'rave':
                    if self.rave_model is not None:
                        with torch.no_grad():
                            # RAVE wymaga MONO - konwertuj stereo→mono
                            if audio.ndim == 2 and audio.shape[1] == 2:
                                mono = audio.mean(axis=1)  # Average L+R
                            else:
                                mono = audio.flatten()
                            
                            # Przetwórz jako mono: [1, 1, samples]
                            tensor = torch.from_numpy(mono).float().unsqueeze(0).unsqueeze(0)
                            output = self.rave_model(tensor)
                            
                            if isinstance(output, (list, tuple)): 
                                output = output[0]
                            elif hasattr(output, 'keys'): 
                                output = list(output.values())[0]
                            
                            # Konwertuj z powrotem na stereo
                            mono_result = output.squeeze().cpu().numpy()
                            if audio.ndim == 2:
                                # Duplikuj mono→stereo
                                audio = np.column_stack([mono_result, mono_result])
                            else:
                                audio = mono_result
                else:
                    audio = self.processors[method].process(audio)
                    
            except Exception as e:
                print(f"\n⚠️ Error in {method}: {e}")
                continue
        
        # Output
        n = min(len(audio), len(outdata))
        outdata[:n] = np.clip(audio[:n], -1, 1)
        if n < len(outdata): 
            outdata[n:] = 0

    def start(self):
        if self.active: return
        
        # Load models if needed
        if 'rave' in self.active_methods and self.rave_model is None:
            if not self.load_rave_model():
                self.active_methods.discard('rave')
                
        if 'deepfilter' in self.active_methods:
            if self.processors['deepfilter'].model is None:
                if not self.processors['deepfilter'].load_model():
                    self.active_methods.discard('deepfilter')

        try:
            self.stream = sd.Stream(
                samplerate=self.fs, blocksize=self.bs, channels=2,
                dtype='float32', latency='high', callback=self.audio_callback
            )
            self.stream.start()
            self.active = True
            
            msg = f"✅ Running: {len(self.active_methods)} methods"
            self._update_status(msg)
            print(f"\n{msg}: {list(self.active_methods)}")
            
        except Exception as e:
            self._update_status(f"❌ Error: {str(e)[:40]}")
            print(f"❌ {e}")
            self.active = False

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except: pass
            self.stream = None
        self.active = False
        print("\n⏹️ Stopped")

    def set_method(self, method, enabled):
        if enabled:
            self.active_methods.add(method)
        else:
            self.active_methods.discard(method)
        
        if self.active:
            msg = f"Active: {len(self.active_methods)} methods"
            self._update_status(msg)

# ==========================================
# 🎨 GUI
# ==========================================
class UltimateAudioApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="🎛️ Ultimate Audio Processor 3.0")
        self.set_default_size(800, 700)
        self.set_border_width(15)
        
        self.processor = UltimateProcessor()
        self.processor.status_callback = self.update_status

        # Scrollable window
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.add(scrolled)

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        scrolled.add(main_vbox)

        # Header
        header = Gtk.Label()
        header.set_markup("<span size='xx-large'><b>🎛️ Ultimate Audio Processor 3.0</b></span>")
        main_vbox.pack_start(header, False, False, 0)

        # Instructions
        info_frame = Gtk.Frame(label="📋 Setup")
        info_box = Gtk.Label()
        info_box.set_markup(
            "<b>1.</b> Play music in your system\n"
            "<b>2.</b> Pavucontrol → Playback → set to 'UltimateAudioProcessor'\n"
            "<b>3.</b> Pavucontrol → Recording → set to 'Monitor of UltimateAudioProcessor'\n"
            "<b>4.</b> Enable methods below and toggle Master ON"
        )
        info_box.set_padding(10, 10)
        info_box.set_line_wrap(True)
        info_frame.add(info_box)
        main_vbox.pack_start(info_frame, False, False, 0)

        # Methods organized by category
        categories = [
            ("✨ Audio Inpainting", [
                ('declick', 'Declick & Inpainting', 'Wykrywa i naprawia trzaski, crackling, iskry (peak detection + cubic interpolation + deblur)'),
            ]),
            ("🔰 Basic Filters", [
                ('highpass', 'High-pass Filter', 'Removes ultra-low frequencies (rumble) <80Hz'),
                ('lowpass', 'Low-pass Filter', 'Removes ultra-high frequencies (hiss) >15kHz'),
                ('simple_gate', 'Simple Noise Gate', 'Threshold-based silence suppression (-45dB)'),
            ]),
            ("🎚️ Advanced Filtering", [
                ('multiband_gate', 'Multi-band Gate', '3-band gate (Low/Mid/High) with separate thresholds'),
                ('spectral_gate', 'Adaptive Spectral Gate', 'Frequency-selective gate (learns noise floor)'),
                ('median', 'Median Filter', 'Removes impulse noise (clicks, pops)'),
            ]),
            ("📊 Spectral Methods", [
                ('spectral_sub', 'Spectral Subtraction', 'Subtracts learned noise profile from spectrum'),
                ('wiener', 'Wiener Filter', 'Statistical adaptive filter'),
                ('kalman', 'Kalman Filter', 'Recursive state estimation (smooth tracking)'),
            ]),
            ("🧠 AI-Powered", [
                ('deepfilter', 'DeepFilterNet', 'Deep neural network denoising (requires GPU)'),
                ('rave', 'RAVE Transform', 'Real-time audio variational encoder (vintage.ts)'),
            ])
        ]

        self.switches = {}
        
        for category_name, methods in categories:
            frame = Gtk.Frame(label=category_name)
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            vbox.set_border_width(10)
            frame.add(vbox)
            
            for method_id, name, description in methods:
                hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
                
                switch = Gtk.Switch()
                switch.connect("notify::active", self.on_method_toggled, method_id)
                self.switches[method_id] = switch
                
                label_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
                name_label = Gtk.Label(xalign=0)
                name_label.set_markup(f"<b>{name}</b>")
                desc_label = Gtk.Label(xalign=0)
                desc_label.set_markup(f"<small>{description}</small>")
                desc_label.set_line_wrap(True)
                desc_label.set_max_width_chars(60)
                
                label_box.pack_start(name_label, False, False, 0)
                label_box.pack_start(desc_label, False, False, 0)
                
                hbox.pack_start(switch, False, False, 0)
                hbox.pack_start(label_box, True, True, 0)
                vbox.pack_start(hbox, False, False, 0)
                
                if method_id != methods[-1][0]:
                    sep = Gtk.Separator()
                    vbox.pack_start(sep, False, False, 0)
            
            main_vbox.pack_start(frame, False, False, 0)

        # Preset buttons
        preset_frame = Gtk.Frame(label="⚡ Quick Presets")
        preset_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        preset_box.set_border_width(10)
        preset_frame.add(preset_box)
        
        presets = [
            ("🎤 Podcast", ['highpass', 'simple_gate', 'spectral_sub']),
            ("🎵 Music Clean", ['declick', 'highpass', 'lowpass', 'spectral_gate']),
            ("💿 Vinyl Restore", ['declick', 'median', 'spectral_sub', 'wiener']),
            ("🔊 Maximum", ['declick', 'highpass', 'lowpass', 'multiband_gate', 'spectral_sub', 'wiener']),
            ("🧠 AI Only", ['deepfilter']),
            ("❌ Clear All", [])
        ]
        
        for name, methods in presets:
            btn = Gtk.Button(label=name)
            btn.connect("clicked", self.on_preset_clicked, methods)
            preset_box.pack_start(btn, True, True, 0)
        
        main_vbox.pack_start(preset_frame, False, False, 0)

        # Master control
        control_frame = Gtk.Frame()
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=20)
        control_box.set_border_width(15)
        control_frame.add(control_box)
        
        self.master_switch = Gtk.Switch()
        self.master_switch.set_size_request(60, 40)
        self.master_switch.connect("notify::active", self.on_master_toggled)
        
        master_label = Gtk.Label()
        master_label.set_markup("<span size='large'><b>🔴 MASTER POWER</b></span>")
        
        control_box.pack_start(master_label, False, False, 0)
        control_box.pack_start(self.master_switch, False, False, 0)
        
        self.status_label = Gtk.Label()
        self.status_label.set_markup("<b>⏸️ Ready</b>")
        control_box.pack_end(self.status_label, False, False, 0)
        
        main_vbox.pack_start(control_frame, False, False, 0)

        # Warning
        if not DF_AVAILABLE:
            warn = Gtk.Label()
            warn.set_markup(
                "<span color='orange'><b>⚠️ DeepFilterNet unavailable</b>\n"
                "Install: <tt>pip install deepfilternet</tt></span>"
            )
            main_vbox.pack_start(warn, False, False, 0)

    def on_method_toggled(self, switch, _, method_id):
        self.processor.set_method(method_id, switch.get_active())
        if self.processor.active:
            self.restart_processor()

    def on_preset_clicked(self, button, methods):
        # Clear all
        for switch in self.switches.values():
            switch.set_active(False)
        
        # Set preset
        for method in methods:
            if method in self.switches:
                self.switches[method].set_active(True)
        
        if self.processor.active:
            self.restart_processor()

    def on_master_toggled(self, switch, _):
        if switch.get_active():
            self.processor.start()
        else:
            self.processor.stop()
            self.update_status("⏸️ Stopped")

    def restart_processor(self):
        if self.processor.active:
            self.processor.stop()
            GLib.timeout_add(250, lambda: (self.processor.start(), False)[1])

    def update_status(self, msg):
        self.status_label.set_markup(f"<b>{msg}</b>")

if __name__ == "__main__":
    print("🎛️ Ultimate Audio Processor 3.0")
    print("=" * 50)
    create_virtual_sink()
    
    app = UltimateAudioApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
