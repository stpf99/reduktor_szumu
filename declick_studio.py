#!/usr/bin/env python3
"""
🎚️ AUDIO DECLICK & INPAINTING STUDIO
=====================================
Profesjonalny procesor do usuwania trzasków, crackling i defektów z audio.

Algorytm:
1. DETECT: Wykryj defekty (amplitude peaks, derivative spikes, statistical outliers)
2. MASK: Zamaskuj defekty + expansion
3. INPAINT: Wypełnij cubic/spectral interpolation
4. RESTORE: Deblur, enhance, normalize

Zastosowania:
- Winyle / kasety (crackling, pops)
- Nagrania archiwalne (degradacja)
- Live recordings (interference, clicks)
- Digital artifacts (clipping, dropouts)
"""

import sys
import os
import subprocess
import numpy as np
import gi
from scipy.signal import find_peaks, medfilt
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

gi.require_version('Gtk', '3.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gst, GLib

import sounddevice as sd

Gst.init(None)

# ==========================================
# SETUP
# ==========================================
def create_virtual_sink(sink_name="declick_sink"):
    try:
        res = subprocess.run(["pactl", "list", "short", "sinks"], 
                           capture_output=True, text=True)
        if sink_name in res.stdout: return
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", 
             f"sink_name={sink_name}", 
             "sink_properties=device.description='DeclickStudio'"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"✅ Virtual sink: {sink_name}")
    except: pass

# ==========================================
# CORE PROCESSING
# ==========================================

class AdvancedDeclicker:
    """Zaawansowany declick z konfigurowalnymi parametrami"""
    
    def __init__(self, fs=48000):
        self.fs = fs
        
        # Detection parameters
        self.amplitude_threshold = 0.7
        self.derivative_threshold = 0.3
        self.use_amplitude_detect = True
        self.use_derivative_detect = True
        self.use_statistical_detect = False
        
        # Repair parameters
        self.mask_expansion = 5
        self.interpolation_method = 'cubic'  # linear, cubic
        
        # Post-processing
        self.apply_deblur = True
        self.deblur_amount = 0.3
        self.deblur_sigma = 1.5
        self.apply_limiting = True
        self.limit_threshold = 0.95
        
        # Statistics
        self.defects_detected = 0
        self.samples_repaired = 0
    
    def detect_amplitude_clicks(self, audio):
        """Wykrywa amplitude spikes"""
        peaks, _ = find_peaks(
            np.abs(audio),
            height=self.amplitude_threshold,
            distance=5
        )
        return peaks
    
    def detect_derivative_clicks(self, audio):
        """Wykrywa nagłe skoki (derivative analysis)"""
        diff = np.diff(audio, prepend=audio[0])
        diff2 = np.diff(diff, prepend=diff[0])
        abs_diff2 = np.abs(diff2)
        
        # ODWRÓCONA LOGIKA: wyższa wartość threshold = NIŻSZY próg = WIĘCEJ wykrywa
        # threshold=2.0 → wykrywa prawie wszystko
        # threshold=0.05 → wykrywa tylko ekstremalne skoki
        percentile_base = 99
        sensitivity_factor = 1.0 / self.derivative_threshold  # Odwrócenie!
        threshold_val = sensitivity_factor * np.percentile(abs_diff2, percentile_base)
        
        spikes = np.where(abs_diff2 > threshold_val)[0]
        return spikes
    
    def detect_statistical_outliers(self, audio, window=100, threshold=3.5):
        """Z-score based outlier detection"""
        outliers = []
        for i in range(len(audio)):
            start = max(0, i - window)
            end = min(len(audio), i + window)
            window_data = audio[start:end]
            
            mean = np.mean(window_data)
            std = np.std(window_data)
            
            if std > 0:
                z = abs((audio[i] - mean) / std)
                if z > threshold:
                    outliers.append(i)
        
        return np.array(outliers)
    
    def detect_all(self, audio):
        """Kombinuje wszystkie metody detekcji"""
        defects = []
        
        if self.use_amplitude_detect:
            amp_clicks = self.detect_amplitude_clicks(audio)
            defects.extend(amp_clicks)
        
        if self.use_derivative_detect:
            der_clicks = self.detect_derivative_clicks(audio)
            defects.extend(der_clicks)
        
        if self.use_statistical_detect:
            stat_clicks = self.detect_statistical_outliers(audio)
            defects.extend(stat_clicks)
        
        return np.unique(defects)
    
    def create_mask(self, defects, length):
        """Tworzy binary mask z expansion"""
        mask = np.zeros(length, dtype=bool)
        for idx in defects:
            start = max(0, idx - self.mask_expansion)
            end = min(length, idx + self.mask_expansion + 1)
            mask[start:end] = True
        return mask
    
    def interpolate(self, audio, mask):
        """Interpolacja - cubic lub linear"""
        result = audio.copy()
        good_idx = np.where(~mask)[0]
        bad_idx = np.where(mask)[0]
        
        if len(good_idx) < 2 or len(bad_idx) == 0:
            return result
        
        if self.interpolation_method == 'cubic' and len(good_idx) >= 4:
            try:
                interp = PchipInterpolator(good_idx, audio[good_idx])
                result[bad_idx] = interp(bad_idx)
            except:
                # Fallback
                interp = interp1d(good_idx, audio[good_idx], 
                                kind='linear', fill_value='extrapolate')
                result[bad_idx] = interp(bad_idx)
        else:
            # Linear
            interp = interp1d(good_idx, audio[good_idx],
                            kind='linear', fill_value='extrapolate')
            result[bad_idx] = interp(bad_idx)
        
        return result
    
    def deblur(self, audio):
        """Unsharp mask deblur"""
        blurred = gaussian_filter1d(audio, sigma=self.deblur_sigma)
        detail = audio - blurred
        return audio + self.deblur_amount * detail
    
    def soft_limit(self, audio):
        """Soft limiting"""
        result = audio.copy()
        mask = np.abs(audio) > self.limit_threshold
        if np.any(mask):
            scale = 1.0 / self.limit_threshold
            result[mask] = np.tanh(audio[mask] * scale) / scale
        return result
    
    def process(self, audio):
        """Main processing pipeline"""
        # To mono
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
            is_stereo = True
        else:
            mono = audio
            is_stereo = False
        
        # 1. Detect
        defects = self.detect_all(mono)
        self.defects_detected = len(defects)
        
        if self.defects_detected == 0:
            return audio
        
        # 2. Mask
        mask = self.create_mask(defects, len(mono))
        self.samples_repaired = np.sum(mask)
        
        # 3. Interpolate
        repaired = self.interpolate(mono, mask)
        
        # 4. Deblur
        if self.apply_deblur:
            repaired = self.deblur(repaired)
        
        # 5. Limit
        if self.apply_limiting:
            repaired = self.soft_limit(repaired)
        
        # To stereo
        if is_stereo:
            result = np.column_stack([repaired, repaired])
        else:
            result = repaired
        
        return result

# ==========================================
# PROCESSOR
# ==========================================

class DeclickStreamProcessor:
    """Real-time stream processor"""
    
    def __init__(self):
        self.declicker = AdvancedDeclicker(fs=48000)
        self.stream = None
        self.active = False
        self.status_callback = None
        self.fs = 48000
        self.bs = 2048
        
        # Stats
        self.total_blocks = 0
        self.total_defects = 0
    
    def audio_callback(self, indata, outdata, frames, time, status):
        """Processing loop"""
        if status:
            pass
        
        # VU meter
        volume = np.linalg.norm(indata) * 10
        if volume > 0.2:
            bars = int(min(volume, 40))
            sys.stdout.write(f"\r🔊 [{'█' * bars:<40}] {volume:.1f} dB")
            sys.stdout.flush()
        
        # Process
        try:
            processed = self.declicker.process(indata)
            
            # Stats
            self.total_blocks += 1
            self.total_defects += self.declicker.defects_detected
            
            # Display detection
            if self.declicker.defects_detected > 0:
                sys.stdout.write(f"\r✂️ Detected: {self.declicker.defects_detected} clicks, "
                               f"Repaired: {self.declicker.samples_repaired} samples")
                sys.stdout.flush()
            
            n = min(len(processed), len(outdata))
            outdata[:n] = processed[:n]
            if n < len(outdata):
                outdata[n:] = 0
                
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            outdata[:] = indata
    
    def start(self):
        if self.active: return
        
        try:
            self.stream = sd.Stream(
                samplerate=self.fs,
                blocksize=self.bs,
                channels=2,
                dtype='float32',
                latency='high',
                callback=self.audio_callback
            )
            self.stream.start()
            self.active = True
            
            if self.status_callback:
                GLib.idle_add(self.status_callback, "✅ Processing...")
            
            print("\n✅ Declick processor started!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.active = False
    
    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except: pass
            self.stream = None
        
        self.active = False
        
        if self.status_callback:
            GLib.idle_add(self.status_callback, "⏸️ Stopped")
        
        print(f"\n⏹️ Stopped. Total: {self.total_defects} defects detected")

# ==========================================
# GUI
# ==========================================

class DeclickStudioApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="🎚️ Audio Declick & Inpainting Studio")
        self.set_default_size(750, 650)
        self.set_border_width(15)
        
        self.processor = DeclickStreamProcessor()
        self.processor.status_callback = self.update_status
        
        # Scrollable
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.add(scrolled)
        
        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        scrolled.add(main_vbox)
        
        # Header
        header = Gtk.Label()
        header.set_markup(
            "<span size='xx-large'><b>🎚️ Audio Declick Studio</b></span>\n"
            "<span size='small'>Professional Click & Crackle Removal</span>"
        )
        header.set_justify(Gtk.Justification.CENTER)
        main_vbox.pack_start(header, False, False, 0)
        
        # Instructions
        info_frame = Gtk.Frame(label="📋 Setup")
        info_label = Gtk.Label()
        info_label.set_markup(
            "<b>1.</b> Play audio (vinyl, tape, etc.)\n"
            "<b>2.</b> Pavucontrol → Playback → 'DeclickStudio'\n"
            "<b>3.</b> Pavucontrol → Recording → 'Monitor of DeclickStudio'\n"
            "<b>4.</b> Configure parameters below\n"
            "<b>5.</b> Toggle Master ON"
        )
        info_label.set_padding(10, 10)
        info_label.set_line_wrap(True)
        info_frame.add(info_label)
        main_vbox.pack_start(info_frame, False, False, 0)
        
        # Detection parameters
        detect_frame = Gtk.Frame(label="🔍 Detection Parameters")
        detect_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        detect_box.set_border_width(10)
        detect_frame.add(detect_box)
        
        # Checkboxes for detection methods
        self.amp_check = Gtk.CheckButton(label="Amplitude Peak Detection")
        self.amp_check.set_active(True)
        self.amp_check.connect("toggled", self.on_param_changed)
        detect_box.pack_start(self.amp_check, False, False, 0)
        
        # Amplitude threshold
        amp_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        amp_box.pack_start(Gtk.Label(label="  Threshold:"), False, False, 0)
        self.amp_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.3, 0.95, 0.05)
        self.amp_scale.set_value(0.7)
        self.amp_scale.connect("value-changed", self.on_param_changed)
        amp_box.pack_start(self.amp_scale, True, True, 0)
        self.amp_value_label = Gtk.Label(label="0.70")
        amp_box.pack_start(self.amp_value_label, False, False, 0)
        detect_box.pack_start(amp_box, False, False, 0)
        
        self.deriv_check = Gtk.CheckButton(label="Derivative Analysis (sudden jumps) - wyżej = WIĘCEJ wykrywa")
        self.deriv_check.set_active(True)
        self.deriv_check.connect("toggled", self.on_param_changed)
        detect_box.pack_start(self.deriv_check, False, False, 0)
        
        # Derivative threshold
        deriv_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        deriv_box.pack_start(Gtk.Label(label="  Sensitivity:"), False, False, 0)
        self.deriv_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.05, 2.0, 0.05)
        self.deriv_scale.set_value(0.3)
        self.deriv_scale.connect("value-changed", self.on_param_changed)
        deriv_box.pack_start(self.deriv_scale, True, True, 0)
        self.deriv_value_label = Gtk.Label(label="0.30")
        deriv_box.pack_start(self.deriv_value_label, False, False, 0)
        detect_box.pack_start(deriv_box, False, False, 0)
        
        self.stat_check = Gtk.CheckButton(label="Statistical Outlier Detection (Z-score)")
        self.stat_check.set_active(False)
        self.stat_check.connect("toggled", self.on_param_changed)
        detect_box.pack_start(self.stat_check, False, False, 0)
        
        main_vbox.pack_start(detect_frame, False, False, 0)
        
        # Repair parameters
        repair_frame = Gtk.Frame(label="🔧 Repair Parameters")
        repair_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        repair_box.set_border_width(10)
        repair_frame.add(repair_box)
        
        # Mask expansion
        exp_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        exp_box.pack_start(Gtk.Label(label="Mask Expansion:"), False, False, 0)
        self.expansion_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1, 20, 1)
        self.expansion_scale.set_value(5)
        self.expansion_scale.connect("value-changed", self.on_param_changed)
        exp_box.pack_start(self.expansion_scale, True, True, 0)
        self.expansion_label = Gtk.Label(label="5 samples")
        exp_box.pack_start(self.expansion_label, False, False, 0)
        repair_box.pack_start(exp_box, False, False, 0)
        
        # Interpolation method
        interp_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        interp_box.pack_start(Gtk.Label(label="Interpolation:"), False, False, 0)
        self.interp_combo = Gtk.ComboBoxText()
        self.interp_combo.append_text("Cubic (PCHIP)")
        self.interp_combo.append_text("Linear")
        self.interp_combo.set_active(0)
        self.interp_combo.connect("changed", self.on_param_changed)
        interp_box.pack_start(self.interp_combo, False, False, 0)
        repair_box.pack_start(interp_box, False, False, 0)
        
        main_vbox.pack_start(repair_frame, False, False, 0)
        
        # Post-processing
        post_frame = Gtk.Frame(label="✨ Post-Processing")
        post_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        post_box.set_border_width(10)
        post_frame.add(post_box)
        
        self.deblur_check = Gtk.CheckButton(label="Apply Deblur (Unsharp Mask)")
        self.deblur_check.set_active(True)
        self.deblur_check.connect("toggled", self.on_param_changed)
        post_box.pack_start(self.deblur_check, False, False, 0)
        
        # Deblur amount
        deblur_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        deblur_box.pack_start(Gtk.Label(label="  Amount:"), False, False, 0)
        self.deblur_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 1.0, 0.1)
        self.deblur_scale.set_value(0.3)
        self.deblur_scale.connect("value-changed", self.on_param_changed)
        deblur_box.pack_start(self.deblur_scale, True, True, 0)
        self.deblur_label = Gtk.Label(label="0.3")
        deblur_box.pack_start(self.deblur_label, False, False, 0)
        post_box.pack_start(deblur_box, False, False, 0)
        
        self.limit_check = Gtk.CheckButton(label="Soft Limiting (anti-clip)")
        self.limit_check.set_active(True)
        self.limit_check.connect("toggled", self.on_param_changed)
        post_box.pack_start(self.limit_check, False, False, 0)
        
        main_vbox.pack_start(post_frame, False, False, 0)
        
        # Presets
        preset_frame = Gtk.Frame(label="⚡ Quick Presets")
        preset_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        preset_box.set_border_width(10)
        preset_frame.add(preset_box)
        
        presets = [
            ("💿 Vinyl", {'amp': 0.6, 'deriv': 0.8, 'expansion': 7}),  # Wyższa sensitivity
            ("📼 Tape", {'amp': 0.7, 'deriv': 0.5, 'expansion': 5}),
            ("🎤 Aggressive", {'amp': 0.5, 'deriv': 1.5, 'expansion': 10}),  # Bardzo wysoka!
            ("🎵 Gentle", {'amp': 0.8, 'deriv': 0.2, 'expansion': 3})  # Niska
        ]
        
        for name, params in presets:
            btn = Gtk.Button(label=name)
            btn.connect("clicked", self.on_preset_clicked, params)
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
        
        # Apply initial parameters
        self.on_param_changed(None)
    
    def on_param_changed(self, widget):
        """Update processor parameters"""
        dec = self.processor.declicker
        
        # Detection
        dec.use_amplitude_detect = self.amp_check.get_active()
        amp_val = self.amp_scale.get_value()
        dec.amplitude_threshold = amp_val
        self.amp_value_label.set_text(f"{amp_val:.2f}")
        
        dec.use_derivative_detect = self.deriv_check.get_active()
        deriv_val = self.deriv_scale.get_value()
        dec.derivative_threshold = deriv_val
        self.deriv_value_label.set_text(f"{deriv_val:.2f}")
        
        dec.use_statistical_detect = self.stat_check.get_active()
        
        # Repair
        exp_val = int(self.expansion_scale.get_value())
        dec.mask_expansion = exp_val
        self.expansion_label.set_text(f"{exp_val} samples")
        
        interp_idx = self.interp_combo.get_active()
        dec.interpolation_method = 'cubic' if interp_idx == 0 else 'linear'
        
        # Post
        dec.apply_deblur = self.deblur_check.get_active()
        deblur_val = self.deblur_scale.get_value()
        dec.deblur_amount = deblur_val
        self.deblur_label.set_text(f"{deblur_val:.1f}")
        
        dec.apply_limiting = self.limit_check.get_active()
    
    def on_preset_clicked(self, button, params):
        """Apply preset"""
        self.amp_scale.set_value(params.get('amp', 0.7))
        self.deriv_scale.set_value(params.get('deriv', 0.3))
        self.expansion_scale.set_value(params.get('expansion', 5))
        self.on_param_changed(None)
    
    def on_master_toggled(self, switch, _):
        if switch.get_active():
            self.processor.start()
        else:
            self.processor.stop()
    
    def update_status(self, msg):
        self.status_label.set_markup(f"<b>{msg}</b>")

if __name__ == "__main__":
    print("🎚️ Audio Declick & Inpainting Studio")
    print("=" * 50)
    create_virtual_sink()
    
    app = DeclickStudioApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    
    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
