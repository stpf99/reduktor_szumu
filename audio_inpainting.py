#!/usr/bin/env python3
"""
🎚️ ADVANCED AUDIO INPAINTING & DECLICKING
==========================================
Algorytm:
1. Wykryj peaks (trzaski, crackling, iskry) 
2. Zamaskuj defekty (zerowanie/blur)
3. Interpoluj brakujące próbki (cubic spline, PCHIP)
4. Deblur i rekonstrukcja szczegółów
5. Normalizacja i soft-limiting

Metody wykrywania:
- Peak detection (amplitude threshold)
- Derivative analysis (nagłe skoki)
- Spectral outliers (频率 anomalie)
- Statistical outliers (Z-score, MAD)

Metody naprawy:
- Linear/Cubic interpolation
- Spectral interpolation
- AR prediction
- Neural inpainting (opcjonalne)
"""

import numpy as np
from scipy.signal import find_peaks, medfilt, resample
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.fft import rfft, rfftfreq, irfft
import sys

# ==========================================
# 🔍 DETECTION ALGORITHMS
# ==========================================

class ClickDetector:
    """Wykrywa trzaski i crackling w audio"""
    
    def __init__(self, fs=48000):
        self.fs = fs
        
    def detect_amplitude_peaks(self, audio, threshold=0.8, min_distance=10):
        """
        Wykrywa peaks przekraczające próg amplitudy
        
        Args:
            threshold: Próg (0-1), gdzie 1.0 = full scale
            min_distance: Minimalna odległość między peaks (próbki)
        """
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Znajdź peaks w wartości bezwzględnej
        peaks, properties = find_peaks(
            np.abs(audio), 
            height=threshold,
            distance=min_distance
        )
        
        return peaks, properties
    
    def detect_derivative_spikes(self, audio, threshold=0.3):
        """
        Wykrywa nagłe skoki przez analizę pochodnej
        Trzaski = nagłe zmiany amplitudy
        
        threshold: 0.05-2.0, wyżej = WIĘCEJ wykrywa (odwrócona logika dla UX)
        """
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # Pierwsza pochodna (różnica między próbkami)
        diff = np.diff(audio, prepend=audio[0])
        
        # Druga pochodna (przyspieszenie)
        diff2 = np.diff(diff, prepend=diff[0])
        
        # Wykryj gdzie druga pochodna jest duża
        abs_diff2 = np.abs(diff2)
        
        # ODWRÓCONA LOGIKA dla lepszego UX
        sensitivity_factor = 1.0 / max(threshold, 0.05)  # Ochrona przed dzieleniem przez 0
        threshold_val = sensitivity_factor * np.percentile(abs_diff2, 99)
        
        spikes = np.where(abs_diff2 > threshold_val)[0]
        
        return spikes
    
    def detect_spectral_outliers(self, audio, window_size=512, threshold=3.0):
        """
        Wykrywa anomalie w dziedzinie częstotliwości
        """
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        outliers = []
        
        # Przetwarzaj w oknach
        hop = window_size // 2
        for i in range(0, len(audio) - window_size, hop):
            window = audio[i:i+window_size]
            
            # FFT
            spectrum = np.abs(rfft(window))
            
            # Sprawdź czy są nietypowe peaks w spektrum
            mean_energy = np.mean(spectrum)
            max_energy = np.max(spectrum)
            
            if max_energy > threshold * mean_energy:
                # Znaleziono anomalię
                outliers.extend(range(i, i+window_size))
        
        return np.unique(outliers)
    
    def detect_statistical_outliers(self, audio, window_size=100, threshold=3.5):
        """
        Wykrywa outliers używając Z-score w przesuwnym oknie
        """
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        outliers = []
        
        for i in range(len(audio)):
            # Weź okno wokół próbki
            start = max(0, i - window_size)
            end = min(len(audio), i + window_size)
            window = audio[start:end]
            
            # Oblicz Z-score
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                z_score = abs((audio[i] - mean) / std)
                if z_score > threshold:
                    outliers.append(i)
        
        return np.array(outliers)
    
    def detect_combined(self, audio, methods=['amplitude', 'derivative']):
        """
        Łączy wyniki z wielu metod
        """
        all_defects = []
        
        if 'amplitude' in methods:
            peaks, _ = self.detect_amplitude_peaks(audio)
            all_defects.extend(peaks)
        
        if 'derivative' in methods:
            spikes = self.detect_derivative_spikes(audio)
            all_defects.extend(spikes)
        
        if 'spectral' in methods:
            spectral = self.detect_spectral_outliers(audio)
            all_defects.extend(spectral)
        
        if 'statistical' in methods:
            statistical = self.detect_statistical_outliers(audio)
            all_defects.extend(statistical)
        
        # Usuń duplikaty i posortuj
        defects = np.unique(all_defects)
        
        return defects

# ==========================================
# 🔧 REPAIR ALGORITHMS
# ==========================================

class AudioInpainter:
    """Naprawia wykryte defekty przez inpainting"""
    
    def __init__(self):
        pass
    
    def expand_mask(self, defects, audio_length, expansion=5):
        """
        Rozszerza maskę defektów o sąsiednie próbki
        (trzaski często wpływają na sąsiedztwo)
        """
        mask = np.zeros(audio_length, dtype=bool)
        
        for idx in defects:
            start = max(0, idx - expansion)
            end = min(audio_length, idx + expansion + 1)
            mask[start:end] = True
        
        return mask
    
    def linear_interpolation(self, audio, mask):
        """Prosta interpolacja liniowa"""
        result = audio.copy()
        
        # Indeksy dobrych próbek
        good_indices = np.where(~mask)[0]
        bad_indices = np.where(mask)[0]
        
        if len(good_indices) < 2 or len(bad_indices) == 0:
            return result
        
        # Interpoluj
        interpolator = interp1d(
            good_indices, 
            audio[good_indices],
            kind='linear',
            fill_value='extrapolate'
        )
        
        result[bad_indices] = interpolator(bad_indices)
        
        return result
    
    def cubic_interpolation(self, audio, mask):
        """Interpolacja kubiczna (smooth)"""
        result = audio.copy()
        
        good_indices = np.where(~mask)[0]
        bad_indices = np.where(mask)[0]
        
        if len(good_indices) < 4 or len(bad_indices) == 0:
            return self.linear_interpolation(audio, mask)
        
        # PCHIP = Piecewise Cubic Hermite Interpolating Polynomial
        # Zachowuje monotoniczność, unika overshooting
        interpolator = PchipInterpolator(
            good_indices,
            audio[good_indices]
        )
        
        result[bad_indices] = interpolator(bad_indices)
        
        return result
    
    def spectral_interpolation(self, audio, mask, overlap=0.5):
        """
        Interpolacja w dziedzinie częstotliwości
        Lepsze dla dłuższych luk
        """
        result = audio.copy()
        
        # FFT całego sygnału
        spectrum = rfft(audio)
        frequencies = rfftfreq(len(audio))
        
        # Dla każdej luki
        mask_indices = np.where(mask)[0]
        if len(mask_indices) == 0:
            return result
        
        # Grupuj sąsiednie defekty
        gaps = []
        gap_start = mask_indices[0]
        
        for i in range(1, len(mask_indices)):
            if mask_indices[i] != mask_indices[i-1] + 1:
                # Koniec luki
                gaps.append((gap_start, mask_indices[i-1]))
                gap_start = mask_indices[i]
        gaps.append((gap_start, mask_indices[-1]))
        
        # Napraw każdą lukę
        for start, end in gaps:
            gap_len = end - start + 1
            
            # Weź kontekst przed i po
            context_len = gap_len * 2
            pre_start = max(0, start - context_len)
            post_end = min(len(audio), end + context_len)
            
            # FFT kontekstu
            if start > 0 and end < len(audio) - 1:
                context = np.concatenate([
                    audio[pre_start:start],
                    audio[end+1:post_end]
                ])
                
                if len(context) > gap_len:
                    # Użyj spektrum kontekstu do wypełnienia luki
                    context_spectrum = rfft(context)
                    
                    # Generuj sygnał o podobnym spektrum
                    phase = np.angle(context_spectrum[:gap_len//2+1])
                    magnitude = np.abs(context_spectrum[:gap_len//2+1])
                    
                    reconstructed_spectrum = magnitude * np.exp(1j * phase)
                    reconstructed = irfft(reconstructed_spectrum, n=gap_len)
                    
                    # Smooth blend
                    blend_len = min(10, gap_len // 4)
                    if start > 0:
                        for i in range(blend_len):
                            alpha = i / blend_len
                            result[start + i] = (1 - alpha) * audio[start-1] + alpha * reconstructed[i]
                        result[start+blend_len:end-blend_len+1] = reconstructed[blend_len:gap_len-blend_len]
                    
                    if end < len(audio) - 1:
                        for i in range(blend_len):
                            alpha = i / blend_len
                            result[end - blend_len + i] = (1 - alpha) * reconstructed[gap_len-blend_len+i] + alpha * audio[end+1]
        
        return result
    
    def ar_prediction(self, audio, mask, order=10):
        """
        Auto-Regressive prediction
        Przewiduje brakujące próbki na podstawie historii
        """
        result = audio.copy()
        
        # Dla każdej luki, użyj AR do przewidzenia
        mask_indices = np.where(mask)[0]
        
        if len(mask_indices) == 0:
            return result
        
        # Grupuj luki
        gaps = []
        gap_start = mask_indices[0]
        
        for i in range(1, len(mask_indices)):
            if mask_indices[i] != mask_indices[i-1] + 1:
                gaps.append((gap_start, mask_indices[i-1]))
                gap_start = mask_indices[i]
        gaps.append((gap_start, mask_indices[-1]))
        
        # Napraw każdą lukę
        for start, end in gaps:
            gap_len = end - start + 1
            
            # Potrzebujemy kontekstu przed luką
            if start >= order:
                # Weź próbki przed luką
                context = audio[start-order:start]
                
                # Prosty AR(p): x[n] = sum(a[i] * x[n-i])
                # Oszacuj współczynniki z kontekstu
                
                # Wypełnij lukę predykcją
                for i in range(gap_len):
                    if start + i >= order:
                        # Użyj ostatnich `order` próbek
                        recent = result[start+i-order:start+i]
                        # Prosta predykcja = średnia ważona
                        weights = np.linspace(1, 0.5, order)
                        weights /= weights.sum()
                        predicted = np.dot(recent, weights)
                        result[start + i] = predicted
        
        return result
    
    def adaptive_filter(self, audio, mask, filter_length=21):
        """
        Adaptacyjny filtr Wienera dla inpainting
        """
        result = audio.copy()
        
        # Dla zamaskowanych regionów, użyj Wiener filter
        from scipy.signal import wiener
        
        # Przetwórz cały sygnał
        filtered = wiener(audio, mysize=filter_length)
        
        # Użyj filtered tylko dla zamaskowanych próbek
        result[mask] = filtered[mask]
        
        # Smooth transition
        for idx in np.where(mask)[0]:
            if 0 < idx < len(audio) - 1:
                # Blend z sąsiadami
                if not mask[idx-1] and not mask[idx+1]:
                    result[idx] = 0.25 * audio[idx-1] + 0.5 * filtered[idx] + 0.25 * audio[idx+1]
        
        return result

# ==========================================
# 🎨 POST-PROCESSING
# ==========================================

class AudioRestoration:
    """Post-processing: deblur, denoise, normalizacja"""
    
    def __init__(self):
        pass
    
    def deblur_wiener(self, audio, noise_power=0.01):
        """Odwraca blur używając deconvolution Wienera"""
        from scipy.signal import wiener
        return wiener(audio, mysize=5, noise=noise_power)
    
    def deblur_unsharp_mask(self, audio, amount=0.5, sigma=2.0):
        """
        Unsharp masking - zwiększa kontrast/ostrość
        result = original + amount * (original - blurred)
        """
        # Blur
        blurred = gaussian_filter1d(audio, sigma=sigma)
        
        # Detail = original - blurred
        detail = audio - blurred
        
        # Dodaj detail z powrotem
        sharpened = audio + amount * detail
        
        return sharpened
    
    def enhance_detail(self, audio, window_size=21):
        """Zwiększa szczegóły przez high-pass filtering"""
        from scipy.signal import butter, lfilter
        
        # High-pass filter
        nyq = 0.5 * 48000  # Zakładam 48kHz
        cutoff = 100  # Hz
        b, a = butter(4, cutoff / nyq, btype='high')
        
        # Ekstrahuj high-frequency content
        high_freq = lfilter(b, a, audio)
        
        # Dodaj z powrotem ze wzmocnieniem
        enhanced = audio + 0.3 * high_freq
        
        return enhanced
    
    def soft_limiting(self, audio, threshold=0.95):
        """
        Soft limiting - zapobiega clipping
        Smooth saturation curve
        """
        result = audio.copy()
        
        # Tanh saturation dla próbek powyżej progu
        mask = np.abs(audio) > threshold
        
        if np.any(mask):
            # Normalizuj do [-1, 1] zakresu dla tanh
            scale = 1.0 / threshold
            result[mask] = np.tanh(audio[mask] * scale) / scale
        
        return result
    
    def normalize(self, audio, target_level=-3.0):
        """
        Normalizuje do target level (dB)
        """
        # Obecny peak level
        peak = np.max(np.abs(audio))
        
        if peak > 0:
            # Target w liniowym
            target_linear = 10 ** (target_level / 20)
            
            # Gain potrzebny
            gain = target_linear / peak
            
            result = audio * gain
        else:
            result = audio
        
        return result

# ==========================================
# 🎛️ MAIN PROCESSOR
# ==========================================

class DeclickProcessor:
    """Główny procesor - pipeline declicking"""
    
    def __init__(self, fs=48000):
        self.fs = fs
        self.detector = ClickDetector(fs=fs)
        self.inpainter = AudioInpainter()
        self.restoration = AudioRestoration()
        
        # Parametry
        self.detection_methods = ['amplitude', 'derivative']
        self.interpolation_method = 'cubic'  # linear, cubic, spectral, ar
        self.mask_expansion = 5
        self.apply_deblur = True
        self.apply_normalization = True
        
    def process(self, audio):
        """
        Kompletny pipeline:
        1. Detect clicks
        2. Mask defects
        3. Interpolate
        4. Deblur
        5. Normalize
        """
        original_shape = audio.shape
        
        # Konwertuj do mono dla przetwarzania
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
            is_stereo = True
        else:
            mono = audio
            is_stereo = False
        
        # 1. DETECT
        defects = self.detector.detect_combined(
            mono, 
            methods=self.detection_methods
        )
        
        num_defects = len(defects)
        
        if num_defects == 0:
            # Brak defektów
            return audio, 0
        
        # 2. EXPAND MASK
        mask = self.inpainter.expand_mask(
            defects, 
            len(mono), 
            expansion=self.mask_expansion
        )
        
        # 3. INTERPOLATE
        if self.interpolation_method == 'linear':
            repaired = self.inpainter.linear_interpolation(mono, mask)
        elif self.interpolation_method == 'cubic':
            repaired = self.inpainter.cubic_interpolation(mono, mask)
        elif self.interpolation_method == 'spectral':
            repaired = self.inpainter.spectral_interpolation(mono, mask)
        elif self.interpolation_method == 'ar':
            repaired = self.inpainter.ar_prediction(mono, mask)
        elif self.interpolation_method == 'adaptive':
            repaired = self.inpainter.adaptive_filter(mono, mask)
        else:
            repaired = self.inpainter.cubic_interpolation(mono, mask)
        
        # 4. DEBLUR (opcjonalne)
        if self.apply_deblur:
            repaired = self.restoration.deblur_unsharp_mask(repaired, amount=0.3, sigma=1.5)
            repaired = self.restoration.enhance_detail(repaired)
        
        # 5. SOFT LIMITING
        repaired = self.restoration.soft_limiting(repaired, threshold=0.95)
        
        # 6. NORMALIZE (opcjonalne)
        if self.apply_normalization:
            repaired = self.restoration.normalize(repaired, target_level=-1.0)
        
        # Konwertuj z powrotem do stereo
        if is_stereo:
            result = np.column_stack([repaired, repaired])
        else:
            result = repaired
        
        return result, num_defects

# ==========================================
# 🧪 TEST
# ==========================================

if __name__ == "__main__":
    print("🎚️ Audio Inpainting & Declicking Test")
    print("=" * 50)
    
    # Generuj test signal z trzaskami
    fs = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Czysty sygnał - sine wave
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
    
    # Dodaj trzaski (impulse noise)
    num_clicks = 20
    click_positions = np.random.randint(1000, len(signal)-1000, num_clicks)
    
    noisy = signal.copy()
    for pos in click_positions:
        # Różne typy trzasków
        if np.random.rand() > 0.5:
            # Spike (pojedynczy peak)
            noisy[pos] = np.random.choice([-1.0, 1.0])
        else:
            # Burst (kilka próbek)
            burst_len = np.random.randint(2, 8)
            noisy[pos:pos+burst_len] = np.random.uniform(-0.8, 0.8, burst_len)
    
    print(f"✓ Wygenerowano sygnał testowy:")
    print(f"  - Częstotliwość: 440 Hz")
    print(f"  - Długość: {duration}s")
    print(f"  - Dodano trzasków: {num_clicks}")
    
    # Przetwórz
    processor = DeclickProcessor(fs=fs)
    processor.detection_methods = ['amplitude', 'derivative', 'statistical']
    processor.interpolation_method = 'cubic'
    
    print(f"\n⚙️ Przetwarzanie...")
    cleaned, defects_found = processor.process(noisy)
    
    print(f"\n✅ Gotowe!")
    print(f"  - Wykryto defektów: {defects_found}")
    print(f"  - Metody detekcji: {processor.detection_methods}")
    print(f"  - Metoda naprawy: {processor.interpolation_method}")
    
    # Oblicz SNR
    noise = noisy - signal
    repaired_noise = cleaned - signal
    
    snr_before = 10 * np.log10(np.sum(signal**2) / np.sum(noise**2))
    snr_after = 10 * np.log10(np.sum(signal**2) / np.sum(repaired_noise**2))
    
    print(f"\n📊 Wyniki:")
    print(f"  - SNR przed: {snr_before:.1f} dB")
    print(f"  - SNR po: {snr_after:.1f} dB")
    print(f"  - Poprawa: {snr_after - snr_before:.1f} dB")
    
    print("\n💡 Parametry do tweakowania:")
    print("  - detection_methods: ['amplitude', 'derivative', 'spectral', 'statistical']")
    print("  - interpolation_method: ['linear', 'cubic', 'spectral', 'ar', 'adaptive']")
    print("  - mask_expansion: 3-10 (próbki wokół defektu)")
    print("  - apply_deblur: True/False")
    print("  - apply_normalization: True/False")
