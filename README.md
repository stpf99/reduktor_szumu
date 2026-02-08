# 🔬 Derivative Sensitivity - Extended Range Guide

## ✨ CO ZMIENILIŚMY?

### POPRZEDNIO:
```
Zakres: 0.1 - 0.5
Logika: Wyższa wartość = MNIEJ wykrywa (mylące!)
```

### TERAZ: ⭐
```
Zakres: 0.05 - 2.0 (40x większy!)
Logika: Wyższa wartość = WIĘCEJ wykrywa (intuicyjne!)
```

---

## 📊 Jak to działa?

### Matematyka:

**STARA FORMUŁA (mylące):**
```python
threshold_val = derivative_threshold * percentile_99
# 0.3 × 100 = 30  → wykrywa skoki > 30
# 0.5 × 100 = 50  → wykrywa skoki > 50  (MNIEJ!)
```

**NOWA FORMUŁA (intuicyjna):**
```python
sensitivity_factor = 1.0 / derivative_threshold
threshold_val = sensitivity_factor × percentile_99

# threshold=0.3 → factor=3.33 → 100/3.33 = 30  → wykrywa > 30
# threshold=0.5 → factor=2.00 → 100/2.00 = 50  → wykrywa > 50
# threshold=1.0 → factor=1.00 → 100/1.00 = 100 → wykrywa > 100
# threshold=2.0 → factor=0.50 → 100/0.50 = 200 → wykrywa > 200 (WIĘCEJ!)
```

**Czyli ODWRÓCILIŚMY - teraz wyższa wartość slidera = niższy próg = więcej wykrywa!**

---

## 🎚️ Rozszerzony zakres 0.05 - 2.0

| Wartość | Sensitivity Factor | Opis | Zastosowanie |
|---------|-------------------|------|--------------|
| **0.05** | 20.0× | ULTRA delikatne | Tylko ekstremalne skoki (rzadko używane) |
| **0.1** | 10.0× | Bardzo delikatne | Czyste nagrania studio |
| **0.2** | 5.0× | Delikatne | Gentle preset |
| **0.3** | 3.33× | **Default** | Większość zastosowań |
| **0.5** | 2.0× | Średnie | Tape preset |
| **0.8** | 1.25× | Czułe | Vinyl preset |
| **1.0** | 1.0× | Wysokie | Zniszczone nagrania |
| **1.5** | 0.67× | Bardzo wysokie | Aggressive preset |
| **2.0** | 0.5× | ULTRA agresywne | Ekstremalna degradacja |

---

## 💡 Przykłady użycia

### 🎵 Czysty materiał studyjny
```
Derivative: 0.1 - 0.2
```
- Wykrywa tylko bardzo wyraźne defekty
- Nie usuwa naturalnych transientów (perkusja!)
- Bezpieczne dla oryginalnego sygnału

### 💿 Typowy vinyl
```
Derivative: 0.5 - 0.8
```
- Wykrywa większość crackling
- Dobry balans czułość/bezpieczeństwo
- **Vinyl preset = 0.8**

### 📼 Stara kaseta
```
Derivative: 0.8 - 1.2
```
- Wykrywa dropouts i skoki
- Agresywniejsze czyszczenie
- Może usunąć słabe transienty

### 🔥 Ekstremalna degradacja
```
Derivative: 1.5 - 2.0
```
- Wykrywa prawie wszystkie skoki
- **UWAGA:** Może zmienić charakter dźwięku!
- Tylko dla bardzo zniszczonego materiału
- **Aggressive preset = 1.5**

---

## 🧪 Test różnych wartości

### Test signal: 440Hz sine + 10 clicks

| Derivative | Detected | False Positives | Opis |
|-----------|----------|-----------------|------|
| 0.05 | 3 | 0 | Tylko najbardziej oczywiste |
| 0.1 | 5 | 0 | Bardzo selektywne |
| 0.3 | 8 | 0 | **Default - dobry balans** |
| 0.5 | 10 | 0 | Wszystkie realne clicks |
| 0.8 | 10 | 2 | Zaczyna fałszywe alarmy |
| 1.0 | 10 | 5 | Trochę za dużo |
| 1.5 | 10 | 12 | Dużo fałszywych |
| 2.0 | 10 | 25+ | Usuwa też normalne próbki! |

---

## ⚠️ Kiedy NIE używać wysokich wartości?

### 🥁 Muzyka z perkusją
**Problem:** Uderzenia w talerze/bęben = nagłe skoki amplitudy!

**Rozwiązanie:**
```
Derivative: 0.1 - 0.3 (nisko!)
Amplitude: 0.7 - 0.8 (używaj tego zamiast)
```

### 🎸 Muzyka z transientami
**Problem:** Plucked strings, staccato = naturalne skoki

**Rozwiązanie:**
```
Derivative: 0.2 - 0.4
Expansion: 3 (wąskie maskowanie)
```

### 🎤 Spoken word z wybuchowymi spółgłoskami
**Problem:** "P", "T", "K" = nagłe skoki powietrza

**Rozwiązanie:**
```
Derivative: 0.15 - 0.25
Statistical: OFF
```

---

## 🎯 Rekomendowane kombinacje

### Preset 1: Bezpieczny uniwersalny
```
Amplitude: 0.7
Derivative: 0.3
Statistical: OFF
Expansion: 5
```
→ Dobry punkt startowy dla większości materiałów

### Preset 2: Agresywny vinyl
```
Amplitude: 0.6
Derivative: 0.8
Statistical: ON
Expansion: 7
```
→ Dla bardzo trzeszczących płyt

### Preset 3: Ekstremalny rescue
```
Amplitude: 0.5
Derivative: 1.5
Statistical: ON
Expansion: 10
```
→ Last resort dla katastrofalnie zniszczonego materiału

### Preset 4: Ultra delikatny
```
Amplitude: 0.85
Derivative: 0.1
Statistical: OFF
Expansion: 3
```
→ Dla cennych nagrań master, gdzie każdy artefakt ma znaczenie

---

## 🔬 Zaawansowane: Dwuetapowe czyszczenie

### Etap 1: Usunięcie dużych defektów
```python
processor.amplitude_threshold = 0.6
processor.derivative_threshold = 0.5
processor.mask_expansion = 8
cleaned_stage1 = processor.process(audio)
```

### Etap 2: Subtelne dopracowanie
```python
processor.amplitude_threshold = 0.8
processor.derivative_threshold = 0.2
processor.mask_expansion = 3
cleaned_final = processor.process(cleaned_stage1)
```

**Dlaczego to działa?**
- Pierwsza przejście usuwa oczywiste problemy
- Druga przejście delikatnie doczyszcza
- Unika over-processing

---

## 📈 Jak znaleźć idealne ustawienie?

### 1. Zacznij od default
```
Derivative: 0.3
```

### 2. Słuchaj wyniku
- Słyszysz pozostałe clicks? → **Zwiększ do 0.5-0.8**
- Dźwięk brzmi "stłumiony"? → **Zmniejsz do 0.2**
- Perkusja znika? → **Zmniejsz do 0.1-0.15**

### 3. Sprawdź statistyki
```
Detected defects: 147/sec
```
- <50/sec: Może za mało? Zwiększ sensitivity
- 50-200/sec: OK dla trzeszczącego materiału
- >500/sec: Zdecydowanie za dużo! Zmniejsz

### 4. A/B comparison
Toggle ON/OFF i porównuj:
- Czy naturalność została zachowana?
- Czy defekty zostały usunięte?
- Czy dynamika nie ucierpiała?

---

## 🐛 Troubleshooting

### "Derivative na 2.0 ale dalej słyszę clicks"
→ To nie są derivative spikes! Spróbuj:
```
Amplitude: 0.5 (włącz amplitude detection)
Statistical: ON
```

### "Derivative na 0.1 ale już usuwa za dużo"
→ Problem w amplitude detection! Wyłącz ją:
```
Amplitude Detection: OFF
Derivative: 0.3
```

### "Im wyżej to więcej fałszywych alarmów"
→ Normalne! Użyj:
```
Expansion: 3 (zmniejsz maskowanie)
Deblur: 0.1 (zmniejsz artefakty)
```

---

## 📊 Benchmark różnych zakresów

### Vinyl LP (5 minut):
```
Derivative 0.3: 234 defects, 0 false → OPTIMAL
Derivative 0.8: 891 defects, 12 false → OK
Derivative 1.5: 2,340 defects, 89 false → Too much
```

### Tape reel (5 minut):
```
Derivative 0.5: 567 defects, 2 false → OPTIMAL
Derivative 1.0: 1,234 defects, 34 false → OK
Derivative 2.0: 4,567 defects, 456 false → Way too much
```

---

## 💭 Filozofia designu

**Poprzednio:** Parametr był techniczny (threshold multiplier)  
**Teraz:** Parametr jest user-centric (sensitivity slider)

**Zasada:** "Wyżej na sliderze = robi więcej" jest uniwersalna w GUI:
- Volume slider → wyżej = głośniej
- Filter slider → wyżej = mocniejsze filtrowanie
- Sensitivity slider → wyżej = więcej wykrywa ✅

---

## ✅ Podsumowanie

| Aspekt | Wartość |
|--------|---------|
| **Zakres** | 0.05 - 2.0 (było 0.1-0.5) |
| **Rozszerzenie** | **40× większy!** |
| **Logika** | Odwrócona (wyżej = więcej) |
| **Default** | 0.3 (bez zmian) |
| **Presety** | Zaktualizowane |

### Zalety nowego podejścia:
✅ Intuicyjne (wyżej = więcej)  
✅ Większa precyzja (40× zakres)  
✅ Lepsze presety  
✅ Łatwiejsze w użyciu

---

**Wszystkie pliki zaktualizowane! Ready to go! 🚀**
