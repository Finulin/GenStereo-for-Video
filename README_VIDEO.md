# test_with_depth_video.py – Dokumentation

Dieses Skript verarbeitet **Videoframes** und generiert für jeden Frame eine neuartige Stereoskopische Ansicht (Novel View Synthesis). Es nutzt **GenStereo** und **Depth-Anything V2** für die Tiefenschätzung sowie FFmpeg für hochwertige H.264-Videocodierung.

---

## 🎯 Funktionsweise

1. **Video-Eingabe**: Liest ein beliebiges Video Frame-für-Frame
2. **Tiefenschätzung**: Schätzt Tiefe per Frame mittels DAM2 (Depth-Anything V2) oder nutzt externes Depth-Video
3. **Disparity-Berechnung**: Konvertiert Tiefe zu Disparity und skaliert sie
4. **GenStereo NVS**: Generiert eine Zielansicht (rechtes Bild) mittels Diffusion
5. **Fusion**: Kombiniert das warped Bild (Depth-basiert) mit dem GenStereo-Ergebnis
6. **Video-Export**: Speichert 4 MP4-Videos mit H.264-Codec (verlustbehaftet, aber effizienzoptimiert)

---

## 📦 Anforderungen

### Python-Abhängigkeiten
```bash
pip install torch torchvision opencv-python pillow diffusers transformers einops omegaconf jaxtyping tqdm
```

### System-Abhängigkeiten
- **FFmpeg** (optional, aber empfohlen für beste Qualität):
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # Falls nicht verfügbar: Fallback auf OpenCV VideoWriter
  ```

### Modell-Checkpoints
Das Skript benötigt folgende Dateien im `checkpoints/`-Verzeichnis:
- `depth_anything_v2_vitl.pth` (automatisch heruntergeladen, ~1.4 GB)
- GenStereo-Checkpoints (v1.5 oder v2.1):
  - `genstereo-v2.1/reference_unet.pth`
  - `genstereo-v2.1/denoising_unet.pth`
  - `genstereo-v2.1/pose_guider.pth`
  - `genstereo-v2.1/fusion_layer.pth`
  - `genstereo-v2.1/config.json`
  - `sd-vae-ft-mse/` (Verzeichnis)
  - `image_encoder/` (Verzeichnis)

Falls noch nicht heruntergeladen, siehe `scripts/download_models.sh`.

---

## 🚀 Basis-Verwendung

### Einfachster Fall – nur Video-Eingabe:
```bash
python test_with_depth_video.py video.mp4
```
- **Depth-Schätzung**: Automatisch per DAM2 (langsamer, aber benötigt kein zusätzliches Depth-Video)
- **Ausgabe**: `./vis/video/` mit 4 MP4-Videos

### Mit bestehendem Depth-Video:
```bash
python test_with_depth_video.py video.mp4 --depth_video_path depth.mp4
```
- **Depth-Video**: Muss gleich viele Frames wie Input-Video haben
- **Geschwindigkeit**: ~2–3x schneller, da Tiefenschätzung übersprungen wird

### Benutzerdefiniertes Ausgabe-Verzeichnis:
```bash
python test_with_depth_video.py video.mp4 --output ./my_output/
```

---

## 📋 Alle Parameter

### Positionelle Argumente

| Parameter | Typ | Beschreibung |
|-----------|-----|-------------|
| `video_path` | string | **Erforderlich**. Pfad zum Input-Video (MP4, MOV, AVI, etc.) |

### Optionale Argumente

| Flag | Typ | Standard | Beschreibung |
|------|-----|---------|-------------|
| `--depth_video_path` | string | `None` | Pfad zu Depth-Video (grayscale oder BGR). Falls `None`: DAM2 pro Frame. |
| `--output` | string | `./vis` | Ausgabe-Verzeichnis für Videos |
| `--scale_factor` | float | `0.15` | Skalierung der Disparity. Höher = größere Parallaxe. Bereich: 0.05–0.5 |
| `--skip_frames` | int | `1` | Jeden n-ten Frame verarbeiten. `2` = jeden 2. Frame (3x schneller) |
| `--use_ffmpeg` | bool | `True` | FFmpeg für Encoding nutzen. `False` = OpenCV Fallback |
| `--crf` | int | `23` | H.264 Qualität (0–51, niedrig=besser). 18–23 empfohlen. |
| `--preset` | string | `medium` | Encoding-Geschwindigkeit: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow` |

---

## 📊 Parameter-Details

### `--scale_factor`
Kontrolliert die Stärke der erzeugten Stereoskopischen Parallaxe.

- **0.05**: Sehr subtil (kaum Tiefeneffekt)
- **0.10–0.15**: Standard, natürlich wirkend
- **0.20–0.30**: Dramatischer Tiefeneffekt
- **0.40+**: Extremer 3D-Effekt (kann Artefakte verursachen)

```bash
# Subtiler Tiefeneffekt
python test_with_depth_video.py video.mp4 --scale_factor 0.10

# Dramatischer Tiefeneffekt
python test_with_depth_video.py video.mp4 --scale_factor 0.25
```

### `--skip_frames`
Ermöglicht schnellere Verarbeitung durch Überspringen von Frames.

- **1** (Standard): Alle Frames verarbeiten
- **2**: Jeden 2. Frame → 2x schneller, FPS halbiert
- **5**: Jeden 5. Frame → 5x schneller, aber ruckeliges Output

```bash
# Schnelle Tests (5x schneller, aber 5x weniger Frames)
python test_with_depth_video.py video.mp4 --skip_frames 5
```

### `--crf` (Constant Rate Factor)
H.264 Qualitäts-Schieberegler. Niedrigere Werte = bessere Qualität, aber größere Dateigröße.

| CRF | Qualität | Dateigröße | Anwendung |
|-----|----------|-----------|----------|
| 18–20 | Sehr hoch | Groß | Archivierung, Veröffentlichung |
| 23 | Gut | Mittel | **Standard** |
| 28 | Akzeptabel | Klein | Schnelle Verarbeitung, Vorschau |
| 35+ | Niedrig | Sehr klein | Tests, Debugging |

```bash
# Hochwertige Archivierung
python test_with_depth_video.py video.mp4 --crf 20

# Kleine Datei für Vorschau
python test_with_depth_video.py video.mp4 --crf 28
```

### `--preset`
Encoding-Geschwindigkeit (trade-off zwischen Verarbeitungszeit und Kompression).

| Preset | Geschwindigkeit | Kompression | CPU-Last |
|--------|-----------------|------------|----------|
| `ultrafast` | 🟢 Sehr schnell | Niedrig | Niedrig |
| `fast` | 🟢 Schnell | Mittel | Mittel |
| `medium` | 🟡 Normal | Gut | Hoch |
| `slow` | 🔴 Langsam | Sehr gut | Sehr hoch |
| `veryslow` | 🔴 Sehr langsam | Maximal | Maximal |

```bash
# Schnelle Verarbeitung (Test)
python test_with_depth_video.py video.mp4 --preset fast

# Beste Kompression (Archiv)
python test_with_depth_video.py video.mp4 --preset slow --crf 20
```

### `--use_ffmpeg`
Falls `False`: Fallback zu OpenCV VideoWriter (niedrigere Qualität, aber keine FFmpeg-Installation nötig).

```bash
# Ohne FFmpeg (wenn nicht installiert oder Fallback erwünscht)
python test_with_depth_video.py video.mp4 --use_ffmpeg False
```

---

## 💾 Output-Format

Das Skript speichert 4 MP4-Videos im Ausgabe-Verzeichnis `<output>/<video_basename>/`:

```
./vis/my_video/
├── left_video.mp4              (ursprüngliches Input-Bild, gecroppt, 768×768)
├── warped_video.mp4            (Warped-Bild aus Tiefenkarte)
├── generated_right_video.mp4   (GenStereo + Fusion-Ergebnis) ⭐ HAUPTAUSGABE
└── disp_video.mp4              (Tiefenkarte-Visualisierung als Heatmap)
```

### Erklärung der Ausgabe-Videos

1. **left_video.mp4**: Original-Input (gecroppt auf Quadrat, resized auf 768×768)
2. **warped_video.mp4**: Rechtes Bild mittels Depth-basiertem Warping erzeugt
3. **generated_right_video.mp4**: Rechtssicht durch GenStereo + AdaptiveFusionLayer
   - **Beste Qualität**: Kombiniert Warping + Diffusions-Generierung
4. **disp_video.mp4**: Disparity-Visualisierung (Inferno-Colormap)

---

## 📈 Verwendungsbeispiele

### Szenario 1: Schnelle Vorschau (Test)
```bash
python test_with_depth_video.py input.mp4 \
  --skip_frames 5 \
  --crf 28 \
  --preset fast \
  --output ./preview/
```
**Resultat**: ~30 Sekunden Verarbeitung, kleine Dateien, ruckelige Output

### Szenario 2: Standard-Qualität
```bash
python test_with_depth_video.py input.mp4 \
  --scale_factor 0.15 \
  --crf 23 \
  --preset medium \
  --output ./standard/
```
**Resultat**: Ausgewogene Qualität und Dateigröße

### Szenario 3: Hochwertige Archivierung
```bash
python test_with_depth_video.py input.mp4 \
  --crf 20 \
  --preset slow \
  --scale_factor 0.15 \
  --output ./archive/
```
**Resultat**: Beste Qualität, aber länger Verarbeitung und größere Dateien

### Szenario 4: Mit bestehendem Depth-Video
```bash
python test_with_depth_video.py input.mp4 \
  --depth_video_path depth_map.mp4 \
  --scale_factor 0.18 \
  --crf 23 \
  --output ./with_depth/
```
**Resultat**: 2–3x schneller, keine DAM2-Inferenz nötig

### Szenario 5: Dramatischer Tiefeneffekt
```bash
python test_with_depth_video.py input.mp4 \
  --scale_factor 0.30 \
  --crf 23 \
  --preset medium \
  --output ./dramatic/
```
**Resultat**: Stärkere Parallaxe, mehr 3D-Effekt

---

## ⚙️ Empfehlung für verschiedene Hardware

### CPU-only (langsam)
```bash
python test_with_depth_video.py input.mp4 \
  --skip_frames 5 \
  --preset fast \
  --crf 25
```

### GPU (CUDA/MPS verfügbar)
```bash
python test_with_depth_video.py input.mp4 \
  --skip_frames 1 \
  --preset medium \
  --crf 23
```

### Mac (M1/M2/M3, MPS)
```bash
python test_with_depth_video.py input.mp4 \
  --preset medium \
  --crf 23
```

---

## 🐛 Troubleshooting

### ❌ FFmpeg nicht gefunden
**Fehler**: `⚠ Warning: FFmpeg not found`

**Lösung**:
```bash
brew install ffmpeg
```
Oder im Skript-Aufruf:
```bash
python test_with_depth_video.py video.mp4 --use_ffmpeg False
```

### ❌ Fehlende Checkpoints
**Fehler**: `RuntimeError: Failed to load checkpoint`

**Lösung**: Download-Skript ausführen:
```bash
bash scripts/download_models.sh
```

### ❌ Out of Memory (OOM)
**Fehler**: `CUDA out of memory` oder `RuntimeError: ENOMEM`

**Lösung**: `--skip_frames` erhöhen:
```bash
python test_with_depth_video.py video.mp4 --skip_frames 2
```

### ❌ Video-Länge nicht erkannt
**Fehler**: `total_frames = 0`

**Lösung**: Video-Format konvertieren:
```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac input_converted.mp4
python test_with_depth_video.py input_converted.mp4
```

### ❌ Schlechte Output-Qualität
**Ursache**: Standardeinstellung `crf=23` zu hoch

**Lösung**: CRF senken:
```bash
python test_with_depth_video.py video.mp4 --crf 20
```

### ❌ Depth-Video hat andere Länge als Input
**Fehler**: `⚠ Warning: Depth video has X frames, input has Y`

**Lösung**: Videos müssen gleich lang sein. Falls nötig, Depth-Video anpassen oder DAM2-Inferenz nutzen.

---

## 📊 Performance-Richtlinien

| Komponente | Zeit pro Frame | Anmerkung |
|------------|----------------|----------|
| DAM2 Depth-Inferenz | ~2–5s | GPU: ~2s; CPU: ~10–15s |
| GenStereo Generierung | ~3–8s | GPU: ~3–5s; CPU: nicht praktikabel |
| Fusion | ~0.5s | Schnell |
| H.264 Encoding (FFmpeg) | ~1–3s | Abhängig von CRF und Preset |
| **Total (mit DAM2)** | ~7–16s/frame | → ~6–30 min pro Minute Video |
| **Total (mit Depth-Video)** | ~4–11s/frame | → ~4–22 min pro Minute Video |

**Beispiel**: 60 Sekunden Video, GPU, `skip_frames=1`:
- Mit DAM2: ~7–16 min
- Mit Depth-Video: ~4–11 min

---

## 🎨 Tipps für bessere Ergebnisse

1. **Input-Qualität**: Hochauflösende, stabile Videos → bessere Depth-Schätzung
2. **Lighting**: Gut beleuchtete Szenen → bessere GenStereo-Generierung
3. **Tiefe**: Videos mit klarer Tiefenstruktur → sichtbarere Stereoskopie
4. **Bewegung**: Langsame Kamerafahrten → glatterere Output
5. **Scale-Factor**: Mit 0.15 starten, dann anpassen

---

## 📝 Ausgabe-Verzeichnisstruktur

```
./vis/
└── my_video/
    ├── left_video.mp4              (Input)
    ├── warped_video.mp4            (Warp-Ergebnis)
    ├── generated_right_video.mp4   (Fusion-Resultat) ⭐
    └── disp_video.mp4              (Tiefenkarte)
```

---

## 📚 Verwandte Skripte

- **`test_with_depth.py`**: Verarbeitet einzelne Bilder (nicht Videos)
- **`test_arbitrary_size.py`**: Bilder beliebiger Größe
- **`test.py`**: Basis-Test ohne Depth-Maps

---

## 📄 Lizenz & Referenzen

- **GenStereo**: [Paper/Repository]
- **Depth-Anything V2**: [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Diffusers**: [https://huggingface.co/diffusers](https://huggingface.co/diffusers)

---

## ❓ Häufig gestellte Fragen

**F: Wie lange dauert die Verarbeitung?**
A: ~7–16 Sekunden pro Frame mit GPU bei DAM2-Inferenz. Mit bestehendem Depth-Video: ~4–11s.

**F: Kann ich ohne GPU arbeiten?**
A: Ja, aber sehr langsam (~30–60s/Frame). `--skip_frames` erhöhen für schnellere Tests.

**F: Was ist `--scale_factor`?**
A: Kontrolliert die Stärke der Parallaxe. Standard 0.15, höher = dramatischer.

**F: Gibt es Möglichkeit, nur eine Ausgabe (z.B. nur `generated_right_video.mp4`) zu speichern?**
A: Aktuell nicht. Alle 4 Videos werden immer erzeugt. Sie können die anderen später löschen.

**F: Kann ich die Auflösung ändern?**
A: Derzeit fest auf 768×768 (v2.1) oder 512×512 (v1.5). Siehe `IMAGE_SIZE` im Skript.

---

**Version**: 1.0  
**Stand**: Februar 2026  
**Autor**: GenStereo Team
