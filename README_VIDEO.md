# GenStereo Video Processing

Stereo-Video-Erzeugung aus Mono-Videos mit Depth-Maps.

## Voraussetzungen

### Erforderlich

- Python 3.10+
- CUDA-fähige GPU (empfohlen) oder CPU
- FFmpeg

### FFmpeg Installation

```bash
# Windows (Chocolatey)
choco install ffmpeg

# Windows (alternativ)
# Download von https://ffmpeg.org und zum PATH hinzufügen

# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```

### Python-Pakete

```bash
pip install -r requirements.txt
```

### Optional: Real-ESRGAN für Upscaling

```bash
pip install basicsr realesrgan
```

> **Hinweis:** Bei Kompatibilitätsproblemen mit `torchvision`:
> Datei `basicsr/data/degradations.py` Zeile 8 ändern:
> ```python
> # Alt:
> from torchvision.transforms.functional_tensor import rgb_to_grayscale
> # Neu:
> try:
>     from torchvision.transforms.functional_tensor import rgb_to_grayscale
> except ImportError:
>     from torchvision.transforms.functional import rgb_to_grayscale
> ```

---

## Verwendung

### Basis-Aufruf

```bash
python test_with_depth.py video.mp4 depth_video.mp4
```

### Alle Parameter

```bash
python test_with_depth.py video.mp4 depth_video.mp4 [OPTIONEN]
```

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `video_path` | Pflicht | - | Pfad zum Input-Video |
| `depth_video_path` | Pflicht | - | Pfad zur Depth-Map (Graustufen-Video) |
| `--output` | Optional | `./vis` | Ausgabeverzeichnis |
| `--convergence` | Optional | `0.02` | Konvergenz-Faktor für Stereo-Tiefe |
| `--upscale` | Optional | `2` | Upscaling-Faktor (0, 2, 4) |
| `--save_all` | Flag | - | Zusätzlich disp.mp4 und warped.mp4 speichern |
| `--crf` | Optional | `23` | H.264 Qualität (0-51, niedriger = besser) |
| `--preset` | Optional | `medium` | Encoding-Geschwindigkeit |
| `--clear_every` | Optional | `10` | Speicherbereinigung alle N Frames |

---

## Beispiele

### Standard (2x Upscaling)

```bash
python test_with_depth.py scene.mp4 scene_depth.mp4
```

### Ohne Upscaling

```bash
python test_with_depth.py scene.mp4 scene_depth.mp4 --upscale 0
```

### 4x Upscaling mit besserer Qualität

```bash
python test_with_depth.py scene.mp4 scene_depth.mp4 --upscale 4 --crf 18 --preset slow
```

### Alle Ausgaben speichern

```bash
python test_with_depth.py scene.mp4 scene_depth.mp4 --save_all
```

### Benutzerdefiniertes Ausgabeverzeichnis

```bash
python test_with_depth.py scene.mp4 scene_depth.mp4 --output ./stereo_output
```

### Aggressives Speicher-Management (bei VRAM-Problemen)

```bash
python test_with_depth.py scene.mp4 scene_depth.mp4 --clear_every 5
```

---

## Eingabeformate

### Video-Input

- **Format:** MP4, AVI, MOV, MKV (alle von FFmpeg unterstützten Formate)
- **Farbraum:** RGB/BGR (automatisch konvertiert)

### Depth-Map-Input

- **Format:** Gleiches Format wie Video
- **Farbraum:** Graustufen (wird automatisch konvertiert)
- **Frames:** Muss gleiche Anzahl Frames wie Video haben

---

## Ausgabe

Im Ausgabeverzeichnis wird ein Unterordner mit dem Videonamen erstellt:

```
vis/
└── scene/
    ├── left.mp4       # Linkes Auge (Original)
    └── right.mp4      # Rechtes Auge (generiert)
```

Mit `--save_all`:

```
vis/
└── scene/
    ├── left.mp4       # Linkes Auge
    ├── right.mp4      # Rechtes Auge
    ├── disp.mp4       # Depth-Map Visualisierung
    └── warped.mp4     # Warped Intermediate
```

---

## Qualitätseinstellungen

### CRF-Werte (H.264)

| CRF | Qualität | Dateigröße | Empfohlen für |
|-----|----------|------------|---------------|
| 18 | Sehr hoch | Groß | Archivierung |
| 23 | Gut | Mittel | Standard |
| 28 | Akzeptabel | Klein | Preview |

### Preset-Werte

| Preset | Geschwindigkeit | Kompression |
|--------|-----------------|-------------|
| `ultrafast` | Sehr schnell | Niedrig |
| `fast` | Schnell | Mittel |
| `medium` | Mittel | Gut |
| `slow` | Langsam | Besser |
| `veryslow` | Sehr langsam | Beste |

---

## Convergence-Parameter

Der `--convergence` Parameter steuert die Stereo-Tiefe:

| Wert | Effekt |
|------|--------|
| `0.01` | Sehr geringe Tiefe |
| `0.02` | Standard (ausgewogen) |
| `0.03` | Stärkere Tiefe |
| `0.05` | Starke Tiefe (kann Artefakte verursachen) |

---

## Fehlerbehebung

### CUDA nicht erkannt

```bash
# Prüfen
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Falls False, PyTorch neu installieren
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### VRAM-Probleme

```bash
# Weniger Upscaling
python test_with_depth.py video.mp4 depth.mp4 --upscale 0

# Häufiger Speicher-Cleanup
python test_with_depth.py video.mp4 depth.mp4 --clear_every 5
```

### FFmpeg nicht gefunden

```bash
# Prüfen
ffmpeg -version

# Falls nicht gefunden, siehe Installation oben
```

### Real-ESRGAN Fehler

```bash
# Neu installieren
pip uninstall basicsr realesrgan -y
pip install basicsr realesrgan
```

---

## Technische Details

### Verwendete Modelle

| Modell | Zweck | Auflösung |
|--------|-------|-----------|
| GenStereo v2.1 | Stereo-Generierung | 768x768 |
| Real-ESRGAN x2plus | 2x Upscaling | - |
| Real-ESRGAN x4plus | 4x Upscaling | - |

### Verarbeitung

1. Video-Frames werden auf 768x768 skaliert (quadratisch)
2. Depth-Map wird auf 768x768 skaliert
3. Stereo-Generierung durch GenStereo
4. Rückskalierung auf ursprüngliches Seitenverhältnis
5. Optional: Upscaling durch Real-ESRGAN
6. Encoding durch FFmpeg (H.264)

### Speicherverwaltung

- Automatisches Cleanup alle 10 Frames
- VRAM-Anzeige alle 50 Frames
- Tiling für Real-ESRGAN (reduziert VRAM)

---

## Systemanforderungen

### Minimum

- GPU: 4GB VRAM
- RAM: 8GB
- Speicher: 10GB frei

### Empfohlen

- GPU: 8GB+ VRAM (z.B. RTX 3060 Ti)
- RAM: 16GB
- Speicher: 50GB frei (für längere Videos)

---

## Lizenz

Siehe Haupt-README.md und die Lizenzen der verwendeten Modelle.