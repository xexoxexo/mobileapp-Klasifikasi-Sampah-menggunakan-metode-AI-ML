# I'm ur Biny

AI-powered waste sorting interactive display application built with Flutter and TFLite.

## Features

- Real-time waste classification using MobileNet TFLite model
- Interactive mascot (Biny) for user engagement
- Single and mixed waste scanning modes
- Education and feedback system

## Getting Started

1. Install Flutter SDK (>= 3.12.1)
2. Run `flutter pub get`
3. Copy `.env` and set `GEMINI_API_KEY` (required for web; optional on Android if using on-device ML)
4. Run on your target platform:

### Android (full on-device ML)

```bash
flutter run
```

Connect an Android device or start an emulator first.

### Web (browser)

```bash
flutter run -d chrome
```

On web, on-device TFLite/RT-DETR is not available — classification uses the Gemini cloud API. The browser will ask for camera permission when scanning.

Other web targets:

```bash
flutter run -d edge
flutter run -d web-server   # serves at http://localhost:8080
```
