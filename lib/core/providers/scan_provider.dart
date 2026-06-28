import 'dart:ui' show Rect;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image/image.dart' as img;

import '../models/scan_result.dart';
import '../models/waste_category.dart';
import '../services/gemini_service.dart';
import '../services/object_validator_service.dart';
import '../services/rtdetr_service.dart';
import '../services/tflite_service.dart';
import '../services/session_service.dart';
import 'local_dataset_provider.dart';
import 'session_provider.dart';

class ScanNotifier extends StateNotifier<AsyncValue<ScanResult?>> {
  final TFLiteService _tfliteService;
  final RTDETRService _rtdetrService;
  final GeminiService _geminiService;
  final Ref _ref;

  List<ScanResult> _multiResults = [];
  int _selectedDetailIndex = 0;
  bool _isCorrected = false;
  bool _lastUsedGemini = false;

  List<ScanResult> get multiResults => _multiResults;
  int get selectedDetailIndex => _selectedDetailIndex;
  bool get isCorrected => _isCorrected;

  /// True if the most recent classification came from the Gemini cloud API
  /// rather than the on-device TFLite/RT-DETR model. Useful for UI badges
  /// ("Dianalisis ulang oleh AI cloud").
  bool get lastUsedGemini => _lastUsedGemini;

  /// True if a Gemini API key is configured. Use this for UI guards —
  /// does NOT trigger model init, so it's safe to call before the first
  /// classification.
  bool get geminiAvailable => _geminiService.isConfigured;

  ScanNotifier(this._ref)
      : _tfliteService = TFLiteService(),
        _rtdetrService = RTDETRService(),
        _geminiService = GeminiService(),
        super(const AsyncValue.data(null));

  Future<ScanResult?> classifyImage(Uint8List imageBytes) async {
    _isCorrected = false;
    _lastUsedGemini = false;
    state = const AsyncValue.loading();
    try {
      if (kIsWeb) {
        final webResult = await _classifyOnWeb(imageBytes);
        state = AsyncValue.data(webResult);
        return webResult;
      }

      // ── Local dataset lookup ──
      // Skip ML entirely if a perceptually similar image is already in the
      // user's on-device dataset. The dataset only contains user-confirmed
      // results (accepted AI prediction or manually-corrected category), so
      // a hit is treated as ground truth.
      final cached = _checkLocalDataset(imageBytes);
      if (cached != null) {
        state = AsyncValue.data(cached);
        return cached;
      }

      if (_rtdetrService.isLoaded || !_rtdetrService.isLoaded) {
        await _rtdetrService.loadModel();
      }
      if (_rtdetrService.isLoaded) {
        final result = await _rtdetrService.detectSingle(imageBytes);
        if (result != null) {
          final withCrop = _ensureCroppedImage(result, imageBytes);
          // Re-check dataset against the cropped object — handles the case
          // where the whole image didn't match but the detected object does.
          final cropCached = withCrop.croppedImage != null
              ? _checkLocalDataset(withCrop.croppedImage!)
              : null;
          final final_ = cropCached ?? withCrop;
          state = AsyncValue.data(final_);
          return final_;
        }
      }

      await _tfliteService.loadModel();
      final result = await _tfliteService.classifyImage(imageBytes);
      state = AsyncValue.data(result);
      return result;
    } catch (e) {
      debugPrint('ScanNotifier: classifyImage error = $e');
      final errorResult = ScanResult(
        itemName: 'Model tidak tersedia',
        category: WasteCategory.lainnya,
        confidence: 0.0,
        description: 'File model TFLite belum ditemukan. '
            'Jalankan export_tflite.py lalu taruh waste_classifier.tflite di assets/models/',
        disposalInfo: 'Hubungi developer untuk setup model.',
        allProbabilities: {},
      );
      state = AsyncValue.data(errorResult);
      return errorResult;
    }
  }

  /// Escalate classification to the Gemini cloud API.
  ///
  /// Returns the new [ScanResult] (also published to [state]) if Gemini is
  /// available AND confident; otherwise returns `null` and leaves the
  /// previous on-device result in [state] so callers can fall back to the
  /// original behaviour (e.g. show the low-confidence screen).
  Future<ScanResult?> classifyWithGemini(Uint8List imageBytes) async {
    // ── Local dataset lookup ──
    // Same rationale as classifyImage — if the user has already confirmed
    // a category for a perceptually similar image, skip the cloud call.
    final cached = _checkLocalDataset(imageBytes);
    if (cached != null) {
      _lastUsedGemini = false;
      _isCorrected = false;
      state = AsyncValue.data(cached);
      return cached;
    }

    if (!_geminiService.isAvailable) {
      // Trigger lazy init so subsequent checks have an accurate flag.
      final result = await _geminiService.classifyImage(imageBytes);
      if (result == null) return null;
      _lastUsedGemini = true;
      _isCorrected = false;
      state = AsyncValue.data(result);
      return result;
    }

    final result = await _geminiService.classifyImage(imageBytes);
    if (result == null) {
      // Gemini rejected or unavailable — keep the existing on-device result.
      return null;
    }

    _lastUsedGemini = true;
    _isCorrected = false;
    state = AsyncValue.data(result);
    return result;
  }

  Future<List<ScanResult>> classifyMultipleImages(Uint8List imageBytes) async {
    try {
      if (kIsWeb) {
        return classifyMultipleWithGemini(imageBytes);
      }

      // Try RT-DETR first
      await _rtdetrService.loadModel();
      if (_rtdetrService.isLoaded) {
        final results = await _rtdetrService.detectObjects(imageBytes);
        if (results.isNotEmpty) {
          debugPrint('[ScanNotifier] RT-DETR found ${results.length} objects');
          // Ensure all have cropped images
          final enriched = results
              .map((r) => _ensureCroppedImage(r, imageBytes))
              .toList();
          // Per-item dataset lookup — replaces any item whose crop matches
          // a previously-confirmed entry with the cached category.
          final withDataset = _enrichMultiWithLocalDataset(enriched);
          _multiResults = withDataset;
          _selectedDetailIndex = 0;
          _saveMultiToHistory(withDataset);
          _invalidateMultiResultProviders();
          return withDataset;
        }
      }

      debugPrint('[ScanNotifier] RT-DETR empty, falling back to TFLite');

      // Fallback to brightness detection + crop + classify
      await _tfliteService.loadModel();

      final validator = ObjectValidatorService();
      final detection = await validator.detectObjectRegions(imageBytes);
      final detectedRegions = detection.regions;

      debugPrint('ScanNotifier: Detected ${detectedRegions.length} object regions');

      final classifyRects = detectedRegions
          .take(5)
          .map((r) => r.classificationRect)
          .toList();
      var results = await _tfliteService.classifyMultipleImages(
        imageBytes,
        regions: classifyRects.isNotEmpty ? classifyRects : null,
      );

      if (results.isEmpty) {
        debugPrint('ScanNotifier: Multi-classify empty, falling back to whole-image');
        try {
          final fallback = await _tfliteService.classifyImage(imageBytes);
          results = [fallback];
        } catch (e) {
          debugPrint('ScanNotifier: Fallback classify error = $e');
        }
      }

      // For fallback results without cropped images, crop from the detected regions
      if (classifyRects.isNotEmpty && results.length <= classifyRects.length) {
        results = List.generate(results.length, (i) {
          if (results[i].croppedImage != null) return results[i];
          final rect = classifyRects[i];
          return _cropResultFromRect(results[i], imageBytes, rect);
        });
      }

      _multiResults = _enrichMultiWithLocalDataset(results);
      _selectedDetailIndex = 0;
      _saveMultiToHistory(_multiResults);
      _invalidateMultiResultProviders();
      return _multiResults;
    } catch (e) {
      debugPrint('ScanNotifier: classifyMultipleImages error = $e');

      try {
        await _tfliteService.loadModel();
        final fallback = await _tfliteService.classifyImage(imageBytes);
        _multiResults = [fallback];
        _selectedDetailIndex = 0;
        _invalidateMultiResultProviders();
        return [fallback];
      } catch (e2) {
        debugPrint('ScanNotifier: Last resort classify error = $e2');
        _multiResults = [];
        _invalidateMultiResultProviders();
        return [];
      }
    }
  }

  /// Cloud-based multi-item classification via Gemini. Mirrors
  /// [classifyMultipleImages] but routes through [GeminiService.classifyMultiple]
  /// instead of the on-device RT-DETR/TFLite pipeline. Triggered when the user
  /// taps "Pindai Lagi" on the multi-result screen — the cloud model is more
  /// accurate for hard cases where on-device detection misfired (e.g. RT-DETR
  /// returns empty and the TFLite fallback splits the image into wrong regions).
  ///
  /// Does NOT filter by confidence so the multi-result UI can render low-
  /// confidence items as "Tidak dikenali" inside the list — same behavior as
  /// the on-device multi path.
  ///
  /// Crops each item from the original image using the normalized bbox Gemini
  /// returns, so each card on the multi-result screen shows the matching
  /// object (not the shared full-frame fallback). Items without a usable bbox
  /// fall back to the full image.
  Future<List<ScanResult>> classifyMultipleWithGemini(Uint8List imageBytes) async {
    try {
      final predictions = await _geminiService.classifyMultiple(imageBytes);

      // Decode once + reuse for all per-item crops. Falls back to no-crop
      // path if the bytes can't be decoded (rare, but guards the UI).
      final decoded = img.decodeImage(imageBytes);

      final results = <ScanResult>[];
      for (final p in predictions) {
        Rect? pixelRect;
        Uint8List? cropped;
        if (decoded != null && p.bbox != null) {
          pixelRect = Rect.fromLTWH(
            p.bbox![1] * decoded.width,
            p.bbox![0] * decoded.height,
            (p.bbox![3] - p.bbox![1]) * decoded.width,
            (p.bbox![2] - p.bbox![0]) * decoded.height,
          );
          cropped = _cropFromBytes(imageBytes, pixelRect);
        }

        results.add(ScanResult(
          itemName: _itemNameForGemini(p.category),
          category: p.category,
          confidence: p.confidence,
          description: p.reason,
          disposalInfo: p.category.disposalInfo,
          boundingBox: pixelRect,
          croppedImage: cropped,
          allProbabilities: {
            p.category.name: p.confidence,
            'Residu': 1.0 - p.confidence,
          },
        ));
      }

      _multiResults = _enrichMultiWithLocalDataset(results);
      _selectedDetailIndex = 0;
      if (_multiResults.isNotEmpty) {
        _lastUsedGemini = true;
        _saveMultiToHistory(_multiResults);
      }
      _invalidateMultiResultProviders();
      return results;
    } catch (e) {
      debugPrint('ScanNotifier: classifyMultipleWithGemini error = $e');
      _multiResults = [];
      _selectedDetailIndex = 0;
      _invalidateMultiResultProviders();
      return [];
    }
  }

  /// Item-name helper that mirrors [GeminiService]'s private `_itemNameFor`
  /// without forcing this class to import the service's private member.
  String _itemNameForGemini(WasteCategory category) {
    switch (category) {
      case WasteCategory.plastik:
        return 'Sampah Plastik';
      case WasteCategory.kertas:
        return 'Sampah Kertas';
      case WasteCategory.organik:
        return 'Sampah Organik';
      case WasteCategory.logam:
        return 'Sampah Logam';
      case WasteCategory.kaca:
        return 'Sampah Kaca';
      case WasteCategory.residu:
        return 'Sampah Residu';
      case WasteCategory.lainnya:
        return 'Sampah Tidak Dikenali';
    }
  }

  /// Force [multiResultsProvider] and [selectedDetailProvider] to recompute
  /// on next read. Both providers wrap `_multiResults` via `ref.read` (so
  /// they don't subscribe to notifier mutations) and would otherwise return
  /// STALE results across navigations — e.g. after a "Pindai Lagi" rescan
  /// the multi-result screen and the detail dialog would still show the
  /// previous TFLite/RT-DETR items instead of the freshly-classified ones.
  /// Must be called whenever `_multiResults` changes.
  void _invalidateMultiResultProviders() {
    _ref.invalidate(multiResultsProvider);
    _ref.invalidate(selectedDetailProvider);
  }

  /// Web builds skip on-device TFLite/RT-DETR and route through Gemini.
  Future<ScanResult> _classifyOnWeb(Uint8List imageBytes) async {
    final cached = _checkLocalDataset(imageBytes);
    if (cached != null) {
      _lastUsedGemini = false;
      return cached;
    }

    if (!_geminiService.isConfigured) {
      return ScanResult(
        itemName: 'Klasifikasi web tidak tersedia',
        category: WasteCategory.lainnya,
        confidence: 0.0,
        description: 'Di browser, inferensi ML on-device tidak didukung. '
            'Isi GEMINI_API_KEY di file .env untuk klasifikasi via AI cloud.',
        disposalInfo: 'Hubungi developer untuk konfigurasi API key.',
        allProbabilities: {},
      );
    }

    final result = await classifyWithGemini(imageBytes);
    if (result != null) return result;

    return ScanResult(
      itemName: 'Gagal menganalisis',
      category: WasteCategory.lainnya,
      confidence: 0.0,
      description: 'Gemini tidak dapat mengklasifikasi gambar ini. Coba lagi.',
      disposalInfo: 'Coba scan ulang.',
      allProbabilities: {},
    );
  }

  /// Look the image up in the user's on-device dataset. Returns a synthetic
  /// high-confidence [ScanResult] if a perceptually similar image has been
  /// confirmed before; otherwise null. Bypasses ML entirely on hit.
  ScanResult? _checkLocalDataset(Uint8List imageBytes) {
    try {
      final match = _ref.read(localDatasetProvider).findMatch(imageBytes);
      if (match == null) return null;
      debugPrint('[ScanNotifier] local dataset hit '
          '(distance=${match.hammingDistance}) → ${match.entry.category.name}');
      final cat = match.entry.category;
      return ScanResult(
        itemName: match.entry.itemName ?? 'Sampah dari dataset lokal',
        category: cat,
        confidence: 1.0,
        description: 'Sudah pernah kamu pindai & simpan sebelumnya '
            '(${match.hammingDistance}/8 mirip).',
        disposalInfo: cat.disposalInfo,
        allProbabilities: {cat.name: 1.0},
      );
    } catch (e) {
      debugPrint('[ScanNotifier] local dataset lookup failed: $e');
      return null;
    }
  }

  /// Per-item dataset enrichment for multi-mode. Replaces each result whose
  /// cropped image perceptually matches a stored entry with a synthetic
  /// 100%-confidence result for that entry's category. Items without a
  /// cropped image or without a match pass through unchanged.
  List<ScanResult> _enrichMultiWithLocalDataset(List<ScanResult> results) {
    final dataset = _ref.read(localDatasetProvider);
    var hitCount = 0;
    final enriched = results.map((r) {
      final crop = r.croppedImage;
      if (crop == null) return r;
      final match = dataset.findMatch(crop);
      if (match == null) return r;
      hitCount++;
      final cat = match.entry.category;
      return ScanResult(
        itemName: match.entry.itemName ?? r.itemName,
        category: cat,
        confidence: 1.0,
        description: 'Sudah pernah kamu pindai & simpan sebelumnya '
            '(${match.hammingDistance}/8 mirip).',
        disposalInfo: cat.disposalInfo,
        boundingBox: r.boundingBox,
        croppedImage: r.croppedImage,
        allProbabilities: {cat.name: 1.0},
      );
    }).toList();
    if (hitCount > 0) {
      debugPrint('[ScanNotifier] multi dataset enrichment: '
          '$hitCount/${results.length} items matched local dataset');
    }
    return enriched;
  }

  /// Persist a single confirmed result to the on-device dataset. Prefers the
  /// cropped image (the object itself) over the full captured frame so
  /// future scans of just that object match cleanly.
  Future<void> _saveToLocalDataset(ScanResult result) async {
    final crop = result.croppedImage;
    final full = _ref.read(capturedImageProvider);
    final bytes = crop ?? full;
    if (bytes == null) {
      debugPrint('[ScanNotifier] saveToLocalDataset skipped — no image bytes');
      return;
    }
    try {
      await _ref.read(localDatasetProvider).saveEntry(
        bytes,
        category: result.category,
        itemName: result.itemName,
        confidence: result.confidence,
      );
    } catch (e) {
      debugPrint('[ScanNotifier] saveToLocalDataset error: $e');
    }
  }

  /// Ensure a ScanResult has a croppedImage by cropping from the original
  /// image bytes using the bounding box if available.
  ScanResult _ensureCroppedImage(ScanResult result, Uint8List originalBytes) {
    if (result.croppedImage != null) return result;
    if (result.boundingBox == null) return result;

    final cropped = _cropFromBytes(originalBytes, result.boundingBox!);
    if (cropped == null) return result;

    return result.copyWith(croppedImage: cropped);
  }

  /// Crop from bounding box rect for fallback TFLite results.
  ScanResult _cropResultFromRect(ScanResult result, Uint8List originalBytes, Rect rect) {
    final cropped = _cropFromBytes(originalBytes, rect);
    if (cropped == null) return result;
    return result.copyWith(
      croppedImage: cropped,
      boundingBox: result.boundingBox ?? rect,
    );
  }

  /// Crop a region from image bytes. Returns JPEG bytes or null on failure.
  Uint8List? _cropFromBytes(Uint8List imageBytes, Rect rect) {
    try {
      final image = img.decodeImage(imageBytes);
      if (image == null) return null;

      final x1 = rect.left.round().clamp(0, image.width);
      final y1 = rect.top.round().clamp(0, image.height);
      final x2 = rect.right.round().clamp(0, image.width);
      final y2 = rect.bottom.round().clamp(0, image.height);
      final w = x2 - x1;
      final h = y2 - y1;

      if (w < 10 || h < 10) return null;

      final cropped = img.copyCrop(image, x: x1, y: y1, width: w, height: h);
      return Uint8List.fromList(img.encodeJpg(cropped, quality: 85));
    } catch (e) {
      debugPrint('[ScanNotifier] Crop failed: $e');
      return null;
    }
  }

  void _saveMultiToHistory(List<ScanResult> results) {
    if (results.isNotEmpty) {
      final session = _ref.read(sessionProvider);
      final updatedHistory = [...session.scanHistory, ...results];
      _ref.read(sessionProvider.notifier).state = session.copyWith(
        scanHistory: updatedHistory,
      );
      // NOTE: local-dataset persistence for multi-mode happens on the
      // "Selesai" handler in multi_result_screen, NOT here, because
      // classification runs before the user has reviewed the items.
    }
  }

  /// Bulk-save multi-mode items to the on-device dataset. Called by the
  /// multi-result screen's "Selesai" button after the user accepts the
  /// (possibly corrected) per-item categories. Each item's croppedImage
  /// is saved as its own dataset entry so future single- or multi-scans
  /// match it individually.
  Future<void> saveMultiToLocalDataset(List<ScanResult> results) async {
    for (final r in results) {
      if (r.croppedImage != null) {
        await _saveToLocalDataset(r);
      }
    }
  }

  void selectDetail(int index) {
    if (index >= 0 && index < _multiResults.length) {
      _selectedDetailIndex = index;
    }
  }

  Future<void> correctResult(WasteCategory newCategory) async {
    final current = state.valueOrNull;
    if (current == null) return;

    final corrected = current.copyWith(
      category: newCategory,
      disposalInfo: newCategory.disposalInfo,
      isCorrected: true,
      originalCategory: current.category,
    );

    state = AsyncValue.data(corrected);
    _isCorrected = true;

    final session = _ref.read(sessionProvider);
    final updatedHistory = session.scanHistory.map((r) {
      if (r.itemName == current.itemName &&
          r.confidence == current.confidence) {
        return corrected;
      }
      return r;
    }).toList();

    _ref.read(sessionProvider.notifier).state =
        session.copyWith(scanHistory: updatedHistory);

    // If the original image is already in the on-device dataset (saved
    // before the user opened manual correction), update its category too
    // so future scans reflect the corrected label. If it's not yet in the
    // dataset, the manual-correction screen will save the corrected entry
    // on "Simpan Koreksi".
    final crop = current.croppedImage;
    final full = _ref.read(capturedImageProvider);
    final bytes = crop ?? full;
    if (bytes != null) {
      try {
        await _ref
            .read(localDatasetProvider)
            .updateCategoryForImage(bytes, newCategory: newCategory);
      } catch (e) {
        debugPrint('[ScanNotifier] updateCategoryForImage failed: $e');
      }
    }

    await _ref.read(sessionProvider.notifier).addCorrection();
    await _ref.read(sessionProvider.notifier).addXP(SessionService.xpCorrection);
  }

  void saveToHistory() {
    final result = state.valueOrNull;
    if (result == null) return;
    final session = _ref.read(sessionProvider);
    final updatedHistory = [...session.scanHistory, result];
    _ref.read(sessionProvider.notifier).state = session.copyWith(
      scanHistory: updatedHistory,
    );
    // Persist to on-device dataset so future scans of the same item reuse
    // this category instead of re-running ML.
    _saveToLocalDataset(result);
  }

  /// Public wrapper around [_saveToLocalDataset] for flows that bypass
  /// [saveToHistory] — most notably the manual-correction screen's
  /// "Simpan Koreksi" handler, which calls [correctResult] (an in-place
  /// update) but never calls [saveToHistory] itself. Without this call,
  /// a correction made directly from the result screen wouldn't reach the
  /// on-device dataset.
  Future<void> saveCorrectedToLocalDataset() async {
    final result = state.valueOrNull;
    if (result == null) return;
    await _saveToLocalDataset(result);
  }

  void clearResult() {
    state = const AsyncValue.data(null);
    _multiResults = [];
    _selectedDetailIndex = 0;
    _isCorrected = false;
    _lastUsedGemini = false;
    _invalidateMultiResultProviders();
  }

  /// Update a specific item in multi-results (used after re-analyzing an unknown item).
  void updateMultiResult(int index, ScanResult newResult) {
    if (index < 0 || index >= _multiResults.length) return;
    _multiResults[index] = newResult;
    // Also update the single result state so analyzing screen can read it
    state = AsyncValue.data(newResult);
    _invalidateMultiResultProviders();
  }
}

final scanProvider =
    StateNotifierProvider<ScanNotifier, AsyncValue<ScanResult?>>((ref) {
  return ScanNotifier(ref);
});

final scanResultProvider = Provider<ScanResult?>((ref) {
  return ref.watch(scanProvider).valueOrNull;
});

final isScanningProvider = Provider<bool>((ref) {
  return ref.watch(scanProvider).isLoading;
});

final isCorrectedProvider = Provider<bool>((ref) {
  return ref.read(scanProvider.notifier).isCorrected;
});

/// When true, the next classification in scanning_screen will use the
/// Gemini cloud API instead of the on-device TFLite/RT-DETR model.
/// Set by retry buttons ("Pindai Lagi", "Analisis AI") and reset by
/// scanning_screen after the classification runs.
final useGeminiProvider = StateProvider<bool>((ref) => false);

/// True if the current result was produced by the Gemini cloud API
/// (vs the on-device TFLite/RT-DETR model).
final lastUsedGeminiProvider = Provider<bool>((ref) {
  return ref.read(scanProvider.notifier).lastUsedGemini;
});

/// True if the Gemini API key is configured and the service initialised.
final geminiAvailableProvider = Provider<bool>((ref) {
  return ref.read(scanProvider.notifier).geminiAvailable;
});

final multiResultsProvider = Provider<List<ScanResult>>((ref) {
  return ref.read(scanProvider.notifier).multiResults;
});

final capturedImageProvider = StateProvider<Uint8List?>((ref) => null);

final selectedDetailIndexProvider = StateProvider<int>((ref) => 0);

/// When set, the analyzing screen is re-analyzing a specific multi-result item.
/// After analysis, it navigates back to /multi-result instead of /result.
final reanalyzeMultiIndexProvider = StateProvider<int?>((ref) => null);

final selectedDetailProvider = Provider<ScanResult?>((ref) {
  final results = ref.read(scanProvider.notifier).multiResults;
  final index = ref.watch(selectedDetailIndexProvider);
  if (index >= 0 && index < results.length) {
    return results[index];
  }
  return null;
});

final tfliteServiceProvider = Provider<TFLiteService>((ref) {
  final service = TFLiteService();
  ref.onDispose(() {
    service.dispose();
  });
  return service;
});

final rtdetrServiceProvider = Provider<RTDETRService>((ref) {
  final service = RTDETRService();
  ref.onDispose(() {
    service.dispose();
  });
  return service;
});
