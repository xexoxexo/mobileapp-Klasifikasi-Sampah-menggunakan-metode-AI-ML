import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../models/scan_result.dart';

/// Web stub — TFLite inference is not available in the browser.
class TFLiteService {
  bool get isLoaded => false;

  ({String name, double confidence})? classifyFrameQuick(Uint8List imageBytes) =>
      null;

  Future<void> loadModel() async {
    debugPrint('TFLiteService: skipped on web (use Gemini API instead)');
  }

  Future<ScanResult> classifyImage(Uint8List imageBytes) async {
    throw UnsupportedError('TFLite is not available on web');
  }

  Future<List<ScanResult>> classifyMultipleImages(
    Uint8List imageBytes, {
    List<Rect>? regions,
  }) async =>
      [];

  void dispose() {}
}
