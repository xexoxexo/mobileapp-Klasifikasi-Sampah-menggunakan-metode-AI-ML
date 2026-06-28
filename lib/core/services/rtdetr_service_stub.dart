
import 'package:flutter/foundation.dart';

import '../models/scan_result.dart';

/// Web stub — RT-DETR inference is not available in the browser.
class RTDETRService {
  bool get isLoaded => false;

  Future<void> loadModel() async {
    debugPrint('[RTDETR] skipped on web (use Gemini API instead)');
  }

  Future<List<ScanResult>> detectObjects(Uint8List imageBytes) async => [];

  Future<ScanResult?> detectSingle(Uint8List imageBytes) async => null;

  void dispose() {}
}
