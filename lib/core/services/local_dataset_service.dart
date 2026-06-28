import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:hive_ce_flutter/hive_ce_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import '../models/waste_category.dart';
import 'dataset_image_storage.dart' as dataset_storage;

/// One persisted entry in the user's local scan dataset.
///
/// Stored in a Hive box keyed by the image's SHA-256 hex (so identical bytes
/// dedupe naturally). The perceptual hash is stored alongside for fast
/// similarity lookup when a NEW photo of an already-scanned object comes in.
class LocalDatasetEntry {
  final String sha256Hex;
  final int pHash;
  final WasteCategory category;
  final String? itemName;
  final double? confidence;
  final DateTime createdAt;
  final String imageFileName;

  const LocalDatasetEntry({
    required this.sha256Hex,
    required this.pHash,
    required this.category,
    required this.imageFileName,
    this.itemName,
    this.confidence,
    required this.createdAt,
  });

  Map<String, dynamic> toMap() => {
        'sha256': sha256Hex,
        'phash': pHash,
        // Use int index for serialization. WasteCategory has a custom `name`
        // getter (extension) that overrides Dart's default `.name` and returns
        // the CAPITALIZED display form ("Residu"), which broke the previous
        // `WasteCategory.values.byName(...)` round-trip. Index is stable as
        // long as the enum declaration order doesn't change — and CLAUDE.md
        // pins that order.
        'category_index': category.index,
        'item_name': itemName,
        'confidence': confidence,
        'image': imageFileName,
        'created_at': createdAt.toIso8601String(),
      };

  factory LocalDatasetEntry.fromMap(Map<dynamic, dynamic> map) {
    return LocalDatasetEntry(
      sha256Hex: map['sha256'] as String,
      pHash: map['phash'] as int,
      category: _categoryFromMap(map),
      itemName: map['item_name'] as String?,
      confidence: (map['confidence'] as num?)?.toDouble(),
      imageFileName: map['image'] as String,
      createdAt: DateTime.parse(map['created_at'] as String),
    );
  }
}

/// Resolve a [WasteCategory] from a stored map. Prefers the new int
/// `category_index` field; falls back to the legacy string form for entries
/// written before the index migration. The fallback supports both the
/// lowercase Dart enum identifier ("residu") AND the capitalized display
/// name from the custom `name` getter ("Residu") so old data isn't lost.
WasteCategory _categoryFromMap(Map<dynamic, dynamic> map) {
  final idx = map['category_index'];
  if (idx is int && idx >= 0 && idx < WasteCategory.values.length) {
    return WasteCategory.values[idx];
  }
  final raw = map['category'];
  if (raw is String) {
    // Lowercase enum identifier
    try {
      return WasteCategory.values.byName(raw.toLowerCase());
    } catch (_) {}
    // Capitalized display name (custom .name getter)
    for (final c in WasteCategory.values) {
      if (c.name == raw) return c;
    }
  }
  return WasteCategory.lainnya;
}

/// Result of a similarity search against the local dataset.
class LocalDatasetMatch {
  final LocalDatasetEntry entry;
  final int hammingDistance;

  const LocalDatasetMatch({
    required this.entry,
    required this.hammingDistance,
  });
}

/// On-device dataset of confirmed scans. Each entry = one image file on disk
/// plus a Hive row with its perceptual hash and final category.
///
/// Lifecycle:
/// - [init] opens the Hive box and ensures `<docs>/dataset/` exists.
/// - [saveEntry] writes the JPEG + Hive row. Dedupes by SHA-256.
/// - [findMatch] returns the closest entry within a Hamming-distance threshold.
///
/// Hashing: 8x8 grayscale aHash (64-bit). Threshold default = 8 bits.
class LocalDatasetService {
  LocalDatasetService._();
  static final LocalDatasetService instance = LocalDatasetService._();

  static const String _boxName = 'local_dataset';
  static const int _defaultThreshold = 8;

  late final Box<dynamic> _box;
  late final String _docsPath;
  bool _initialized = false;

  /// Idempotent. Must be called once at app start (see `main.dart`).
  Future<void> init() async {
    if (_initialized) return;
    // Hive needs to know where to put its box files. `initFlutter` resolves
    // the app documents directory under the hood. Without this the first
    // `openBox` throws "You need to initialize Hive or provide a path".
    await Hive.initFlutter();
    _docsPath = (await getApplicationDocumentsDirectory()).path;
    await dataset_storage.ensureDatasetStorage(_docsPath);
    if (!Hive.isBoxOpen(_boxName)) {
      _box = await Hive.openBox<dynamic>(_boxName);
    } else {
      _box = Hive.box<dynamic>(_boxName);
    }
    _initialized = true;
    debugPrint('[LocalDataset] initialized with ${_box.length} entries');
  }

  /// Persist a confirmed scan. Dedupes by SHA-256 (same exact bytes →
  /// overwrite existing entry; lets manual correction update a prior entry).
  ///
  /// Returns the persisted entry, or null if [imageBytes] couldn't be hashed
  /// (decode failure).
  Future<LocalDatasetEntry?> saveEntry(
    Uint8List imageBytes, {
    required WasteCategory category,
    String? itemName,
    double? confidence,
  }) async {
    if (!_initialized) {
      debugPrint('[LocalDataset] saveEntry called before init — skipping');
      return null;
    }

    final shaHex = sha256.convert(imageBytes).toString();
    final phash = _pHash(imageBytes);
    if (phash == 0) {
      debugPrint('[LocalDataset] image decode failed — skipping save');
      return null;
    }

    // Use first 16 hex chars for filename to keep paths short.
    final imageFileName = '${shaHex.substring(0, 16)}.jpg';
    await dataset_storage.writeDatasetImage(
      _docsPath,
      imageFileName,
      imageBytes,
    );

    final entry = LocalDatasetEntry(
      sha256Hex: shaHex,
      pHash: phash,
      category: category,
      itemName: itemName,
      confidence: confidence,
      imageFileName: imageFileName,
      createdAt: DateTime.now(),
    );

    await _box.put(shaHex, entry.toMap());
    debugPrint('[LocalDataset] saved ${entry.category.name} '
        '(phash=0x${phash.toRadixString(16)}, total=${_box.length})');
    return entry;
  }

  /// Update an existing entry's category without re-writing the image file.
  /// Used by `correctResult()` so a manual correction replaces the prior
  /// classification. No-op if no entry matches [imageBytes].
  Future<void> updateCategoryForImage(
    Uint8List imageBytes, {
    required WasteCategory newCategory,
  }) async {
    if (!_initialized) return;
    final shaHex = sha256.convert(imageBytes).toString();
    final existing = _box.get(shaHex);
    if (existing == null) return;
    final map = Map<dynamic, dynamic>.from(existing as Map);
    map['category'] = newCategory.name;
    await _box.put(shaHex, map);
    debugPrint('[LocalDataset] updated entry $shaHex → ${newCategory.name}');
  }

  /// Returns the nearest match within [threshold] Hamming bits, or null.
  ///
  /// Computing the input pHash is the only decoding work; the stored pHashes
  /// are compared in-memory, so this scales well to thousands of entries.
  LocalDatasetMatch? findMatch(
    Uint8List imageBytes, {
    int threshold = _defaultThreshold,
  }) {
    if (!_initialized || _box.isEmpty) return null;
    final phash = _pHash(imageBytes);
    if (phash == 0) return null;

    LocalDatasetEntry? bestEntry;
    int bestDistance = threshold + 1;

    for (final raw in _box.values) {
      final map = raw as Map<dynamic, dynamic>;
      final entry = LocalDatasetEntry.fromMap(map);
      final distance = _hammingDistance(phash, entry.pHash);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestEntry = entry;
      }
    }

    if (bestEntry == null) return null;
    return LocalDatasetMatch(entry: bestEntry, hammingDistance: bestDistance);
  }

  /// All entries, newest first. Useful for future admin/debug UIs.
  List<LocalDatasetEntry> allEntries() {
    if (!_initialized) return const [];
    return _box.values
        .map((raw) => LocalDatasetEntry.fromMap(raw as Map<dynamic, dynamic>))
        .toList()
      ..sort((a, b) => b.createdAt.compareTo(a.createdAt));
  }

  /// Compute an 8x8 grayscale aHash (64-bit). Returns 0 on decode failure.
  ///
  /// Bit i (i = 0..63) is 1 if pixel[i] > mean(pixel[0..63]).
  int _pHash(Uint8List imageBytes) {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return 0;

    final small = img.copyResize(decoded, width: 8, height: 8);
    final lumens = List<int>.filled(64, 0);
    var sum = 0;
    for (var y = 0; y < 8; y++) {
      for (var x = 0; x < 8; x++) {
        final p = small.getPixel(x, y);
        // Rec. 601 luma — robust to color shifts.
        final luma = (0.299 * p.r + 0.587 * p.g + 0.114 * p.b).round();
        lumens[y * 8 + x] = luma;
        sum += luma;
      }
    }
    final mean = sum ~/ 64;

    var hash = 0;
    for (var i = 0; i < 64; i++) {
      if (lumens[i] > mean) hash |= (1 << i);
    }
    return hash;
  }

  /// Bit-count of XOR — number of differing bits between two 64-bit hashes.
  int _hammingDistance(int a, int b) {
    var x = a ^ b;
    var count = 0;
    while (x != 0) {
      count += x & 1;
      x = x >>> 1;
    }
    return count;
  }
}
