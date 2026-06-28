import 'dart:io';
import 'dart:typed_data';

Future<void> ensureDatasetStorage(String docsPath) async {
  final dir = Directory('$docsPath/dataset');
  if (!dir.existsSync()) {
    dir.createSync(recursive: true);
  }
}

Future<void> writeDatasetImage(
  String docsPath,
  String fileName,
  Uint8List imageBytes,
) async {
  await File('$docsPath/dataset/$fileName')
      .writeAsBytes(imageBytes, flush: true);
}
