import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app.dart';
import 'core/services/local_dataset_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SharedPreferences.getInstance();
  await dotenv.load(fileName: '.env');
  // Open the local-scan-dataset box + ensure image dir exists before UI mounts
  // so the first scan can read/write without a startup race.
  await LocalDatasetService.instance.init();
  runApp(const ProviderScope(child: MyApp()));
}

