import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/camera.dart';

class ApiService {
  final String host;
  final int port;
  final String? apiKey;

  ApiService({required this.host, this.port = 8000, this.apiKey});

  String get baseUrl => 'http://$host:$port';

  Map<String, String> get _headers => {
    if (apiKey != null && apiKey!.isNotEmpty) 'X-API-Key': apiKey!,
  };

  /// Probe check — fetches current system config from server
  Future<Map<String, dynamic>?> fetchConfig() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/config'), headers: _headers)
          .timeout(const Duration(seconds: 4));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
    } catch (e) {
      print('[ApiService] fetchConfig error: $e');
    }
    return null;
  }

  /// Fetches all camera states from the REST API
  Future<Map<String, Camera>?> fetchCameras(Map<String, String> currentPowerStates) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/cameras'), headers: _headers)
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return data.map((camId, jsonMap) {
          // Rewrite relative image path to absolute URL with cache-bust timestamp
          final relativePath = (jsonMap as Map<String, dynamic>)['image_path'] as String? ?? '';
          final absolutePath = relativePath.isNotEmpty
              ? '$baseUrl$relativePath?ts=${DateTime.now().millisecondsSinceEpoch}'
              : '';
          final updatedJson = {...jsonMap, 'image_path': absolutePath};
          return MapEntry(
            camId,
            Camera.fromJson(camId, updatedJson, powerState: currentPowerStates[camId] ?? 'ON'),
          );
        });
      }
    } catch (e) {
      print('[ApiService] fetchCameras error: $e');
    }
    return null;
  }

  /// Posts updated system configuration to the server
  Future<bool> saveConfig(Map<String, int> settings) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/config'),
        headers: {..._headers, 'Content-Type': 'application/json'},
        body: jsonEncode(settings),
      );
      return response.statusCode == 200;
    } catch (e) {
      print('[ApiService] saveConfig error: $e');
      return false;
    }
  }
}
