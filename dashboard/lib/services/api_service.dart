import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/camera.dart';

class ApiService {
  final String host;
  final int port;
  final String? apiKey;

  ApiService({required this.host, this.port = 8000, this.apiKey});

  String get baseUrl => 'http://$host:$port';

  Map<String, String> get _headers {
    final headers = <String, String>{};
    if (apiKey != null && apiKey!.isNotEmpty) {
      headers['X-API-Key'] = apiKey!;
    }
    return headers;
  }

  /// Probe check to see if server is online and valid config exists
  Future<Map<String, dynamic>?> fetchConfig() async {
    try {
      final url = Uri.parse('$baseUrl/api/config');
      final response = await http.get(url, headers: _headers).timeout(const Duration(seconds: 4));
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
    } catch (e) {
      print('[ApiService] Error fetching config: $e');
    }
    return null;
  }

  /// Fetches camera properties and statuses
  Future<Map<String, Camera>?> fetchCameras(Map<String, String> currentPowerStates) async {
    try {
      final url = Uri.parse('$baseUrl/api/cameras');
      final response = await http.get(url, headers: _headers).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final Map<String, Camera> cameras = {};
        
        data.forEach((camId, jsonMap) {
          final powerState = currentPowerStates[camId] ?? 'ON';
          final relativePath = jsonMap['image_path'] as String? ?? '';
          final timestamp = DateTime.now().millisecondsSinceEpoch;
          final absolutePath = relativePath.isNotEmpty 
              ? '$baseUrl$relativePath?ts=$timestamp' 
              : '';
          
          final Map<String, dynamic> updatedJson = Map<String, dynamic>.from(jsonMap);
          updatedJson['image_path'] = absolutePath;

          cameras[camId] = Camera.fromJson(
            camId,
            updatedJson,
            powerState: powerState,
          );
        });
        return cameras;
      }
    } catch (e) {
      print('[ApiService] Error fetching cameras: $e');
    }
    return null;
  }
}
