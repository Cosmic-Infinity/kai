import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/camera.dart';

class ApiService {
  final String host;
  final int port;
  final String? apiKey;

  ApiService({required this.host, this.port = 8000, this.apiKey});

  /// Human-readable reason for the most recent failed request (null after a success)
  String? lastError;

  String _describeFailure(http.Response response) {
    if (response.statusCode == 401) {
      return 'Server rejected the API key (401 Unauthorized).';
    }
    try {
      final body = jsonDecode(response.body);
      if (body is Map && body['error'] is String) return body['error'] as String;
    } catch (_) {
      // Body wasn't JSON - fall through to the status code
    }
    return 'Server returned HTTP ${response.statusCode}.';
  }

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
        lastError = null;
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      lastError = _describeFailure(response);
    } catch (e) {
      lastError = 'Could not reach server: $e';
      debugPrint('[ApiService] fetchConfig error: $e');
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
        lastError = null;
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
            // Locally tracked state (kept current by MQTT) wins; otherwise take what the
            // server last recorded, so a fresh start shows cameras the system already powered off.
            Camera.fromJson(
              camId,
              updatedJson,
              powerState: currentPowerStates[camId] ?? updatedJson['power_state'] as String? ?? 'ON',
            ),
          );
        });
      }
      lastError = _describeFailure(response);
    } catch (e) {
      lastError = 'Could not reach server: $e';
      debugPrint('[ApiService] fetchCameras error: $e');
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
      if (response.statusCode == 200) {
        lastError = null;
        return true;
      }
      lastError = _describeFailure(response);
      return false;
    } catch (e) {
      lastError = 'Could not reach server: $e';
      debugPrint('[ApiService] saveConfig error: $e');
      return false;
    }
  }
}
