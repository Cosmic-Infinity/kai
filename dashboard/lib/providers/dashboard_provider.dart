import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import '../models/camera.dart';
import '../services/api_service.dart';
import '../services/mqtt_service.dart';

class DashboardProvider with ChangeNotifier {
  // Connection states
  String _host = '';
  String _username = '';
  String _password = '';
  String _apiKey = '';
  ThemeMode _themeMode = ThemeMode.dark;

  String get host => _host;
  String get username => _username;
  String get password => _password;
  String get apiKey => _apiKey;
  ThemeMode get themeMode => _themeMode;

  ApiService? _apiService;
  MqttService? _mqttService;

  // Camera states
  Map<String, Camera> _cameras = {};
  Map<String, String> _powerStates = {}; // Local memory for power state per camera id
  final Set<String> _pendingForceUpdates = {};

  Map<String, Camera> get cameras => _cameras;
  Set<String> get pendingForceUpdates => _pendingForceUpdates;

  // Filter & Search states
  String _searchQuery = '';
  String _currentFilter = 'All';
  int _gridCols = 3;

  String get searchQuery => _searchQuery;
  String get currentFilter => _currentFilter;
  int get gridCols => _gridCols;

  // Timing/Refresh states
  int _refreshInterval = 30;
  int _elapsed = 0;
  Timer? _timer;

  int get refreshInterval => _refreshInterval;
  int get remainingSeconds => (_refreshInterval - _elapsed).clamp(0, _refreshInterval);

  bool _isMqttConnected = false;
  bool get isMqttConnected => _isMqttConnected;

  // Initialize and load saved config on app launch
  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    _host = prefs.getString('host') ?? '';
    _username = prefs.getString('username') ?? '';
    _password = prefs.getString('password') ?? '';
    _apiKey = prefs.getString('apiKey') ?? '';
    final themeStr = prefs.getString('theme') ?? 'Dark';
    _themeMode = themeStr == 'Light' ? ThemeMode.light : ThemeMode.dark;
    notifyListeners();
  }

  /// Perform login probe check
  Future<bool> login(String host, String username, String password, String apiKey) async {
    final api = ApiService(host: host, apiKey: apiKey);
    final config = await api.fetchConfig();

    if (config != null) {
      _host = host;
      _username = username;
      _password = password;
      _apiKey = apiKey;

      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('host', host);
      await prefs.setString('username', username);
      await prefs.setString('password', password);
      await prefs.setString('apiKey', apiKey);

      _refreshInterval = config['DASHBOARD_INTERVAL'] as int? ?? 30;
      _apiService = api;

      notifyListeners();
      return true;
    }
    return false;
  }

  /// Start background dashboard routines (HTTP fetching and MQTT broker setup)
  Future<void> startDashboard() async {
    _mqttService?.disconnect();
    if (_apiService == null) {
      _apiService = ApiService(host: _host, apiKey: _apiKey);
    }

    // Connect to MQTT
    _mqttService = MqttService(
      host: _host,
      username: _username,
      password: _password,
    );

    _mqttService!.onConnectedCallback = () {
      _isMqttConnected = true;
      notifyListeners();
    };

    _mqttService!.onDisconnectedCallback = () {
      _isMqttConnected = false;
      _pendingForceUpdates.clear();
      notifyListeners();
    };

    _mqttService!.onForceServedReceived = (msg) {
      if (msg.startsWith('FORCE_SERVED_')) {
        final camId = msg.replaceAll('FORCE_SERVED_', '');
        if (_pendingForceUpdates.contains(camId)) {
          _pendingForceUpdates.remove(camId);
          print('[DashboardProvider] Force update served for $camId via MQTT');
          // Reload specifically this camera or refresh all
          loadCameras(updatedCams: {camId});
        }
      }
    };

    _mqttService!.onPowerReceived = (msg) {
      // Expecting payload: <CAM_ID>_ON or <CAM_ID>_OFF
      final parts = msg.split('_');
      if (parts.length >= 3) {
        final camId = '${parts[0]}_${parts[1]}'; // CAM_X
        final state = parts[2];
        _powerStates[camId] = state;
        if (_cameras.containsKey(camId)) {
          _cameras[camId] = _cameras[camId]!.copyWith(powerState: state);
          notifyListeners();
        }
      }
    };

    await _mqttService!.connect();

    // Initial load
    await loadCameras();

    // Start UI elapsed timer
    _elapsed = 0;
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _elapsed++;
      if (_elapsed >= _refreshInterval) {
        loadCameras();
        _elapsed = 0;
      }
      notifyListeners();
    });
  }

  /// Stop dashboard operations
  void stopDashboard() {
    _timer?.cancel();
    _mqttService?.disconnect();
    _cameras.clear();
    _pendingForceUpdates.clear();
    _elapsed = 0;
  }

  /// Perform configuration logout
  Future<void> logout() async {
    stopDashboard();
    _host = '';
    _username = '';
    _password = '';
    _apiKey = '';

    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('host');
    await prefs.remove('username');
    await prefs.remove('password');
    await prefs.remove('apiKey');
    notifyListeners();
  }

  /// Fetches camera properties from REST API
  Future<void> loadCameras({Set<String>? updatedCams}) async {
    if (_apiService == null) return;
    final fetched = await _apiService!.fetchCameras(_powerStates);
    if (fetched != null) {
      if (updatedCams == null) {
        _cameras = fetched;
        // Seed our powerStates memory if not present
        fetched.forEach((camId, cam) {
          _powerStates.putIfAbsent(camId, () => cam.powerState);
        });
      } else {
        // Only update specific cameras so we don't reload network textures for all
        fetched.forEach((camId, cam) {
          if (updatedCams.contains(camId) || !_cameras.containsKey(camId)) {
            _cameras[camId] = cam;
            _powerStates[camId] = cam.powerState;
          }
        });
      }
      notifyListeners();
    }
  }

  /// Triggers a MQTT Force Update
  void requestForceUpdate(String cameraId) {
    _pendingForceUpdates.add(cameraId);
    _mqttService?.publishForceUpdateRequest(cameraId);
    notifyListeners();

    // Auto-timeout after 12 seconds to prevent infinite load spinner if packet is lost or server is slow
    Timer(const Duration(seconds: 12), () {
      if (_pendingForceUpdates.contains(cameraId)) {
        _pendingForceUpdates.remove(cameraId);
        notifyListeners();
        print('[DashboardProvider] Force update timed out for $cameraId');
      }
    });
  }

  /// Triggers a MQTT power toggle command
  void togglePowerState(String cameraId, String currentState) {
    final newState = currentState == 'ON' ? 'OFF' : 'ON';
    _powerStates[cameraId] = newState;
    if (_cameras.containsKey(cameraId)) {
      _cameras[cameraId] = _cameras[cameraId]!.copyWith(powerState: newState);
    }
    _mqttService?.publishControlCommand(cameraId, newState);
    notifyListeners();
  }

  /// Filters cameras based on current criteria and search query
  List<Camera> get filteredCameras {
    List<Camera> list = _cameras.values.toList()..sort((a, b) => a.id.compareTo(b.id));

    if (_searchQuery.isNotEmpty) {
      list = list.where((c) => c.id.toLowerCase().contains(_searchQuery.toLowerCase())).toList();
    }

    final filter = _currentFilter;
    if (filter != 'All') {
      list = list.where((c) {
        final isOccupied = c.status == 'YES';
        final isPowerOn = c.powerState == 'ON';

        switch (filter) {
          case 'Occupied':
            return isOccupied;
          case 'Empty':
            return !isOccupied;
          case 'Power ON':
            return isPowerOn;
          case 'Power OFF':
            return !isPowerOn;
          case 'Occupied & Power ON':
            return isOccupied && isPowerOn;
          case 'Occupied & Power OFF':
            return isOccupied && !isPowerOn;
          case 'Empty & Power ON':
            return !isOccupied && isPowerOn;
          case 'Empty & Power OFF':
            return !isOccupied && !isPowerOn;
          default:
            return true;
        }
      }).toList();
    }

    return list;
  }

  void updateSearchQuery(String query) {
    _searchQuery = query;
    notifyListeners();
  }

  void updateFilter(String filter) {
    _currentFilter = filter;
    notifyListeners();
  }

  void cycleGridCols({required int maxCols}) {
    if (maxCols <= 1) {
      _gridCols = 1;
    } else {
      int current = _gridCols.clamp(1, maxCols);
      _gridCols = (current % maxCols) + 1;
    }
    notifyListeners();
  }

  void setGridCols(int cols) {
    _gridCols = cols;
    notifyListeners();
  }

  void forceRefresh() {
    loadCameras();
    _elapsed = 0;
    if (!_isMqttConnected) {
      print('[DashboardProvider] Manual refresh detected MQTT disconnected. Reconnecting...');
      startDashboard();
    }
    notifyListeners();
  }

  /// Toggle Theme mode locally and in shared preferences
  Future<void> toggleTheme() async {
    _themeMode = _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('theme', _themeMode == ThemeMode.light ? 'Light' : 'Dark');
    notifyListeners();
  }

  /// Fetches config from API (used for Settings configuration dialogue)
  Future<Map<String, dynamic>?> fetchSystemConfig() async {
    return await _apiService?.fetchConfig();
  }

  /// Updates backend configuration settings via POST request
  Future<bool> saveSystemSettings(Map<String, int> settings) async {
    if (_apiService == null) return false;
    try {
      final headers = {
        'Content-Type': 'application/json',
      };
      if (_apiKey.isNotEmpty) {
        headers['X-API-Key'] = _apiKey;
      }
      final response = await http.post(
        Uri.parse('${_apiService!.baseUrl}/api/config'),
        headers: headers,
        body: jsonEncode(settings),
      );
      if (response.statusCode == 200) {
        if (settings.containsKey('DASHBOARD_INTERVAL')) {
          _refreshInterval = settings['DASHBOARD_INTERVAL']!;
          _elapsed = 0;
        }
        notifyListeners();
        return true;
      }
    } catch (e) {
      print('[DashboardProvider] Failed to post configuration: $e');
    }
    return false;
  }
}
