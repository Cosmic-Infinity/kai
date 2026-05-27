import 'dart:async';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/camera.dart';
import '../services/api_service.dart';
import '../services/mqtt_service.dart';

class DashboardProvider with ChangeNotifier {
  String _host = '';
  String _username = '';
  String _password = '';
  String _apiKey = '';
  ThemeMode _themeMode = ThemeMode.system;

  String get host => _host;
  String get username => _username;
  String get password => _password;
  String get apiKey => _apiKey;
  ThemeMode get themeMode => _themeMode;

  ApiService? _apiService;
  MqttService? _mqttService;

  Map<String, Camera> _cameras = {};
  // Power states are tracked locally so MQTT updates survive HTTP refreshes
  final Map<String, String> _powerStates = {};
  final Set<String> _pendingForceUpdates = {};

  Map<String, Camera> get cameras => _cameras;
  Set<String> get pendingForceUpdates => _pendingForceUpdates;

  String _searchQuery = '';
  final Set<String> _selectedFilters = {};
  int _gridCols = 0; // 0 means unset / auto

  String get searchQuery => _searchQuery;
  Set<String> get selectedFilters => _selectedFilters;
  int get gridCols => _gridCols;

  String get filterLabel {
    if (_selectedFilters.isEmpty) return 'All';
    return _selectedFilters.join(' + ');
  }

  int _refreshInterval = 30;
  int _elapsed = 0;
  Timer? _timer;

  int get refreshInterval => _refreshInterval;
  int get remainingSeconds => (_refreshInterval - _elapsed).clamp(0, _refreshInterval);
  
  // ValueNotifier for UI timer updates without full widget tree rebuilds
  final ValueNotifier<int> remainingSecondsNotifier = ValueNotifier(0);

  bool _isMqttConnected = false;
  bool get isMqttConnected => _isMqttConnected;

  /// Load persisted credentials and theme on app launch
  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    _host = prefs.getString('host') ?? '';
    _username = prefs.getString('username') ?? '';
    _password = prefs.getString('password') ?? '';
    _apiKey = prefs.getString('apiKey') ?? '';
    final themeStr = prefs.getString('theme') ?? 'System';
    _themeMode = switch (themeStr) {
      'Light' => ThemeMode.light,
      'Dark' => ThemeMode.dark,
      _ => ThemeMode.system,
    };
    _gridCols = prefs.getInt('gridCols') ?? 0;
    notifyListeners();
  }

  /// Probe the server with the given credentials; persist on success
  Future<bool> login(String host, String username, String password, String apiKey) async {
    final api = ApiService(host: host, apiKey: apiKey);
    final config = await api.fetchConfig();
    if (config == null) return false;

    _host = host;
    _username = username;
    _password = password;
    _apiKey = apiKey;
    _refreshInterval = config['DASHBOARD_INTERVAL'] as int? ?? 30;
    _apiService = api;

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('host', host);
    await prefs.setString('username', username);
    await prefs.setString('password', password);
    await prefs.setString('apiKey', apiKey);

    notifyListeners();
    return true;
  }

  /// Start (or restart) background MQTT + HTTP polling
  Future<void> startDashboard() async {
    // Tear down any existing connection cleanly before creating a new one
    _mqttService?.disconnect();
    _mqttService = null;

    _apiService ??= ApiService(host: _host, apiKey: _apiKey);

    // Fetch latest config to synchronize _refreshInterval on startup/resume/reconnect
    final config = await _apiService!.fetchConfig();
    if (config != null && config.containsKey('DASHBOARD_INTERVAL')) {
      _refreshInterval = config['DASHBOARD_INTERVAL'] as int? ?? 30;
    }

    _mqttService = MqttService(host: _host, username: _username, password: _password);

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
      if (!msg.startsWith('FORCE_SERVED_')) return;
      final camId = msg.substring('FORCE_SERVED_'.length);
      if (_pendingForceUpdates.remove(camId)) {
        print('[DashboardProvider] Force update served for $camId');
        loadCameras(updatedCams: {camId});
      }
    };

    _mqttService!.onPowerReceived = (msg) {
      // Payload format: CAM_X_ON or CAM_X_OFF
      final lastUnderscore = msg.lastIndexOf('_');
      if (lastUnderscore <= 0) return;
      final camId = msg.substring(0, lastUnderscore);
      final state = msg.substring(lastUnderscore + 1);
      _powerStates[camId] = state;
      if (_cameras.containsKey(camId)) {
        _cameras[camId] = _cameras[camId]!.copyWith(powerState: state);
        notifyListeners();
      }
    };

    await _mqttService!.connect();
    await loadCameras();

    _elapsed = 0;
    remainingSecondsNotifier.value = remainingSeconds;
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      _elapsed++;
      if (_elapsed >= _refreshInterval) {
        loadCameras();
        _elapsed = 0;
      }
      remainingSecondsNotifier.value = remainingSeconds;
      // Do NOT call notifyListeners() here to avoid rebuilding the entire dashboard every second
    });
  }

  /// Stop all background operations (called on logout)
  void stopDashboard() {
    _timer?.cancel();
    _timer = null;
    _mqttService?.disconnect();
    _mqttService = null;
    _cameras.clear();
    _pendingForceUpdates.clear();
    _isMqttConnected = false;
    _elapsed = 0;
  }

  /// Clear credentials and return to login
  Future<void> logout() async {
    stopDashboard();
    _host = '';
    _username = '';
    _password = '';
    _apiKey = '';
    _apiService = null;

    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('host');
    await prefs.remove('username');
    await prefs.remove('password');
    await prefs.remove('apiKey');
    notifyListeners();
  }

  /// Fetch camera states from REST API.
  /// If [updatedCams] is provided, only those cameras are replaced (avoids
  /// invalidating cached network images for unrelated cameras).
  Future<void> loadCameras({Set<String>? updatedCams}) async {
    if (_apiService == null) return;

    // Run both network requests concurrently to optimize performance and sync settings
    final results = await Future.wait([
      _apiService!.fetchConfig(),
      _apiService!.fetchCameras(_powerStates),
    ]);

    final Map<String, dynamic>? config = results[0];
    final fetched = results[1] as Map<String, Camera>?;

    if (config != null && config.containsKey('DASHBOARD_INTERVAL')) {
      final newInterval = config['DASHBOARD_INTERVAL'] as int? ?? 30;
      if (_refreshInterval != newInterval) {
        _refreshInterval = newInterval;
        if (_elapsed >= _refreshInterval) {
          _elapsed = 0;
        }
      }
    }

    if (fetched == null) return;

    if (updatedCams == null) {
      _cameras = fetched;
      for (final entry in fetched.entries) {
        _powerStates.putIfAbsent(entry.key, () => entry.value.powerState);
      }
    } else {
      for (final entry in fetched.entries) {
        if (updatedCams.contains(entry.key) || !_cameras.containsKey(entry.key)) {
          _cameras[entry.key] = entry.value;
          _powerStates[entry.key] = entry.value.powerState;
        }
      }
    }
    notifyListeners();
  }

  /// Publish a force-capture request via MQTT; auto-cancels spinner after 12 s
  void requestForceUpdate(String cameraId) {
    _pendingForceUpdates.add(cameraId);
    _mqttService?.publishForceUpdateRequest(cameraId);
    notifyListeners();

    Timer(const Duration(seconds: 12), () {
      if (_pendingForceUpdates.remove(cameraId)) {
        print('[DashboardProvider] Force update timed out for $cameraId');
        notifyListeners();
      }
    });
  }

  /// Optimistically update power state locally, then publish MQTT command
  void togglePowerState(String cameraId, String currentState) {
    final newState = currentState == 'ON' ? 'OFF' : 'ON';
    _powerStates[cameraId] = newState;
    if (_cameras.containsKey(cameraId)) {
      _cameras[cameraId] = _cameras[cameraId]!.copyWith(powerState: newState);
    }
    _mqttService?.publishControlCommand(cameraId, newState);
    notifyListeners();
  }

  /// Force an immediate camera image refresh (also reconnects if MQTT is down)
  void forceRefresh() {
    _elapsed = 0;
    remainingSecondsNotifier.value = remainingSeconds;
    loadCameras(); // async, will notifyListeners on completion
    if (!_isMqttConnected) {
      print('[DashboardProvider] Manual refresh — MQTT offline, reconnecting...');
      startDashboard();
    }
  }

  List<Camera> get filteredCameras {
    var list = _cameras.values.toList()..sort((a, b) => a.id.compareTo(b.id));

    if (_searchQuery.isNotEmpty) {
      list = list.where((c) => c.id.toLowerCase().contains(_searchQuery.toLowerCase())).toList();
    }

    if (_selectedFilters.isNotEmpty) {
      list = list.where((c) {
        // Occupancy filtering
        final hasOccupied = _selectedFilters.contains('Occupied');
        final hasEmpty = _selectedFilters.contains('Empty');
        if (hasOccupied && !hasEmpty) {
          if (c.status != 'YES') return false;
        } else if (hasEmpty && !hasOccupied) {
          if (c.status != 'NO') return false;
        }

        // Power state filtering
        final hasPowerOn = _selectedFilters.contains('Power ON');
        final hasPowerOff = _selectedFilters.contains('Power OFF');
        if (hasPowerOn && !hasPowerOff) {
          if (c.powerState != 'ON') return false;
        } else if (hasPowerOff && !hasPowerOn) {
          if (c.powerState != 'OFF') return false;
        }

        return true;
      }).toList();
    }

    return list;
  }

  void updateSearchQuery(String query) {
    _searchQuery = query;
    notifyListeners();
  }

  void toggleFilter(String filter) {
    if (_selectedFilters.contains(filter)) {
      _selectedFilters.remove(filter);
    } else {
      _selectedFilters.add(filter);
    }
    notifyListeners();
  }

  void clearFilters() {
    _selectedFilters.clear();
    notifyListeners();
  }

  void setQuickFilter(List<String> filters) {
    _selectedFilters.clear();
    _selectedFilters.addAll(filters);
    notifyListeners();
  }

  void cycleGridCols({required int maxCols}) {
    if (maxCols <= 1) {
      _gridCols = 1;
    } else {
      int currentCols = _gridCols == 0 ? (maxCols >= 3 ? 3 : 1) : _gridCols;
      _gridCols = (currentCols.clamp(1, maxCols) % maxCols) + 1;
    }
    SharedPreferences.getInstance().then((prefs) {
      prefs.setInt('gridCols', _gridCols);
    });
    notifyListeners();
  }

  void setGridCols(int cols) {
    _gridCols = cols;
    SharedPreferences.getInstance().then((prefs) {
      prefs.setInt('gridCols', _gridCols);
    });
    notifyListeners();
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    final prefs = await SharedPreferences.getInstance();
    final themeStr = switch (mode) {
      ThemeMode.light => 'Light',
      ThemeMode.dark => 'Dark',
      ThemeMode.system => 'System',
    };
    await prefs.setString('theme', themeStr);
    notifyListeners();
  }

  Future<Map<String, dynamic>?> fetchSystemConfig() async {
    final config = await _apiService?.fetchConfig();
    if (config != null && config.containsKey('DASHBOARD_INTERVAL')) {
      final newInterval = config['DASHBOARD_INTERVAL'] as int? ?? 30;
      if (_refreshInterval != newInterval) {
        _refreshInterval = newInterval;
        if (_elapsed >= _refreshInterval) {
          _elapsed = 0;
        }
        remainingSecondsNotifier.value = remainingSeconds;
        notifyListeners();
      }
    }
    return config;
  }

  Future<bool> saveSystemSettings(Map<String, int> settings) async {
    final success = await (_apiService?.saveConfig(settings) ?? Future.value(false));
    if (success && settings.containsKey('DASHBOARD_INTERVAL')) {
      _refreshInterval = settings['DASHBOARD_INTERVAL']!;
      _elapsed = 0;
      notifyListeners();
    }
    return success;
  }
}
