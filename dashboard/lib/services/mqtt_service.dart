import 'dart:io';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

class MqttService {
  final String host;
  final int port;
  final String username;
  final String password;
  final String clientId;

  MqttServerClient? _client;
  bool _isConnected = false;

  // Topics
  static const String topicForceRequest = 'kai/force_request';
  static const String topicForceServed = 'kai/force_served';
  static const String topicControl = 'kai/control';
  static const String topicPower = 'kai/power';

  // Callbacks
  void Function(String message)? onForceServedReceived;
  void Function(String message)? onPowerReceived;
  void Function()? onConnectedCallback;
  void Function()? onDisconnectedCallback;

  MqttService({
    required this.host,
    this.port = 1883,
    required this.username,
    required this.password,
    String? clientId,
  }) : clientId = clientId ?? 'kai_flutter_dashboard_${DateTime.now().millisecondsSinceEpoch}';

  bool get isConnected => _isConnected;

  Future<bool> connect() async {
    _client = MqttServerClient.withPort(host, clientId, port);
    _client!.logging(on: false);
    _client!.keepAlivePeriod = 120;
    _client!.onConnected = _onConnected;
    _client!.onDisconnected = _onDisconnected;
    _client!.onSubscribed = _onSubscribed;
    _client!.pongCallback = _pong;
    _client!.autoReconnect = true;
    _client!.resubscribeOnAutoReconnect = true;

    final connMessage = MqttConnectMessage()
        .withClientIdentifier(clientId)
        .authenticateAs(username, password)
        .startClean()
        .withWillQos(MqttQos.atLeastOnce);
    _client!.connectionMessage = connMessage;

    try {
      await _client!.connect();
    } catch (e) {
      print('[MqttService] Connection failed: $e');
      _client!.disconnect();
      return false;
    }

    if (_client!.connectionStatus!.state == MqttConnectionState.connected) {
      _isConnected = true;
      _setupListeners();
      _subscribeToTopics();
      return true;
    } else {
      print('[MqttService] Connection failed - status: ${_client!.connectionStatus}');
      _client!.disconnect();
      return false;
    }
  }

  void disconnect() {
    _client?.disconnect();
    _isConnected = false;
  }

  void _onConnected() {
    print('[MqttService] Connected successfully');
    _isConnected = true;
    onConnectedCallback?.call();
  }

  void _onDisconnected() {
    print('[MqttService] Disconnected');
    _isConnected = false;
    onDisconnectedCallback?.call();
  }

  void _onSubscribed(String topic) {
    print('[MqttService] Subscribed to topic: $topic');
  }

  void _pong() {
    // Keep-alive pong received
  }

  void _subscribeToTopics() {
    if (!_isConnected || _client == null) return;
    _client!.subscribe(topicForceServed, MqttQos.atLeastOnce);
    _client!.subscribe(topicPower, MqttQos.atLeastOnce);
  }

  void _setupListeners() {
    if (_client == null) return;
    _client!.updates!.listen((List<MqttReceivedMessage<MqttMessage>> c) {
      final MqttPublishMessage recMess = c[0].payload as MqttPublishMessage;
      final String pt = MqttPublishPayload.bytesToStringAsString(recMess.payload.message);
      final String topic = c[0].topic;

      if (topic == topicForceServed) {
        onForceServedReceived?.call(pt);
      } else if (topic == topicPower) {
        onPowerReceived?.call(pt);
      }
    });
  }

  /// Publishes a force update request for a camera
  void publishForceUpdateRequest(String cameraId) {
    if (!_isConnected || _client == null) return;
    final builder = MqttClientPayloadBuilder();
    builder.addString('FORCE_UPDATE_$cameraId');
    _client!.publishMessage(topicForceRequest, MqttQos.atLeastOnce, builder.payload!);
    print('[MqttService] Published force update request for $cameraId');
  }

  /// Publishes a power command toggle for a camera
  void publishControlCommand(String cameraId, String state) {
    if (!_isConnected || _client == null) return;
    final builder = MqttClientPayloadBuilder();
    builder.addString('SET_${cameraId}_${state.toUpperCase()}');
    _client!.publishMessage(topicControl, MqttQos.atLeastOnce, builder.payload!);
    print('[MqttService] Published control command SET_${cameraId}_$state');
  }
}
