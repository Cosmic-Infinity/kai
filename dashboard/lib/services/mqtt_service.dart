import 'package:flutter/foundation.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

class MqttService {
  final String host;
  final int port;
  final String username;
  final String password;
  final String clientId;

  MqttServerClient? _client;

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
  }) : clientId = clientId ?? 'kai_flutter_${DateTime.now().millisecondsSinceEpoch}';

  bool get isConnected =>
      _client?.connectionStatus?.state == MqttConnectionState.connected;

  Future<bool> connect() async {
    _client = MqttServerClient.withPort(host, clientId, port);
    _client!.logging(on: false);
    _client!.keepAlivePeriod = 60;
    _client!.autoReconnect = true;
    _client!.resubscribeOnAutoReconnect = true;
    _client!.onConnected = _onConnected;
    _client!.onDisconnected = _onDisconnected;

    _client!.connectionMessage = MqttConnectMessage()
        .withClientIdentifier(clientId)
        .authenticateAs(username, password)
        .startClean()
        .withWillQos(MqttQos.atLeastOnce);

    try {
      await _client!.connect();
    } catch (e) {
      debugPrint('[MqttService] Connection failed: $e');
      _safeDisconnect();
      return false;
    }

    if (_client!.connectionStatus!.state == MqttConnectionState.connected) {
      _setupListeners();
      _subscribeToTopics();
      return true;
    } else {
      debugPrint('[MqttService] Connection failed - status: ${_client!.connectionStatus}');
      _safeDisconnect();
      return false;
    }
  }

  void disconnect() => _safeDisconnect();

  void _safeDisconnect() {
    try {
      _client?.disconnect();
    } catch (_) {
      // Socket may already be dead — safe to ignore
    }
    _client = null;
  }

  void _onConnected() {
    debugPrint('[MqttService] Connected');
    onConnectedCallback?.call();
  }

  void _onDisconnected() {
    debugPrint('[MqttService] Disconnected');
    onDisconnectedCallback?.call();
  }

  void _subscribeToTopics() {
    _client?.subscribe(topicForceServed, MqttQos.atLeastOnce);
    _client?.subscribe(topicPower, MqttQos.atLeastOnce);
  }

  void _setupListeners() {
    _client!.updates!.listen(
      (List<MqttReceivedMessage<MqttMessage>> c) {
        final recMess = c[0].payload as MqttPublishMessage;
        final payload = MqttPublishPayload.bytesToStringAsString(recMess.payload.message);
        final topic = c[0].topic;

        if (topic == topicForceServed) {
          onForceServedReceived?.call(payload);
        } else if (topic == topicPower) {
          onPowerReceived?.call(payload);
        }
      },
      onError: (Object error) {
        // Socket may die silently when app is backgrounded on Android.
        // The lifecycle observer will force-reconnect on resume, so just log here.
        debugPrint('[MqttService] Stream error (socket likely dead): $error');
      },
      cancelOnError: false,
    );
  }

  /// Publishes a force update request for a camera. Returns false if it could not be sent.
  bool publishForceUpdateRequest(String cameraId) {
    return _publish(topicForceRequest, 'FORCE_UPDATE_$cameraId');
  }

  /// Publishes a power toggle command for a camera. Returns false if it could not be sent.
  bool publishControlCommand(String cameraId, String state) {
    return _publish(topicControl, 'SET_${cameraId}_${state.toUpperCase()}');
  }

  bool _publish(String topic, String payload) {
    if (!isConnected || _client == null) {
      debugPrint('[MqttService] Not connected - dropped $payload');
      return false;
    }
    try {
      final builder = MqttClientPayloadBuilder()..addString(payload);
      _client!.publishMessage(topic, MqttQos.atLeastOnce, builder.payload!);
      debugPrint('[MqttService] $topic → $payload');
      return true;
    } catch (e) {
      // Socket can die between the state check and the write
      debugPrint('[MqttService] Publish failed for $payload: $e');
      return false;
    }
  }
}
