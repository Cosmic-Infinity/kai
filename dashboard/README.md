# kai_dashboard

Flutter UI for the **kai** system (Android / Windows). Shows camera occupancy, requests on-demand room checks, and sends manual power overrides.

- Camera list, images and settings come from the HTTP API Server (`http://<host>:8000`, `X-API-Key` header).
- Real-time actions and telemetry use MQTT (`<host>:1883`) with the dashboard account created by `configure_kai.py`.

See the [root README](../README.md) for setup and architecture.

```bash
flutter pub get
flutter run
```
