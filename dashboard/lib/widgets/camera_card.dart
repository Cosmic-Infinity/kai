import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter_cache_manager/flutter_cache_manager.dart';
import '../models/camera.dart';
import '../providers/dashboard_provider.dart';
import 'fullscreen_preview.dart';

// ─── Cache Manager for transient camera feeds ────────────────────────────────
final cameraCacheManager = CacheManager(
  Config(
    'camera_feed_cache',
    stalePeriod: const Duration(hours: 6), // Keep until max objects limit is hit
    maxNrOfCacheObjects: 128, // Keep disk usage very low
  ),
);
// ─────────────────────────────────────────────────────────────────────────────

// ─── Design tokens shared across card states ──────────────────────────────────
const _colorSuccess = Color(0xFF10B981); // emerald
const _colorDanger = Color(0xFFF43F5E); // rose
const _colorUnknown = Color(0xFF64748B); // slate

const _badgeOccupied = Color(
  0xFF065F46,
); // deep emerald (0.92 alpha applied at use-site)
const _badgeEmpty = Color(0xFF9F1239); // deep rose
const _badgePowerOff = Color(0xFF991B1B); // dark red
const _badgeUnknown = Color(0xFF334155); // slate-700
// ─────────────────────────────────────────────────────────────────────────────

class CameraCard extends StatelessWidget {
  final Camera camera;

  const CameraCard({super.key, required this.camera});

  void _onOfflineAction(BuildContext context, DashboardProvider provider) {
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(
        SnackBar(
          content: const Row(
            children: [
              Icon(Icons.wifi_off, color: Colors.amberAccent, size: 20),
              SizedBox(width: 10),
              Expanded(
                child: Text('MQTT disconnected. Attempting to reconnect...'),
              ),
            ],
          ),
          duration: const Duration(seconds: 3),
          action: SnackBarAction(
            label: 'RETRY',
            textColor: Colors.amberAccent,
            onPressed: provider.startDashboard,
          ),
        ),
      );
    provider.startDashboard();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final provider = Provider.of<DashboardProvider>(context);
    final isMqttConnected = provider.isMqttConnected;

    final isOccupied = camera.status == 'YES';
    final isEmpty = camera.status == 'NO';
    final isPowerOn = camera.powerState == 'ON';
    final isPendingUpdate = provider.pendingForceUpdates.contains(camera.id);

    final statusColor = isOccupied
        ? _colorSuccess
        : isEmpty
        ? _colorDanger
        : _colorUnknown;
    final badgeColor =
        (isOccupied
                ? _badgeOccupied
                : isEmpty
                ? _badgeEmpty
                : _badgeUnknown)
            .withValues(alpha: 0.92);
    final statusText = isOccupied
        ? 'OCCUPIED'
        : isEmpty
        ? 'EMPTY'
        : 'UNKNOWN';

    return Card(
      elevation: isDark ? 6 : 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => showDialog(
          context: context,
          useSafeArea: false,
          builder: (_) => FullscreenPreview(camera: camera),
        ),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isCompact = constraints.maxWidth < 240;

            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // ── Header: Camera ID + status dot ──
                Padding(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12.0,
                    vertical: 6.0,
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: statusColor,
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          camera.id,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 14,
                            color: isDark
                                ? Colors.white
                                : const Color(0xFF0F172A),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                // ── Camera feed image ──
                Expanded(
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      // Feed image (or placeholder)
                      camera.imagePath.isNotEmpty
                          ? CachedNetworkImage(
                              imageUrl: camera.imagePath,
                              cacheManager:
                                  cameraCacheManager, // Prevent unbounded disk caching
                              fit: BoxFit.cover,
                              memCacheWidth: 600, // Optimize memory usage
                              httpHeaders: provider.apiKey.isNotEmpty
                                  ? {'X-API-Key': provider.apiKey}
                                  : null,
                              placeholder: (context, url) =>
                                  _feedPlaceholder(isDark, loading: true),
                              errorWidget: (context, url, error) =>
                                  _feedPlaceholder(isDark),
                            )
                          : _feedPlaceholder(isDark),

                      // Occupancy badge (standard mode only)
                      if (!isCompact)
                        Positioned(
                          top: 8,
                          left: 8,
                          child: _badge(statusText, badgeColor),
                        ),

                      // Power state badge (standard mode only)
                      if (!isCompact)
                        Positioned(
                          top: 8,
                          right: 8,
                          child: _badge(
                            'POWER: ${camera.powerState}',
                            isPowerOn
                                ? _badgeOccupied.withValues(alpha: 0.92)
                                : _badgePowerOff.withValues(alpha: 0.92),
                          ),
                        ),

                      // Offline overlay (camera powered off)
                      if (!isPowerOn)
                        Container(
                          color: Colors.black.withValues(alpha: 0.65),
                          child: const Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.power_off,
                                  color: Colors.white70,
                                  size: 28,
                                ),
                                SizedBox(height: 4),
                                Text(
                                  'OFFLINE',
                                  style: TextStyle(
                                    color: Colors.white70,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 1.0,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                      // Compact action overlay (bottom strip)
                      if (isCompact)
                        Positioned(
                          bottom: 0,
                          left: 0,
                          right: 0,
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.65),
                              border: Border(
                                top: BorderSide(
                                  color: isDark
                                      ? const Color(0xFF334155)
                                      : const Color(
                                          0xFFE2E8F0,
                                        ).withValues(alpha: 0.3),
                                  width: 0.8,
                                ),
                              ),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 2),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: [
                                // Power toggle
                                InkWell(
                                  onTap: () => isMqttConnected
                                      ? provider.togglePowerState(
                                          camera.id,
                                          camera.powerState,
                                        )
                                      : _onOfflineAction(context, provider),
                                  borderRadius: BorderRadius.circular(4),
                                  child: Opacity(
                                    opacity: isMqttConnected ? 1.0 : 0.4,
                                    child: Padding(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 16,
                                        vertical: 6,
                                      ),
                                      child: Icon(
                                        isPowerOn
                                            ? Icons.power_settings_new
                                            : Icons.power_off,
                                        size: 18,
                                        color: isPowerOn
                                            ? _colorSuccess
                                            : _colorDanger,
                                      ),
                                    ),
                                  ),
                                ),
                                Container(
                                  width: 1,
                                  height: 16,
                                  color: Colors.white24,
                                ),
                                // Force refresh
                                if (isPendingUpdate)
                                  const Padding(
                                    padding: EdgeInsets.symmetric(
                                      horizontal: 16,
                                      vertical: 6,
                                    ),
                                    child: SizedBox(
                                      height: 18,
                                      width: 18,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                        color: Colors.blue,
                                      ),
                                    ),
                                  )
                                else
                                  InkWell(
                                    onTap: () => isMqttConnected
                                        ? provider.requestForceUpdate(camera.id)
                                        : _onOfflineAction(context, provider),
                                    borderRadius: BorderRadius.circular(4),
                                    child: Opacity(
                                      opacity: isMqttConnected ? 1.0 : 0.4,
                                      child: const Padding(
                                        padding: EdgeInsets.symmetric(
                                          horizontal: 16,
                                          vertical: 6,
                                        ),
                                        child: Icon(
                                          Icons.refresh,
                                          size: 18,
                                          color: Colors.white70,
                                        ),
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),

                // ── Standard footer: action buttons ──
                if (!isCompact)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Row(
                      children: [
                        Expanded(
                          child: Opacity(
                            opacity: isMqttConnected ? 1.0 : 0.5,
                            child: OutlinedButton.icon(
                              onPressed: () => isMqttConnected
                                  ? provider.togglePowerState(
                                      camera.id,
                                      camera.powerState,
                                    )
                                  : _onOfflineAction(context, provider),
                              icon: Icon(
                                isPowerOn
                                    ? Icons.power_settings_new
                                    : Icons.power_off,
                                size: 14,
                                color: isPowerOn ? _colorSuccess : _colorDanger,
                              ),
                              label: Text(
                                isPowerOn ? 'SHUTDOWN' : 'ACTIVATE',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                  color: isDark ? Colors.white : Colors.black87,
                                ),
                              ),
                              style: OutlinedButton.styleFrom(
                                padding: const EdgeInsets.symmetric(
                                  vertical: 8,
                                ),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                side: BorderSide(
                                  color: isPowerOn
                                      ? _colorSuccess.withValues(alpha: 0.5)
                                      : _colorDanger.withValues(alpha: 0.5),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        SizedBox(
                          width: 42,
                          child: Opacity(
                            opacity: isMqttConnected ? 1.0 : 0.5,
                            child: ElevatedButton(
                              onPressed: isPendingUpdate
                                  ? null
                                  : () => isMqttConnected
                                        ? provider.requestForceUpdate(camera.id)
                                        : _onOfflineAction(context, provider),
                              style: ElevatedButton.styleFrom(
                                padding: EdgeInsets.zero,
                                backgroundColor: isDark
                                    ? const Color(0xFF334155)
                                    : const Color(0xFFCBD5E1),
                                foregroundColor: isDark
                                    ? Colors.white
                                    : Colors.black87,
                                elevation: 0,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                              child: isPendingUpdate
                                  ? const SizedBox(
                                      height: 16,
                                      width: 16,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                        color: Colors.blue,
                                      ),
                                    )
                                  : const Icon(Icons.refresh, size: 18),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  static Widget _feedPlaceholder(bool isDark, {bool loading = false}) {
    final bg = isDark ? const Color(0xFF0F172A) : const Color(0xFFE2E8F0);
    return Container(
      color: bg,
      child: Center(
        child: loading
            ? const CircularProgressIndicator(strokeWidth: 2)
            : const Icon(Icons.videocam_off, size: 36, color: Colors.grey),
      ),
    );
  }

  static Widget _badge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
          fontSize: 10,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
