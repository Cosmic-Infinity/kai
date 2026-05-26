import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../models/camera.dart';
import '../providers/dashboard_provider.dart';
import 'fullscreen_preview.dart';

class CameraCard extends StatelessWidget {
  final Camera camera;

  const CameraCard({super.key, required this.camera});

  void _showDisconnectedSnackBar(BuildContext context, DashboardProvider provider) {
    ScaffoldMessenger.of(context).clearSnackBars();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: const Row(
          children: [
            Icon(Icons.wifi_off, color: Colors.amberAccent, size: 20),
            SizedBox(width: 10),
            Expanded(
              child: Text(
                'MQTT disconnected. Attempting to reconnect...',
                style: TextStyle(fontSize: 13),
              ),
            ),
          ],
        ),
        duration: const Duration(seconds: 3),
        action: SnackBarAction(
          label: 'RETRY',
          textColor: Colors.amberAccent,
          onPressed: () {
            provider.startDashboard();
          },
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

    // Color definitions mapped from Kivy hex constants
    final colorSuccess = const Color(0xFF10B981); // emerald
    final colorDanger = const Color(0xFFF43F5E); // rose
    final colorUnknown = const Color(0xFF64748B); // slate

    final badgeSuccess = const Color(0xFF065F46).withOpacity(0.92);
    final badgeDanger = const Color(0xFF9F1239).withOpacity(0.92);
    final badgeUnknown = const Color(0xFF334155).withOpacity(0.92);

    final isOccupied = camera.status == 'YES';
    final isEmpty = camera.status == 'NO';

    final Color statusColor = isOccupied
        ? colorSuccess
        : isEmpty
            ? colorDanger
            : colorUnknown;

    final Color badgeColor = isOccupied
        ? badgeSuccess
        : isEmpty
            ? badgeDanger
            : badgeUnknown;

    final String statusText = isOccupied
        ? 'OCCUPIED'
        : isEmpty
            ? 'EMPTY'
            : 'UNKNOWN';

    final bool isPowerOn = camera.powerState == 'ON';
    final isPendingUpdate = provider.pendingForceUpdates.contains(camera.id);

    return Card(
      color: isDark ? const Color(0xFF1E293B) : Colors.white, // slate-800 / white
      elevation: isDark ? 6 : 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: isDark ? const Color(0xFF334155) : const Color(0xFFCBD5E1),
          width: 1,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () {
          showDialog(
            context: context,
            useSafeArea: false,
            builder: (_) => FullscreenPreview(camera: camera),
          );
        },
        child: LayoutBuilder(
          builder: (context, constraints) {
            final isCompact = constraints.maxWidth < 240;

            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header: Camera ID & Status Dot / Action Icons
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12.0, vertical: 6.0),
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
                            color: isDark ? Colors.white : const Color(0xFF0F172A),
                          ),
                        ),
                      ),

                    ],
                  ),
                ),

                // Camera Feed Image (Middle Section)
                Expanded(
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      camera.imagePath.isNotEmpty
                          ? CachedNetworkImage(
                              imageUrl: camera.imagePath,
                              fit: BoxFit.cover,
                              httpHeaders: provider.apiKey.isNotEmpty
                                  ? {'X-API-Key': provider.apiKey}
                                  : null,
                              placeholder: (context, url) => Container(
                                color: isDark ? const Color(0xFF0F172A) : const Color(0xFFE2E8F0),
                                child: const Center(child: CircularProgressIndicator(strokeWidth: 2)),
                              ),
                              errorWidget: (context, url, error) => Container(
                                color: isDark ? const Color(0xFF0F172A) : const Color(0xFFE2E8F0),
                                child: const Center(
                                  child: Icon(Icons.videocam_off, size: 36, color: Colors.grey),
                                ),
                              ),
                            )
                          : Container(
                              color: isDark ? const Color(0xFF0F172A) : const Color(0xFFE2E8F0),
                              child: const Center(
                                child: Icon(Icons.videocam_off, size: 36, color: Colors.grey),
                              ),
                            ),

                      // Overlay Badge: Occupancy status (only in standard mode to keep image clean in grid)
                      if (!isCompact)
                        Positioned(
                          top: 8,
                          left: 8,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: badgeColor,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              statusText,
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 10,
                                letterSpacing: 0.5,
                              ),
                            ),
                          ),
                        ),

                      // Overlay Badge: Power state (only in standard mode)
                      if (!isCompact)
                        Positioned(
                          top: 8,
                          right: 8,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: isPowerOn
                                  ? const Color(0xFF065F46).withOpacity(0.92)
                                  : const Color(0xFF991B1B).withOpacity(0.92),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              'POWER: ${camera.powerState}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 10,
                              ),
                            ),
                          ),
                        ),

                      // OFFLINE Premium Overlay (when camera is powered down)
                      if (!isPowerOn)
                        Container(
                          color: Colors.black.withOpacity(0.65),
                          child: const Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.power_off, color: Colors.white70, size: 28),
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

                      // Compact action controls overlaid at the bottom of the feed image
                      if (isCompact)
                        Positioned(
                          bottom: 0,
                          left: 0,
                          right: 0,
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.65),
                              border: Border(
                                top: BorderSide(
                                  color: isDark ? const Color(0xFF334155) : const Color(0xFFE2E8F0).withOpacity(0.3),
                                  width: 0.8,
                                ),
                              ),
                            ),
                            padding: const EdgeInsets.symmetric(vertical: 2),
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                              children: [
                                // Toggle Power Button with large tap target
                                InkWell(
                                  onTap: () {
                                    if (isMqttConnected) {
                                      provider.togglePowerState(camera.id, camera.powerState);
                                    } else {
                                      _showDisconnectedSnackBar(context, provider);
                                    }
                                  },
                                  borderRadius: BorderRadius.circular(4),
                                  child: Opacity(
                                    opacity: isMqttConnected ? 1.0 : 0.4,
                                    child: Padding(
                                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                                      child: Icon(
                                        isPowerOn ? Icons.power_settings_new : Icons.power_off,
                                        size: 18,
                                        color: isPowerOn ? colorSuccess : colorDanger,
                                      ),
                                    ),
                                  ),
                                ),
                                // Vertical spacer line
                                Container(
                                  width: 1,
                                  height: 16,
                                  color: Colors.white24,
                                ),
                                // Force Refresh Button with large tap target
                                isPendingUpdate
                                    ? const SizedBox(
                                        height: 16,
                                        width: 16,
                                        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.blue),
                                      )
                                    : InkWell(
                                        onTap: () {
                                          if (isMqttConnected) {
                                            provider.requestForceUpdate(camera.id);
                                          } else {
                                            _showDisconnectedSnackBar(context, provider);
                                          }
                                        },
                                        borderRadius: BorderRadius.circular(4),
                                        child: Opacity(
                                          opacity: isMqttConnected ? 1.0 : 0.4,
                                          child: Padding(
                                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                                            child: const Icon(
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

                if (!isCompact)
                  Padding(
                    padding: const EdgeInsets.only(left: 8.0, right: 8.0, bottom: 8.0, top: 8.0),
                    child: Row(
                      children: [
                        // Toggle Power Button
                        Expanded(
                          child: Opacity(
                            opacity: isMqttConnected ? 1.0 : 0.5,
                            child: OutlinedButton.icon(
                              onPressed: () {
                                if (isMqttConnected) {
                                  provider.togglePowerState(camera.id, camera.powerState);
                                } else {
                                  _showDisconnectedSnackBar(context, provider);
                                }
                              },
                              icon: Icon(
                                isPowerOn ? Icons.power_settings_new : Icons.power_off,
                                size: 14,
                                color: isPowerOn ? colorSuccess : colorDanger,
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
                                padding: const EdgeInsets.symmetric(vertical: 8),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                side: BorderSide(
                                  color: isPowerOn ? colorSuccess.withOpacity(0.5) : colorDanger.withOpacity(0.5),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),

                        // Force Update Button
                        SizedBox(
                          width: 42,
                          child: Opacity(
                            opacity: isMqttConnected ? 1.0 : 0.5,
                            child: ElevatedButton(
                              onPressed: isPendingUpdate
                                  ? null
                                  : () {
                                      if (isMqttConnected) {
                                        provider.requestForceUpdate(camera.id);
                                      } else {
                                        _showDisconnectedSnackBar(context, provider);
                                      }
                                    },
                              style: ElevatedButton.styleFrom(
                                padding: EdgeInsets.zero,
                                backgroundColor: isDark ? const Color(0xFF334155) : const Color(0xFFCBD5E1),
                                foregroundColor: isDark ? Colors.white : Colors.black87,
                                elevation: 0,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              ),
                              child: isPendingUpdate
                                  ? const SizedBox(
                                      height: 16,
                                      width: 16,
                                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.blue),
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
}
