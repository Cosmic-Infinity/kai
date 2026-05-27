import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import '../models/camera.dart';
import '../providers/dashboard_provider.dart';
import 'camera_card.dart';

class FullscreenPreview extends StatefulWidget {
  final Camera camera;

  const FullscreenPreview({super.key, required this.camera});

  @override
  State<FullscreenPreview> createState() => _FullscreenPreviewState();
}

class _FullscreenPreviewState extends State<FullscreenPreview> {
  bool _showBboxes = false;
  final TransformationController _transformationController = TransformationController();
  double _zoomLevel = 1.0;
  Size _viewportSize = Size.zero;

  void _onOfflineAction(BuildContext context, DashboardProvider provider) {
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(
        content: const Row(children: [
          Icon(Icons.wifi_off, color: Colors.amberAccent, size: 20),
          SizedBox(width: 10),
          Expanded(child: Text('MQTT disconnected. Attempting to reconnect...')),
        ]),
        duration: const Duration(seconds: 3),
        action: SnackBarAction(
          label: 'RETRY',
          textColor: Colors.amberAccent,
          onPressed: provider.startDashboard,
        ),
      ));
    provider.startDashboard();
  }

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  void _zoomTo(double newZoom) {
    if (_viewportSize == Size.zero) return;
    setState(() {
      _zoomLevel = newZoom.clamp(1.0, 8.0);
      final double cx = _viewportSize.width / 2;
      final double cy = _viewportSize.height / 2;
      _transformationController.value = Matrix4.identity()
        ..translate(cx, cy, 0.0)
        ..scale(_zoomLevel)
        ..translate(-cx, -cy, 0.0);
    });
  }

  void _zoomIn() => _zoomTo(_zoomLevel * 1.25);
  void _zoomOut() => _zoomTo(_zoomLevel * 0.8);
  void _resetZoom() => _zoomTo(1.0);

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final provider = Provider.of<DashboardProvider>(context);

    const colorSuccess = Color(0xFF10B981);
    const colorDanger  = Color(0xFFF43F5E);
    const colorUnknown = Color(0xFF64748B);

    final currentCamera = provider.cameras[widget.camera.id] ?? widget.camera;
    final isOccupied = currentCamera.status == 'YES';
    final isEmpty = currentCamera.status == 'NO';
    final isPowerOn = currentCamera.powerState == 'ON';
    final isPendingUpdate = provider.pendingForceUpdates.contains(currentCamera.id);
    final isMqttConnected = provider.isMqttConnected;

    final statusColor = isOccupied ? colorSuccess : isEmpty ? colorDanger : colorUnknown;
    final statusText  = isOccupied ? 'OCCUPIED'   : isEmpty ? 'EMPTY'     : 'UNKNOWN';

    return Dialog(
      backgroundColor: Colors.transparent,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      insetPadding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 32.0),
        child: Container(
          decoration: BoxDecoration(
            color: isDark ? const Color(0xFF0F172A) : const Color(0xFFF1F5F9), // slate-900 / slate-100
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: isDark ? const Color(0xFF334155) : const Color(0xFFCBD5E1),
              width: 1,
            ),
          ),
          child: SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header Row
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      Container(
                        width: 12,
                        height: 12,
                        decoration: BoxDecoration(
                          color: statusColor,
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              currentCamera.id,
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                                color: isDark ? Colors.white : Colors.black87,
                              ),
                            ),
                            Text(
                              'Status: $statusText • Power: ${currentCamera.powerState}',
                              style: TextStyle(
                                fontSize: 12,
                                color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF475569),
                              ),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: () => Navigator.of(context).pop(),
                        color: isDark ? Colors.white : Colors.black87,
                      ),
                    ],
                  ),
                ),

              // Image Area with Pan & Zoom & Custom BBoxes
              Expanded(
                child: Container(
                  color: Colors.black.withValues(alpha: 0.2),
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      _viewportSize = Size(constraints.maxWidth, constraints.maxHeight);
                      return Stack(
                        alignment: Alignment.center,
                        children: [
                          InteractiveViewer(
                            transformationController: _transformationController,
                            maxScale: 8.0,
                            onInteractionUpdate: (details) {
                              // Dynamic update zoom indicator
                              setState(() {
                                _zoomLevel = _transformationController.value.getMaxScaleOnAxis();
                              });
                            },
                            child: _BBoxImagePreview(
                              imageUrl: currentCamera.imagePath,
                              bboxes: currentCamera.boundingBoxes,
                              showBboxes: _showBboxes,
                            ),
                          ),

                          // Zoom Overlay Tag
                          Positioned(
                            bottom: 16,
                            left: 16,
                            child: Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                              decoration: BoxDecoration(
                                color: Colors.black.withValues(alpha: 0.6),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                'Zoom: ${(_zoomLevel * 100).toInt()}%',
                                style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
                              ),
                            ),
                          ),

                          // Floating Glassmorphic Zoom Controls
                          Positioned(
                            bottom: 16,
                            right: 16,
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(12),
                              child: Container(
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.65),
                                  border: Border.all(
                                    color: Colors.white.withValues(alpha: 0.15),
                                    width: 1,
                                  ),
                                ),
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    IconButton(
                                      icon: const Icon(Icons.add, color: Colors.white, size: 20),
                                      onPressed: _zoomIn,
                                      tooltip: 'Zoom In',
                                    ),
                                    Container(
                                      width: 24,
                                      height: 1,
                                      color: Colors.white24,
                                    ),
                                    IconButton(
                                      icon: const Icon(Icons.remove, color: Colors.white, size: 20),
                                      onPressed: _zoomOut,
                                      tooltip: 'Zoom Out',
                                    ),
                                    Container(
                                      width: 24,
                                      height: 1,
                                      color: Colors.white24,
                                    ),
                                    IconButton(
                                      icon: const Icon(Icons.youtube_searched_for, color: Colors.white, size: 20),
                                      onPressed: _resetZoom,
                                      tooltip: 'Reset Zoom',
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                ),
              ),

              // Control & Toolbar Row
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final screenWidth = MediaQuery.of(context).size.width;
                    final isMobile = screenWidth < 500;

                    if (isMobile) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          // Toggle BBoxes Button (Row 1)
                          ElevatedButton.icon(
                            onPressed: () {
                              setState(() {
                                _showBboxes = !_showBboxes;
                              });
                            },
                            icon: Icon(
                              _showBboxes ? Icons.visibility : Icons.visibility_off,
                              size: 16,
                            ),
                            label: Text(
                              _showBboxes ? 'HIDE AI OVERLAY' : 'SHOW AI OVERLAY',
                              style: const TextStyle(fontSize: 12),
                            ),
                            style: ElevatedButton.styleFrom(
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              padding: const EdgeInsets.symmetric(vertical: 10),
                            ),
                          ),
                          const SizedBox(height: 8),
                          // Power & Refresh Actions (Row 2)
                          Row(
                            children: [
                              Expanded(
                                child: Opacity(
                                  opacity: isMqttConnected ? 1.0 : 0.5,
                                  child: ElevatedButton.icon(
                                    onPressed: () {
                                      if (isMqttConnected) {
                                        provider.togglePowerState(currentCamera.id, currentCamera.powerState);
                                      } else {
                                        _onOfflineAction(context, provider);
                                      }
                                    },
                                    icon: Icon(
                                      isPowerOn ? Icons.power_settings_new : Icons.power_off,
                                      size: 16,
                                    ),
                                    label: Text(isPowerOn ? 'SHUTDOWN' : 'ACTIVATE'),
                                    style: ElevatedButton.styleFrom(
                              backgroundColor: isPowerOn ? const Color(0xFF166534) : const Color(0xFF991B1B),
                                      foregroundColor: Colors.white,
                                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                      padding: const EdgeInsets.symmetric(vertical: 10),
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              SizedBox(
                                height: 38,
                                width: 56,
                                child: Opacity(
                                  opacity: isMqttConnected ? 1.0 : 0.5,
                                  child: OutlinedButton(
                                    onPressed: isPendingUpdate
                                        ? null
                                        : () {
                                            if (isMqttConnected) {
                                              provider.requestForceUpdate(currentCamera.id);
                                            } else {
                                              _onOfflineAction(context, provider);
                                            }
                                          },
                                    style: OutlinedButton.styleFrom(
                                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                      padding: EdgeInsets.zero,
                                    ),
                                    child: isPendingUpdate
                                        ? const SizedBox(
                                            height: 16,
                                            width: 16,
                                            child: CircularProgressIndicator(strokeWidth: 2),
                                          )
                                        : const Icon(Icons.refresh, size: 18),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      );
                    }

                    // Desktop Row Layout
                    return Row(
                      children: [
                        // Toggle BBoxes Button
                        ElevatedButton.icon(
                          onPressed: () {
                            setState(() {
                              _showBboxes = !_showBboxes;
                            });
                          },
                          icon: Icon(
                            _showBboxes ? Icons.visibility : Icons.visibility_off,
                            size: 16,
                          ),
                          label: Text(
                            _showBboxes ? 'HIDE AI OVERLAY' : 'SHOW AI OVERLAY',
                            style: const TextStyle(fontSize: 12),
                          ),
                          style: ElevatedButton.styleFrom(
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          ),
                        ),

                        const Spacer(),

                        // Shutdown / Activate Button
                        Opacity(
                          opacity: isMqttConnected ? 1.0 : 0.5,
                          child: ElevatedButton.icon(
                            onPressed: () {
                              if (isMqttConnected) {
                                provider.togglePowerState(currentCamera.id, currentCamera.powerState);
                              } else {
                                _onOfflineAction(context, provider);
                              }
                            },
                            icon: Icon(
                              isPowerOn ? Icons.power_settings_new : Icons.power_off,
                              size: 16,
                            ),
                            label: Text(isPowerOn ? 'SHUTDOWN' : 'ACTIVATE'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: isPowerOn ? const Color(0xFF166534) : const Color(0xFF991B1B),
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),

                        // Request update Button
                        SizedBox(
                          height: 40,
                          child: Opacity(
                            opacity: isMqttConnected ? 1.0 : 0.5,
                            child: OutlinedButton(
                              onPressed: isPendingUpdate
                                  ? null
                                 : () {
                                      if (isMqttConnected) {
                                        provider.requestForceUpdate(currentCamera.id);
                                      } else {
                                        _onOfflineAction(context, provider);
                                      }
                                    },
                              style: OutlinedButton.styleFrom(
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                padding: const EdgeInsets.symmetric(horizontal: 16),
                              ),
                              child: isPendingUpdate
                                  ? const SizedBox(
                                      height: 16,
                                      width: 16,
                                      child: CircularProgressIndicator(strokeWidth: 2),
                                    )
                                  : const Text('REFRESH'),
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Custom Canvas Painter to draw dynamic bounding boxes and badges on top of the image
class BBoxPainter extends CustomPainter {
  final List<BoundingBox> bboxes;

  BBoxPainter({required this.bboxes});

  @override
  void paint(Canvas canvas, Size size) {
    for (var bbox in bboxes) {
      // Parse Color Hex (e.g. #ff0000)
      Color color = Colors.red;
      try {
        final hex = bbox.colorHex.replaceAll('#', '');
        if (hex.length == 6) {
          color = Color(int.parse('FF$hex', radix: 16));
        }
      } catch (_) {}

      // Calculate pixel coordinates
      // Since coordinates are fractional (0.0 -> 1.0), we scale them to size.
      // The Python backend serves coordinates mapped to Kivy's bottom-left origin:
      // fx = x1, fy = 1.0 - y2, fw = x2 - x1, fh = y2 - y1.
      // In Flutter, the origin is top-left, so:
      // left = fx, width = fw, height = fh, and top = 1.0 - (fy + fh).
      final double left = bbox.x * size.width;
      final double top = (1.0 - (bbox.y + bbox.height)) * size.height;
      final double width = bbox.width * size.width;
      final double height = bbox.height * size.height;

      // Draw bounding box rectangle
      final paint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;

      final rect = Rect.fromLTWH(left, top, width, height);
      canvas.drawRect(rect, paint);

      // Draw dynamic label badge
      final textSpan = TextSpan(
        text: ' ${bbox.label} ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final badgeHeight = textPainter.height + 4;
      final badgeWidth = textPainter.width + 8;
      // Position at top of rect
      double badgeY = top;
      if (badgeY - badgeHeight < 0) {
        badgeY = top + height - badgeHeight;
      } else {
        badgeY = top - badgeHeight;
      }

      // Draw background rectangle for text
      final badgeRect = Rect.fromLTWH(left, badgeY, badgeWidth, badgeHeight);
      final bgPaint = Paint()..color = const Color(0xFF0F172A).withValues(alpha: 0.85);
      final borderPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0;

      canvas.drawRect(badgeRect, bgPaint);
      canvas.drawRect(badgeRect, borderPaint);

      // Render the Text label
      textPainter.paint(canvas, Offset(left + 4, badgeY + 2));
    }
  }

  @override
  bool shouldRepaint(covariant BBoxPainter oldDelegate) {
    return oldDelegate.bboxes != bboxes;
  }
}

/// Helper widget to resolve the image aspect ratio dynamically and constrain BBox painter overlays to exact image bounds
class _BBoxImagePreview extends StatefulWidget {
  final String imageUrl;
  final List<BoundingBox> bboxes;
  final bool showBboxes;

  const _BBoxImagePreview({
    required this.imageUrl,
    required this.bboxes,
    required this.showBboxes,
  });

  @override
  State<_BBoxImagePreview> createState() => _BBoxImagePreviewState();
}

class _BBoxImagePreviewState extends State<_BBoxImagePreview> {
  double? _aspectRatio;
  bool _loading = true;
  ImageStream? _imageStream;
  ImageStreamListener? _imageStreamListener;

  @override
  void initState() {
    super.initState();
    _resolveImage();
  }

  @override
  void didUpdateWidget(_BBoxImagePreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.imageUrl != widget.imageUrl) {
      _resolveImage();
    }
  }

  @override
  void dispose() {
    _cleanupStream();
    super.dispose();
  }

  void _cleanupStream() {
    if (_imageStream != null && _imageStreamListener != null) {
      _imageStream!.removeListener(_imageStreamListener!);
    }
    _imageStream = null;
    _imageStreamListener = null;
  }

  void _resolveImage() {
    _cleanupStream();
    if (widget.imageUrl.isEmpty) {
      setState(() {
        _loading = false;
        _aspectRatio = null;
      });
      return;
    }

    setState(() {
      _loading = true;
    });

    final apiKey = Provider.of<DashboardProvider>(context, listen: false).apiKey;
    final provider = CachedNetworkImageProvider(
      widget.imageUrl,
      cacheManager: cameraCacheManager,
      headers: apiKey.isNotEmpty ? {'X-API-Key': apiKey} : null,
    );
    _imageStream = provider.resolve(const ImageConfiguration());
    
    _imageStreamListener = ImageStreamListener(
      (ImageInfo info, bool synchronousCall) {
        if (mounted) {
          setState(() {
            _aspectRatio = info.image.width / info.image.height;
            _loading = false;
          });
        }
      },
      onError: (dynamic exception, StackTrace? stackTrace) {
        if (mounted) {
          setState(() {
            _loading = false;
            _aspectRatio = null;
          });
        }
      },
    );

    _imageStream!.addListener(_imageStreamListener!);
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    if (_aspectRatio == null) {
      return const Center(
        child: Icon(Icons.videocam_off, size: 64, color: Colors.grey),
      );
    }

    final apiKey = Provider.of<DashboardProvider>(context, listen: false).apiKey;

    return Center(
      child: AspectRatio(
        aspectRatio: _aspectRatio!,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CachedNetworkImage(
              imageUrl: widget.imageUrl,
              cacheManager: cameraCacheManager,
              fit: BoxFit.fill,
              httpHeaders: apiKey.isNotEmpty ? {'X-API-Key': apiKey} : null,
              placeholder: (context, url) => const Center(
                child: CircularProgressIndicator(),
              ),
              errorWidget: (context, url, error) => const Center(
                child: Icon(Icons.videocam_off, size: 64, color: Colors.grey),
              ),
            ),
            if (widget.showBboxes && widget.bboxes.isNotEmpty)
              Positioned.fill(
                child: CustomPaint(
                  painter: BBoxPainter(
                    bboxes: widget.bboxes,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
