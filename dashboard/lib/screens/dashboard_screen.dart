import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/dashboard_provider.dart';
import '../widgets/camera_card.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> with WidgetsBindingObserver {
  final _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    // Connect to services on screen mount
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<DashboardProvider>(context, listen: false).startDashboard();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _searchController.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // Always force reconnect on resume. The TCP socket dies silently when
      // the app is backgrounded on Android, but the MQTT client may still
      // report isConnected=true until the keepalive timeout fires.
      print('[DashboardScreen] App resumed. Forcing MQTT reconnect...');
      Provider.of<DashboardProvider>(context, listen: false).startDashboard();
    }
  }

  Widget _buildGroupHeader(BuildContext context, IconData icon, String title, bool isDark) {
    return Padding(
      padding: const EdgeInsets.only(top: 8.0, bottom: 6.0),
      child: Row(
        children: [
          Icon(
            icon,
            size: 16,
            color: isDark ? const Color(0xFF60A5FA) : Theme.of(context).primaryColor,
          ),
          const SizedBox(width: 8),
          Text(
            title,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 12,
              letterSpacing: 0.5,
              color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF475569),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfigField(
    BuildContext context,
    String label,
    TextEditingController controller,
    String description,
  ) {
    return TextFormField(
      controller: controller,
      keyboardType: TextInputType.number,
      decoration: InputDecoration(
        labelText: label,
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        suffixIcon: Tooltip(
          message: description,
          triggerMode: TooltipTriggerMode.tap,
          padding: const EdgeInsets.all(12),
          margin: const EdgeInsets.symmetric(horizontal: 24),
          showDuration: const Duration(seconds: 4),
          child: IconButton(
            icon: const Icon(Icons.info_outline, size: 20),
            onPressed: () {
              showDialog(
                context: context,
                builder: (infoCtx) => AlertDialog(
                  insetPadding: const EdgeInsets.symmetric(horizontal: 56.0, vertical: 24.0),
                  title: Row(
                    children: [
                      const Icon(Icons.info_outline, color: Colors.blue),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          label,
                          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),
                  content: Text(
                    description,
                    style: const TextStyle(fontSize: 14, height: 1.4),
                  ),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.of(infoCtx).pop(),
                      child: const Text('OK'),
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  // Open System settings modal dialouge
  void _openSettingsDialog(BuildContext context) async {
    final provider = Provider.of<DashboardProvider>(context, listen: false);
    final currentConfig = await provider.fetchSystemConfig();

    if (currentConfig == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to load system settings from server')),
        );
      }
      return;
    }

    final imageIntervalController = TextEditingController(text: currentConfig['IMAGE_SERVER_INTERVAL']?.toString() ?? '60');
    final dashIntervalController = TextEditingController(text: currentConfig['DASHBOARD_INTERVAL']?.toString() ?? '30');
    final controlIntervalController = TextEditingController(text: currentConfig['CONTROL_SERVER_INTERVAL']?.toString() ?? '30');
    final thresholdController = TextEditingController(text: currentConfig['INACTIVITY_THRESHOLD']?.toString() ?? '10');
    final keepaliveController = TextEditingController(text: currentConfig['MQTT_KEEPALIVE']?.toString() ?? '120');
    final reconnectController = TextEditingController(text: currentConfig['MQTT_RECONNECT_DELAY']?.toString() ?? '2');

    if (!mounted) return;

    showDialog(
      context: context,
      builder: (dialogContext) {
        final isDark = Theme.of(dialogContext).brightness == Brightness.dark;
        return AlertDialog(
          title: const Text('System Configuration', style: TextStyle(fontWeight: FontWeight.bold)),
          content: SizedBox(
            width: 400,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildGroupHeader(dialogContext, Icons.camera_enhance_outlined, 'Image Processing Module', isDark),
                  const SizedBox(height: 4),
                  _buildConfigField(
                    dialogContext,
                    'Image Capture Interval (s)',
                    imageIntervalController,
                    'Interval (in seconds) at which the Image Server captures and analyzes new camera frames.',
                  ),
                  const SizedBox(height: 16),
                  
                  _buildGroupHeader(dialogContext, Icons.dashboard_outlined, 'Dashboard UI Module', isDark),
                  const SizedBox(height: 4),
                  _buildConfigField(
                    dialogContext,
                    'Dashboard Refresh Interval (s)',
                    dashIntervalController,
                    'Interval (in seconds) at which the Dashboard UI automatically fetches the latest camera images.',
                  ),
                  const SizedBox(height: 16),
                  
                  _buildGroupHeader(dialogContext, Icons.settings_suggest_outlined, 'Control Server Module', isDark),
                  const SizedBox(height: 4),
                  _buildConfigField(
                    dialogContext,
                    'Control Server Loop Interval (s)',
                    controlIntervalController,
                    'Interval (in seconds) at which the Control Server checks camera statuses to make power decisions.',
                  ),
                  const SizedBox(height: 10),
                  _buildConfigField(
                    dialogContext,
                    'Inactivity Shutdown Threshold',
                    thresholdController,
                    'Number of consecutive "Empty" readings before room appliances are automatically turned off.',
                  ),
                  const SizedBox(height: 16),
                  
                  _buildGroupHeader(dialogContext, Icons.wifi_tethering_outlined, 'Network & MQTT Broker', isDark),
                  const SizedBox(height: 4),
                  _buildConfigField(
                    dialogContext,
                    'MQTT Keepalive (s)',
                    keepaliveController,
                    'Maximum time (in seconds) allowed between client-broker messages before connection loss is assumed.',
                  ),
                  const SizedBox(height: 10),
                  _buildConfigField(
                    dialogContext,
                    'MQTT Reconnect Delay (s)',
                    reconnectController,
                    'Time (in seconds) to wait before attempting to reconnect to the MQTT broker after disconnection.',
                  ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(),
              child: const Text('CANCEL'),
            ),
            ElevatedButton(
              onPressed: () async {
                final settings = {
                  'IMAGE_SERVER_INTERVAL': int.tryParse(imageIntervalController.text) ?? 60,
                  'DASHBOARD_INTERVAL': int.tryParse(dashIntervalController.text) ?? 30,
                  'CONTROL_SERVER_INTERVAL': int.tryParse(controlIntervalController.text) ?? 30,
                  'INACTIVITY_THRESHOLD': int.tryParse(thresholdController.text) ?? 10,
                  'MQTT_KEEPALIVE': int.tryParse(keepaliveController.text) ?? 120,
                  'MQTT_RECONNECT_DELAY': int.tryParse(reconnectController.text) ?? 2,
                };
                final success = await provider.saveSystemSettings(settings);
                if (success && mounted) {
                  Navigator.of(dialogContext).pop();
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Settings saved successfully')),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Error updating server settings')),
                  );
                }
              },
              child: const Text('SAVE'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final provider = Provider.of<DashboardProvider>(context);
    final camerasList = provider.filteredCameras;

    // Mapping icons to filters matching Kivy list
    final List<String> filters = [
      'All',
      'Occupied',
      'Empty',
      'Power ON',
      'Power OFF',
      'Occupied & Power ON',
      'Occupied & Power OFF',
      'Empty & Power ON',
      'Empty & Power OFF'
    ];

    final Map<String, IconData> filterIcons = {
      'All': Icons.tag,
      'Occupied': Icons.account_box,
      'Empty': Icons.person_off,
      'Power ON': Icons.power,
      'Power OFF': Icons.power_off,
      'Occupied & Power ON': Icons.flash_on,
      'Occupied & Power OFF': Icons.flash_off,
      'Empty & Power ON': Icons.bolt,
      'Empty & Power OFF': Icons.hotel
    };

    // Calculate allowed grid columns dynamically based on screen width
    double screenWidth = MediaQuery.of(context).size.width;
    int maxCols = (screenWidth / 140).floor().clamp(1, 24); // max cols based on min width of 140px, capped at 24
    int columns = provider.gridCols.clamp(1, maxCols);

    return Scaffold(
        backgroundColor: isDark ? const Color(0xFF0F172A) : const Color(0xFFF1F5F9), // slate-900 / slate-100
        appBar: AppBar(
          backgroundColor: isDark ? const Color(0xFF1E293B) : Colors.white, // slate-800 / white
          title: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'KAI Dashboard',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: screenWidth < 500 ? 16 : 18,
                    ),
                  ),
                  Text(
                    screenWidth > 500
                        ? 'Host: ${provider.host} • ${camerasList.length} Cam${camerasList.length == 1 ? "" : "s"}'
                        : '${camerasList.length} Cam${camerasList.length == 1 ? "" : "s"} • ${provider.remainingSeconds}s',
                    style: TextStyle(
                      fontSize: 11,
                      color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF475569),
                    ),
                  ),
                ],
              ),
              const SizedBox(width: 6),
              // MQTT Connected dot
              Container(
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  color: provider.isMqttConnected ? Colors.green : Colors.amber,
                  shape: BoxShape.circle,
                ),
              ),
            ],
          ),
        actions: [
          // Density control (hidden if screen too small for grid layout)
          if (maxCols > 1)
            IconButton(
              icon: Icon(
                columns == 1
                    ? Icons.view_stream
                    : columns == 2
                        ? Icons.grid_view
                        : columns == 3
                            ? Icons.grid_on
                            : Icons.apps,
              ),
              onPressed: () => provider.cycleGridCols(maxCols: maxCols),
              tooltip: 'Cycle Grid Layout ($columns col${columns > 1 ? "s" : ""})',
            ),

          // Manual refresh
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => provider.forceRefresh(),
            tooltip: 'Immediate Refresh',
          ),

          // Timer/Refresh countdown Badge (hidden on mobile, integrated into subtitle)
          if (screenWidth > 500)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 12.0),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: isDark ? const Color(0xFF334155) : const Color(0xFFE2E8F0),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${provider.remainingSeconds}s',
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
                ),
              ),
            ),

          // Dropdown settings menu
          PopupMenuButton<String>(
            onSelected: (value) {
              if (value == 'settings') {
                _openSettingsDialog(context);
              } else if (value == 'theme') {
                provider.toggleTheme();
              } else if (value == 'logout') {
                provider.logout();
                Navigator.of(context).pushReplacementNamed('/login');
              }
            },
            itemBuilder: (_) => [
              const PopupMenuItem(
                value: 'settings',
                child: Row(
                  children: [
                    Icon(Icons.settings, size: 18),
                    SizedBox(width: 8),
                    Text('Update Timers'),
                  ],
                ),
              ),
              PopupMenuItem(
                value: 'theme',
                child: Row(
                  children: [
                    Icon(
                      provider.themeMode == ThemeMode.dark ? Icons.light_mode : Icons.dark_mode,
                      size: 18,
                    ),
                    const SizedBox(width: 8),
                    Text(provider.themeMode == ThemeMode.dark ? 'Switch to Light Mode' : 'Switch to Dark Mode'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'logout',
                child: Row(
                  children: [
                    Icon(Icons.logout, size: 18, color: Colors.redAccent),
                    SizedBox(width: 8),
                    Text('Switch Server / Logout', style: TextStyle(color: Colors.redAccent)),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
      body: Column(
        children: [
          // Filter Bar & Search Bar combined
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: isDark ? const Color(0xFF1E293B) : Colors.white,
              border: Border(
                bottom: BorderSide(
                  color: isDark ? const Color(0xFF334155) : const Color(0xFFCBD5E1),
                ),
              ),
            ),
            child: Row(
              children: [
                // Filter Dropdown Button
                PopupMenuButton<String>(
                  onSelected: (value) => provider.updateFilter(value),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                    decoration: BoxDecoration(
                      color: isDark ? const Color(0xFF334155) : const Color(0xFFF1F5F9),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: isDark ? const Color(0xFF475569) : const Color(0xFFCBD5E1),
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          filterIcons[provider.currentFilter] ?? Icons.tag,
                          size: 16,
                          color: Theme.of(context).colorScheme.primary,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Filter: ${provider.currentFilter}',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            color: isDark ? Colors.white : Colors.black87,
                          ),
                        ),
                        const SizedBox(width: 6),
                        const Icon(Icons.arrow_drop_down, size: 16),
                      ],
                    ),
                  ),
                  itemBuilder: (_) => filters
                      .map(
                        (f) => PopupMenuItem(
                          value: f,
                          child: Row(
                            children: [
                              Icon(filterIcons[f] ?? Icons.tag, size: 18),
                              const SizedBox(width: 10),
                              Text(f, style: const TextStyle(fontSize: 13)),
                            ],
                          ),
                        ),
                      )
                      .toList(),
                ),
                const SizedBox(width: 12),

                // Search Bar
                Expanded(
                  child: TextField(
                    controller: _searchController,
                    onChanged: (val) => provider.updateSearchQuery(val),
                    style: TextStyle(color: isDark ? Colors.white : Colors.black),
                    decoration: InputDecoration(
                      hintText: 'Search cameras by ID...',
                      prefixIcon: const Icon(Icons.search, size: 18),
                      suffixIcon: _searchController.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear, size: 18),
                              onPressed: () {
                                _searchController.clear();
                                provider.updateSearchQuery('');
                              },
                            )
                          : null,
                      contentPadding: const EdgeInsets.symmetric(vertical: 8),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: BorderSide(
                          color: isDark ? const Color(0xFF334155) : const Color(0xFFCBD5E1),
                        ),
                      ),
                      filled: true,
                      fillColor: isDark ? const Color(0xFF0F172A) : const Color(0xFFF8FAFC),
                    ),
                  ),
                ),
              ],
            ),
          ),

          if (!provider.isMqttConnected)
            GestureDetector(
              onTap: () {
                ScaffoldMessenger.of(context).clearSnackBars();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Attempting to reconnect MQTT...')),
                );
                provider.startDashboard();
              },
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.amber.withOpacity(0.12),
                  border: Border(
                    bottom: BorderSide(
                      color: isDark ? const Color(0xFF334155) : const Color(0xFFCBD5E1),
                    ),
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.wifi_off, color: Colors.amber, size: 18),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'MQTT Disconnected. Live controls are offline.',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: isDark ? Colors.amber[200] : Colors.amber[900],
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.amber.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(
                          color: Colors.amber.withOpacity(0.4),
                          width: 1,
                        ),
                      ),
                      child: Text(
                        'RECONNECT',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: isDark ? Colors.amber[100] : Colors.amber[950],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          // Grid View Area
          Expanded(
            child: camerasList.isNotEmpty
                ? GridView.builder(
                    padding: EdgeInsets.all(columns > 6 ? 8.0 : columns > 4 ? 12.0 : 16.0),
                    gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: columns,
                      crossAxisSpacing: columns > 6 ? 6.0 : columns > 4 ? 10.0 : 16.0,
                      mainAxisSpacing: columns > 6 ? 6.0 : columns > 4 ? 10.0 : 16.0,
                      childAspectRatio: columns == 1
                          ? (screenWidth < 500 ? 1.05 : 1.6) // squarish on mobile!
                          : columns == 2
                              ? (screenWidth < 500 ? 0.92 : 1.3) // squarish on mobile!
                              : columns == 3
                                  ? 1.2
                                  : columns == 4
                                      ? 1.05
                                      : columns == 5
                                          ? 0.95
                                          : columns == 6
                                              ? 0.9
                                              : 0.85,
                    ),
                    itemCount: camerasList.length,
                    itemBuilder: (context, index) {
                      return CameraCard(camera: camerasList[index]);
                    },
                  )
                : Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.videocam_off,
                          size: 64,
                          color: isDark ? const Color(0xFF475569) : const Color(0xFF94A3B8),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'No cameras match the query.',
                          style: TextStyle(
                            fontSize: 16,
                            color: isDark ? const Color(0xFF94A3B8) : const Color(0xFF475569),
                          ),
                        ),
                      ],
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}
