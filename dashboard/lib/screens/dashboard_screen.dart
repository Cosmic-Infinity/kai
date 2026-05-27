import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/dashboard_provider.dart';
import '../widgets/camera_card.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with WidgetsBindingObserver {
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

  // Open System settings modal dialog
  void _openSettingsDialog(BuildContext context) async {
    final provider = Provider.of<DashboardProvider>(context, listen: false);
    final scaffoldMessenger = ScaffoldMessenger.of(
      context,
    ); // capture before await
    final currentConfig = await provider.fetchSystemConfig();

    if (currentConfig == null) {
      if (mounted) {
        scaffoldMessenger.showSnackBar(
          const SnackBar(
            content: Text('Failed to load system settings from server'),
          ),
        );
      }
      return;
    }

    if (!mounted) return;

    showDialog(
      context: context,
      builder: (dialogContext) {
        return _SettingsDialog(
          currentConfig: currentConfig,
          provider: provider,
        );
      },
    );
  }

  // Open Theme selection dialog
  void _openThemeDialog(BuildContext context) {
    final provider = Provider.of<DashboardProvider>(context, listen: false);
    final isDark = Theme.of(context).brightness == Brightness.dark;

    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: isDark ? const Color(0xFF161E2E) : Colors.white,
          title: const Text(
            'Select Theme Mode',
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              RadioListTile<ThemeMode>(
                title: const Text('System Default (Auto)'),
                value: ThemeMode.system,
                groupValue: provider.themeMode,
                activeColor: Theme.of(context).colorScheme.primary,
                onChanged: (mode) {
                  if (mode != null) {
                    provider.setThemeMode(mode);
                    Navigator.of(dialogContext).pop();
                  }
                },
              ),
              RadioListTile<ThemeMode>(
                title: const Text('Light Mode'),
                value: ThemeMode.light,
                groupValue: provider.themeMode,
                activeColor: Theme.of(context).colorScheme.primary,
                onChanged: (mode) {
                  if (mode != null) {
                    provider.setThemeMode(mode);
                    Navigator.of(dialogContext).pop();
                  }
                },
              ),
              RadioListTile<ThemeMode>(
                title: const Text('Dark Mode'),
                value: ThemeMode.dark,
                groupValue: provider.themeMode,
                activeColor: Theme.of(context).colorScheme.primary,
                onChanged: (mode) {
                  if (mode != null) {
                    provider.setThemeMode(mode);
                    Navigator.of(dialogContext).pop();
                  }
                },
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(),
              child: const Text('CANCEL'),
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

    // Calculate allowed grid columns dynamically based on screen width
    double screenWidth = MediaQuery.of(context).size.width;
    int maxCols = (screenWidth / 140).floor().clamp(1, 24); // max cols based on min width of 140px, capped at 24
    int defaultCols = screenWidth < 500 ? 1 : 3;
    int colsToUse = provider.gridCols == 0 ? defaultCols : provider.gridCols;
    int columns = colsToUse.clamp(1, maxCols);

    return Scaffold(
      backgroundColor: isDark
          ? const Color(0xFF0B0F19)
          : const Color(0xFFF3F4F6), // Deep Cosmic Space / Light Grey
      appBar: AppBar(
        backgroundColor: isDark
            ? const Color(0xFF161E2E)
            : Colors.white, // Sleek Navy / White
        title: Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'KAI Dashboard',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: screenWidth < 500 ? 16 : 18,
                  ),
                ),
                ValueListenableBuilder<int>(
                  valueListenable: provider.remainingSecondsNotifier,
                  builder: (context, remaining, child) {
                    return Text(
                      screenWidth > 500
                          ? 'Host: ${provider.host} • ${camerasList.length} Cam${camerasList.length == 1 ? "" : "s"}'
                          : '${camerasList.length} Cam${camerasList.length == 1 ? "" : "s"} • ${remaining}s',
                      style: TextStyle(
                        fontSize: 11,
                        color: isDark
                            ? const Color(0xFF9CA3AF)
                            : const Color(0xFF4B5563),
                      ),
                    );
                  },
                ),
              ],
            ),
            const SizedBox(width: 10),
            // MQTT Connected Status Pill
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: provider.isMqttConnected
                    ? Colors.green.withValues(alpha: 0.12)
                    : Colors.amber.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: provider.isMqttConnected
                      ? Colors.green.withValues(alpha: 0.3)
                      : Colors.amber.withValues(alpha: 0.3),
                  width: 1,
                ),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: provider.isMqttConnected
                          ? Colors.green
                          : Colors.amber,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    provider.isMqttConnected ? 'ONLINE' : 'OFFLINE',
                    style: TextStyle(
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5,
                      color: provider.isMqttConnected
                          ? (isDark ? Colors.green[300] : Colors.green[700])
                          : (isDark ? Colors.amber[300] : Colors.amber[800]),
                    ),
                  ),
                ],
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
              tooltip:
                  'Cycle Grid Layout ($columns col${columns > 1 ? "s" : ""})',
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
              padding: const EdgeInsets.symmetric(
                horizontal: 8.0,
                vertical: 12.0,
              ),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: isDark
                      ? const Color(0xFF334155)
                      : const Color(0xFFE2E8F0),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ValueListenableBuilder<int>(
                  valueListenable: provider.remainingSecondsNotifier,
                  builder: (context, remaining, child) {
                    return Text(
                      '${remaining}s',
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 12,
                      ),
                    );
                  },
                ),
              ),
            ),

          // Dropdown settings menu
          PopupMenuButton<String>(
            onSelected: (value) {
              if (value == 'settings') {
                _openSettingsDialog(context);
              } else if (value == 'theme') {
                _openThemeDialog(context);
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
              const PopupMenuItem(
                value: 'theme',
                child: Row(
                  children: [
                    Icon(Icons.palette_outlined, size: 18),
                    SizedBox(width: 8),
                    Text('Select Theme Mode'),
                  ],
                ),
              ),
              const PopupMenuItem(
                value: 'logout',
                child: Row(
                  children: [
                    Icon(Icons.logout, size: 18, color: Colors.redAccent),
                    SizedBox(width: 8),
                    Text(
                      'Switch Server / Logout',
                      style: TextStyle(color: Colors.redAccent),
                    ),
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
              color: isDark ? const Color(0xFF161E2E) : Colors.white,
              border: Border(
                bottom: BorderSide(
                  color: isDark
                      ? const Color(0xFF222F43)
                      : const Color(0xFFCBD5E1),
                ),
              ),
            ),
            child: Row(
              children: [
                // Filter dropdown with Multi-select checkable options
                MenuAnchor(
                  consumeOutsideTap: true,
                  style: MenuStyle(
                    backgroundColor: WidgetStateProperty.all(
                      isDark ? const Color(0xFF161E2E) : Colors.white,
                    ),
                    elevation: WidgetStateProperty.all(8),
                    shape: WidgetStateProperty.all(
                      RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                        side: BorderSide(
                          color: isDark
                              ? const Color(0xFF222F43)
                              : const Color(0xFFCBD5E1),
                        ),
                      ),
                    ),
                  ),
                  builder: (context, controller, child) {
                    final bool isNarrow = screenWidth < 550;
                    final activeFiltersCount = provider.selectedFilters.length;
                    final filterButtonLabel = isNarrow
                        ? (activeFiltersCount == 0
                              ? 'Filter'
                              : 'Filter ($activeFiltersCount)')
                        : 'Filter: ${provider.filterLabel}';

                    return InkWell(
                      onTap: () {
                        if (controller.isOpen) {
                          controller.close();
                        } else {
                          controller.open();
                        }
                      },
                      borderRadius: BorderRadius.circular(10),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 10,
                        ),
                        decoration: BoxDecoration(
                          color: provider.selectedFilters.isNotEmpty
                              ? Theme.of(
                                  context,
                                ).colorScheme.primary.withValues(alpha: 0.15)
                              : (isDark
                                    ? const Color(0xFF222F43)
                                    : const Color(0xFFF1F5F9)),
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: provider.selectedFilters.isNotEmpty
                                ? Theme.of(
                                    context,
                                  ).colorScheme.primary.withValues(alpha: 0.5)
                                : (isDark
                                      ? const Color(0xFF37465F)
                                      : const Color(0xFFCBD5E1)),
                            width: provider.selectedFilters.isNotEmpty
                                ? 1.5
                                : 1,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.filter_alt_outlined,
                              size: 16,
                              color: provider.selectedFilters.isNotEmpty
                                  ? Theme.of(context).colorScheme.primary
                                  : (isDark
                                        ? const Color(0xFF9CA3AF)
                                        : const Color(0xFF37465F)),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              filterButtonLabel,
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                                color: provider.selectedFilters.isNotEmpty
                                    ? Theme.of(context).colorScheme.primary
                                    : (isDark ? Colors.white : Colors.black87),
                              ),
                            ),
                            const SizedBox(width: 6),
                            Icon(
                              Icons.arrow_drop_down,
                              size: 16,
                              color: provider.selectedFilters.isNotEmpty
                                  ? Theme.of(context).colorScheme.primary
                                  : (isDark ? Colors.white70 : Colors.black54),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                  menuChildren: [
                    Consumer<DashboardProvider>(
                      builder: (context, menuProvider, _) {
                        final activeFilters = menuProvider.selectedFilters;
                        return Container(
                          width: 280,
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    'OCCUPANCY STATUS',
                                    style: TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.bold,
                                      letterSpacing: 0.5,
                                      color: isDark
                                          ? const Color(0xFF94A3B8)
                                          : const Color(0xFF475569),
                                    ),
                                  ),
                                  if (activeFilters.isNotEmpty)
                                    TextButton.icon(
                                      onPressed: () =>
                                          menuProvider.clearFilters(),
                                      icon: const Icon(
                                        Icons.clear_all,
                                        size: 14,
                                      ),
                                      label: const Text(
                                        'Reset',
                                        style: TextStyle(
                                          fontSize: 11,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                      style: TextButton.styleFrom(
                                        foregroundColor: Colors.redAccent,
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 8,
                                          vertical: 4,
                                        ),
                                        minimumSize: Size.zero,
                                        tapTargetSize:
                                            MaterialTapTargetSize.shrinkWrap,
                                      ),
                                    ),
                                ],
                              ),
                              const SizedBox(height: 4),
                              CheckboxListTile(
                                title: const Text(
                                  'Occupied',
                                  style: TextStyle(fontSize: 13),
                                ),
                                value: activeFilters.contains('Occupied'),
                                dense: true,
                                controlAffinity:
                                    ListTileControlAffinity.leading,
                                activeColor: Theme.of(
                                  context,
                                ).colorScheme.primary,
                                contentPadding: EdgeInsets.zero,
                                onChanged: (val) {
                                  menuProvider.toggleFilter('Occupied');
                                },
                              ),
                              CheckboxListTile(
                                title: const Text(
                                  'Empty',
                                  style: TextStyle(fontSize: 13),
                                ),
                                value: activeFilters.contains('Empty'),
                                dense: true,
                                controlAffinity:
                                    ListTileControlAffinity.leading,
                                activeColor: Theme.of(
                                  context,
                                ).colorScheme.primary,
                                contentPadding: EdgeInsets.zero,
                                onChanged: (val) {
                                  menuProvider.toggleFilter('Empty');
                                },
                              ),
                              const Divider(height: 16),
                              Text(
                                'POWER STATE',
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.5,
                                  color: isDark
                                      ? const Color(0xFF94A3B8)
                                      : const Color(0xFF475569),
                                ),
                              ),
                              const SizedBox(height: 4),
                              CheckboxListTile(
                                title: const Text(
                                  'Power ON',
                                  style: TextStyle(fontSize: 13),
                                ),
                                value: activeFilters.contains('Power ON'),
                                dense: true,
                                controlAffinity:
                                    ListTileControlAffinity.leading,
                                activeColor: Theme.of(
                                  context,
                                ).colorScheme.primary,
                                contentPadding: EdgeInsets.zero,
                                onChanged: (val) {
                                  menuProvider.toggleFilter('Power ON');
                                },
                              ),
                              CheckboxListTile(
                                title: const Text(
                                  'Power OFF',
                                  style: TextStyle(fontSize: 13),
                                ),
                                value: activeFilters.contains('Power OFF'),
                                dense: true,
                                controlAffinity:
                                    ListTileControlAffinity.leading,
                                activeColor: Theme.of(
                                  context,
                                ).colorScheme.primary,
                                contentPadding: EdgeInsets.zero,
                                onChanged: (val) {
                                  menuProvider.toggleFilter('Power OFF');
                                },
                              ),
                              const Divider(height: 16),
                              Text(
                                'QUICK PRESETS',
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 0.5,
                                  color: isDark
                                      ? const Color(0xFF94A3B8)
                                      : const Color(0xFF475569),
                                ),
                              ),
                              const SizedBox(height: 8),
                              Wrap(
                                spacing: 6,
                                runSpacing: 6,
                                children: [
                                  ActionChip(
                                    avatar: const Icon(
                                      Icons.flash_on,
                                      size: 12,
                                      color: Colors.green,
                                    ),
                                    label: const Text(
                                      'Occupied & ON',
                                      style: TextStyle(fontSize: 10),
                                    ),
                                    padding: EdgeInsets.zero,
                                    materialTapTargetSize:
                                        MaterialTapTargetSize.shrinkWrap,
                                    onPressed: () {
                                      menuProvider.setQuickFilter([
                                        'Occupied',
                                        'Power ON',
                                      ]);
                                    },
                                  ),
                                  ActionChip(
                                    avatar: const Icon(
                                      Icons.hotel,
                                      size: 12,
                                      color: Colors.amber,
                                    ),
                                    label: const Text(
                                      'Empty & OFF',
                                      style: TextStyle(fontSize: 10),
                                    ),
                                    padding: EdgeInsets.zero,
                                    materialTapTargetSize:
                                        MaterialTapTargetSize.shrinkWrap,
                                    onPressed: () {
                                      menuProvider.setQuickFilter([
                                        'Empty',
                                        'Power OFF',
                                      ]);
                                    },
                                  ),
                                ],
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ],
                ),
                const SizedBox(width: 12),

                // Search Bar
                Expanded(
                  child: TextField(
                    controller: _searchController,
                    onChanged: (val) => provider.updateSearchQuery(val),
                    style: TextStyle(
                      color: isDark ? Colors.white : Colors.black,
                    ),
                    decoration: InputDecoration(
                      hintText: 'Camera ID...',
                      prefixIcon: const Padding(
                        padding: EdgeInsets.only(left: 12, right: 8),
                        child: Icon(Icons.search, size: 20),
                      ),
                      prefixIconConstraints: const BoxConstraints(
                        minWidth: 40,
                        minHeight: 40,
                      ),
                      suffixIcon: _searchController.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear, size: 18),
                              onPressed: () {
                                _searchController.clear();
                                provider.updateSearchQuery('');
                              },
                            )
                          : null,
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                        borderSide: BorderSide(
                          color: isDark
                              ? const Color(0xFF222F43)
                              : const Color(0xFFCBD5E1),
                        ),
                      ),
                      filled: true,
                      fillColor: isDark
                          ? const Color(0xFF0B0F19)
                          : const Color(0xFFF8FAFC),
                    ),
                  ),
                ),
              ],
            ),
          ),

          if (provider.selectedFilters.isNotEmpty)
            Container(
               padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: isDark
                    ? const Color(0xFF0B0F19)
                    : const Color(0xFFF8FAFC),
                border: Border(
                  bottom: BorderSide(
                    color: isDark
                        ? const Color(0xFF161E2E)
                        : const Color(0xFFCBD5E1),
                  ),
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    size: 14,
                    color: Theme.of(context).colorScheme.primary,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Showing filtered cameras (${camerasList.length} of ${provider.cameras.length}). Active: ${provider.filterLabel}',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w500,
                        color: isDark
                            ? const Color(0xFF9CA3AF)
                            : const Color(0xFF37465F),
                      ),
                    ),
                  ),
                  GestureDetector(
                    onTap: () => provider.clearFilters(),
                    child: Text(
                      'Clear',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).colorScheme.primary,
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
                  const SnackBar(
                    content: Text('Attempting to reconnect MQTT...'),
                  ),
                );
                provider.startDashboard();
              },
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 10,
                ),
                decoration: BoxDecoration(
                  color: Colors.amber.withValues(alpha: 0.12),
                  border: Border(
                    bottom: BorderSide(
                      color: isDark
                          ? const Color(0xFF334155)
                          : const Color(0xFFCBD5E1),
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
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.amber.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(
                          color: Colors.amber.withValues(alpha: 0.4),
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
                    padding: EdgeInsets.all(
                      columns > 6
                          ? 8.0
                          : columns > 4
                          ? 12.0
                          : 16.0,
                    ),
                    gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: columns,
                      crossAxisSpacing: columns > 6
                          ? 6.0
                          : columns > 4
                          ? 10.0
                          : 16.0,
                      mainAxisSpacing: columns > 6
                          ? 6.0
                          : columns > 4
                          ? 10.0
                          : 16.0,
                      childAspectRatio: columns == 1
                          ? (screenWidth < 500
                                ? 1.05
                                : 1.6) // squarish on mobile!
                          : columns == 2
                          ? (screenWidth < 500
                                ? 0.92
                                : 1.3) // squarish on mobile!
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
                          color: isDark
                              ? const Color(0xFF475569)
                              : const Color(0xFF94A3B8),
                        ),
                        const SizedBox(height: 12),
                        Text(
                          'No cameras match the query.',
                          style: TextStyle(
                            fontSize: 16,
                            color: isDark
                                ? const Color(0xFF94A3B8)
                                : const Color(0xFF475569),
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

class _SettingsDialog extends StatefulWidget {
  final Map<String, dynamic> currentConfig;
  final DashboardProvider provider;

  const _SettingsDialog({
    Key? key,
    required this.currentConfig,
    required this.provider,
  }) : super(key: key);

  @override
  State<_SettingsDialog> createState() => _SettingsDialogState();
}

class _SettingsDialogState extends State<_SettingsDialog> {
  late TextEditingController imageIntervalController;
  late TextEditingController dashIntervalController;
  late TextEditingController controlIntervalController;
  late TextEditingController thresholdController;
  late TextEditingController keepaliveController;
  late TextEditingController reconnectController;

  @override
  void initState() {
    super.initState();
    imageIntervalController = TextEditingController(
      text: widget.currentConfig['IMAGE_SERVER_INTERVAL']?.toString() ?? '60',
    );
    dashIntervalController = TextEditingController(
      text: widget.currentConfig['DASHBOARD_INTERVAL']?.toString() ?? '30',
    );
    controlIntervalController = TextEditingController(
      text: widget.currentConfig['CONTROL_SERVER_INTERVAL']?.toString() ?? '30',
    );
    thresholdController = TextEditingController(
      text: widget.currentConfig['INACTIVITY_THRESHOLD']?.toString() ?? '10',
    );
    keepaliveController = TextEditingController(
      text: widget.currentConfig['MQTT_KEEPALIVE']?.toString() ?? '120',
    );
    reconnectController = TextEditingController(
      text: widget.currentConfig['MQTT_RECONNECT_DELAY']?.toString() ?? '2',
    );
  }

  @override
  void dispose() {
    imageIntervalController.dispose();
    dashIntervalController.dispose();
    controlIntervalController.dispose();
    thresholdController.dispose();
    keepaliveController.dispose();
    reconnectController.dispose();
    super.dispose();
  }

  Widget _buildGroupHeader(
    BuildContext context,
    IconData icon,
    String title,
    bool isDark,
  ) {
    return Padding(
      padding: const EdgeInsets.only(top: 8.0, bottom: 6.0),
      child: Row(
        children: [
          Icon(
            icon,
            size: 16,
            color: isDark
                ? const Color(0xFF60A5FA)
                : Theme.of(context).colorScheme.primary,
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
                  insetPadding: const EdgeInsets.symmetric(
                    horizontal: 56.0,
                    vertical: 24.0,
                  ),
                  title: Row(
                    children: [
                      const Icon(Icons.info_outline, color: Colors.blue),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          label,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
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

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return AlertDialog(
      title: const Text(
        'System Configuration',
        style: TextStyle(fontWeight: FontWeight.bold),
      ),
      content: SizedBox(
        width: 400,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildGroupHeader(
                context,
                Icons.camera_enhance_outlined,
                'Image Processing Module',
                isDark,
              ),
              const SizedBox(height: 4),
              _buildConfigField(
                context,
                'Image Capture Interval (s)',
                imageIntervalController,
                'Interval (in seconds) at which the Image Server captures and analyzes new camera frames.',
              ),
              const SizedBox(height: 16),

              _buildGroupHeader(
                context,
                Icons.dashboard_outlined,
                'Dashboard UI Module',
                isDark,
              ),
              const SizedBox(height: 4),
              _buildConfigField(
                context,
                'Dashboard Refresh Interval (s)',
                dashIntervalController,
                'Interval (in seconds) at which the Dashboard UI automatically fetches the latest camera images.',
              ),
              const SizedBox(height: 16),

              _buildGroupHeader(
                context,
                Icons.settings_suggest_outlined,
                'Control Server Module',
                isDark,
              ),
              const SizedBox(height: 4),
              _buildConfigField(
                context,
                'Control Server Loop Interval (s)',
                controlIntervalController,
                'Interval (in seconds) at which the Control Server checks camera statuses to make power decisions.',
              ),
              const SizedBox(height: 10),
              _buildConfigField(
                context,
                'Inactivity Shutdown Threshold',
                thresholdController,
                'Number of consecutive "Empty" readings before room appliances are automatically turned off.',
              ),
              const SizedBox(height: 16),

              _buildGroupHeader(
                context,
                Icons.wifi_tethering_outlined,
                'Network & MQTT Broker',
                isDark,
              ),
              const SizedBox(height: 4),
              _buildConfigField(
                context,
                'MQTT Keepalive (s)',
                keepaliveController,
                'Maximum time (in seconds) allowed between client-broker messages before connection loss is assumed.',
              ),
              const SizedBox(height: 10),
              _buildConfigField(
                context,
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
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('CANCEL'),
        ),
        ElevatedButton(
          onPressed: () async {
            final scaffoldMessenger = ScaffoldMessenger.of(context);
            final nav = Navigator.of(context);
            final settings = {
              'IMAGE_SERVER_INTERVAL':
                  int.tryParse(imageIntervalController.text) ?? 60,
              'DASHBOARD_INTERVAL':
                  int.tryParse(dashIntervalController.text) ?? 30,
              'CONTROL_SERVER_INTERVAL':
                  int.tryParse(controlIntervalController.text) ?? 30,
              'INACTIVITY_THRESHOLD':
                  int.tryParse(thresholdController.text) ?? 10,
              'MQTT_KEEPALIVE': int.tryParse(keepaliveController.text) ?? 120,
              'MQTT_RECONNECT_DELAY':
                  int.tryParse(reconnectController.text) ?? 2,
            };
            final success = await widget.provider.saveSystemSettings(settings);
            if (success) {
              nav.pop();
              scaffoldMessenger.showSnackBar(
                const SnackBar(content: Text('Settings saved successfully')),
              );
            } else {
              scaffoldMessenger.showSnackBar(
                const SnackBar(content: Text('Error updating server settings')),
              );
            }
          },
          child: const Text('SAVE'),
        ),
      ],
    );
  }
}
