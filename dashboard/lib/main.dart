import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:dynamic_color/dynamic_color.dart';
import 'providers/dashboard_provider.dart';
import 'screens/login_screen.dart';
import 'screens/dashboard_screen.dart';

// ─── Cyber-Telemetry Premium Design Tokens ───────────────────────────────────
const _cosmicNavy = Color(0xFF161E2E);
const _cosmicSpace = Color(0xFF0B0F19);
const _electricCyan = Color(0xFF00E5FF);
const _electricIndigo = Color(0xFF4F46E5);
// ─────────────────────────────────────────────────────────────────────────────

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Edge-to-edge: transparent status bar
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: _cosmicSpace,
  ));

  final provider = DashboardProvider();
  await provider.initialize();

  runApp(
    ChangeNotifierProvider.value(
      value: provider,
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Only rebuild MaterialApp when themeMode or login state changes, not on every timer/camera update
    final themeMode = context.select<DashboardProvider, ThemeMode>((p) => p.themeMode);
    final hasHost = context.select<DashboardProvider, bool>((p) => p.host.isNotEmpty);

    return DynamicColorBuilder(
      builder: (ColorScheme? lightDynamic, ColorScheme? darkDynamic) {
        // Fallback color schemes if Material You is unavailable (e.g. older Android versions or Windows)
        final lightScheme = lightDynamic ?? ColorScheme.fromSeed(
          seedColor: _electricIndigo,
          brightness: Brightness.light,
          primary: _electricIndigo,
          surface: Colors.white,
        );

        final darkScheme = darkDynamic ?? ColorScheme.fromSeed(
          seedColor: _electricCyan,
          brightness: Brightness.dark,
          primary: _electricCyan,
          surface: _cosmicNavy,
        );

        return MaterialApp(
          title: 'kai Dashboard',
          debugShowCheckedModeBanner: false,
          themeMode: themeMode,

          theme: ThemeData(
            useMaterial3: true,
            colorScheme: lightScheme,
            cardTheme: const CardThemeData(elevation: 2),
            appBarTheme: const AppBarTheme(
              centerTitle: false,
              elevation: 0,
              systemOverlayStyle: SystemUiOverlayStyle(
                statusBarColor: Colors.transparent,
                statusBarIconBrightness: Brightness.dark,
                systemNavigationBarColor: Colors.white,
              ),
            ),
          ),

          darkTheme: ThemeData(
            useMaterial3: true,
            colorScheme: darkScheme,
            scaffoldBackgroundColor: darkDynamic != null ? null : _cosmicSpace,
            cardTheme: CardThemeData(
              color: darkDynamic != null ? null : _cosmicNavy,
              elevation: 4,
            ),
            appBarTheme: AppBarTheme(
              backgroundColor: darkDynamic != null ? null : _cosmicNavy,
              centerTitle: false,
              elevation: 0,
              systemOverlayStyle: const SystemUiOverlayStyle(
                statusBarColor: Colors.transparent,
                statusBarIconBrightness: Brightness.light,
                systemNavigationBarColor: _cosmicSpace,
              ),
            ),
          ),

          home: hasHost ? const DashboardScreen() : const LoginScreen(),
          routes: {
            '/login': (_) => const LoginScreen(),
            '/dashboard': (_) => const DashboardScreen(),
          },
        );
      },
    );
  }
}
