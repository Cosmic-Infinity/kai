import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'providers/dashboard_provider.dart';
import 'screens/login_screen.dart';
import 'screens/dashboard_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Make status bar transparent and edge-to-edge
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: Color(0xFF0F172A),
  ));

  final dashboardProvider = DashboardProvider();
  await dashboardProvider.initialize();

  runApp(
    ChangeNotifierProvider(
      create: (_) => dashboardProvider,
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<DashboardProvider>(context);

    // Premium Material 3 styling
    return MaterialApp(
      title: 'KAI Dashboard',
      debugShowCheckedModeBanner: false,
      themeMode: provider.themeMode,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.light,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3B82F6), // Blue accent
          brightness: Brightness.light,
          primary: const Color(0xFF2563EB),
          surface: Colors.white,
        ),
        cardTheme: const CardThemeData(elevation: 2),
        appBarTheme: AppBarTheme(
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
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3B82F6), // Blue accent
          brightness: Brightness.dark,
          primary: const Color(0xFF3B82F6),
          surface: const Color(0xFF1E293B), // slate-800
          background: const Color(0xFF0F172A), // slate-900
        ),
        scaffoldBackgroundColor: const Color(0xFF0F172A),
        cardTheme: const CardThemeData(
          color: Color(0xFF1E293B),
          elevation: 6,
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: const Color(0xFF1E293B),
          centerTitle: false,
          elevation: 0,
          systemOverlayStyle: SystemUiOverlayStyle(
            statusBarColor: Colors.transparent,
            statusBarIconBrightness: Brightness.light,
            systemNavigationBarColor: const Color(0xFF0F172A),
          ),
        ),
      ),
      // Auto routing: if credentials exist, skip login screen and boot straight to dashboard
      home: provider.host.isNotEmpty && provider.username.isNotEmpty
          ? const DashboardScreen()
          : const LoginScreen(),
      routes: {
        '/login': (_) => const LoginScreen(),
        '/dashboard': (_) => const DashboardScreen(),
      },
    );
  }
}
