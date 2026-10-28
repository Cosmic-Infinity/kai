import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:kai_dashboard/main.dart';
import 'package:kai_dashboard/providers/dashboard_provider.dart';

void main() {
  testWidgets('Shows the login screen when no server is configured', (WidgetTester tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => DashboardProvider(),
        child: const MyApp(),
      ),
    );

    expect(find.text('kai Dashboard'), findsOneWidget);
    expect(find.text('User Login'), findsOneWidget);
  });
}
