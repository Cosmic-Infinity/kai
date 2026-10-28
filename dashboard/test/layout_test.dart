import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:kai_dashboard/models/camera.dart';
import 'package:kai_dashboard/providers/dashboard_provider.dart';
import 'package:kai_dashboard/widgets/camera_card.dart';

// Pixel 9-like phone: 412x915 logical pixels, edge-to-edge with status bar and gesture bar
const _statusBar = 52.0;
const _navBar = 48.0;

final _camera = Camera(
  id: 'CAM_TEST',
  status: 'NO',
  imagePath: '',
  powerState: 'ON',
  boundingBoxes: const [],
);

Future<void> _pumpPhone(WidgetTester tester, {required bool landscape}) async {
  final view = tester.view;
  view.devicePixelRatio = 2.625;
  view.physicalSize = landscape ? const Size(915 * 2.625, 412 * 2.625) : const Size(412 * 2.625, 915 * 2.625);
  view.padding = landscape
      ? const FakeViewPadding(left: _statusBar * 2.625, bottom: _navBar * 2.625)
      : const FakeViewPadding(top: _statusBar * 2.625, bottom: _navBar * 2.625);
  view.viewPadding = view.padding;
  addTearDown(view.reset);

  await tester.pumpWidget(
    ChangeNotifierProvider(
      create: (_) => DashboardProvider(),
      child: MaterialApp(
        home: Scaffold(
          body: Center(
            child: SizedBox(width: 300, height: 320, child: CameraCard(camera: _camera)),
          ),
        ),
      ),
    ),
  );
}

void main() {
  for (final landscape in [false, true]) {
    final orientation = landscape ? 'landscape' : 'portrait';

    testWidgets('Lightbox stays inside the safe area ($orientation)', (tester) async {
      await _pumpPhone(tester, landscape: landscape);

      // Open the lightbox from the card
      await tester.tap(find.text('CAM_TEST'));
      await tester.pumpAndSettle();

      final dialog = tester.getRect(find.byType(Dialog));
      final screen = tester.view.physicalSize / tester.view.devicePixelRatio;
      final safeTop = landscape ? 0.0 : _statusBar;
      final safeLeft = landscape ? _statusBar : 0.0;

      expect(dialog.top, greaterThanOrEqualTo(safeTop), reason: 'dialog runs under the status bar');
      expect(dialog.left, greaterThanOrEqualTo(safeLeft), reason: 'dialog runs under the camera cutout');
      expect(dialog.bottom, lessThanOrEqualTo(screen.height - _navBar), reason: 'dialog runs under the navigation bar');
      // pumpAndSettle would have thrown on a RenderFlex overflow; also check the controls are reachable
      expect(find.text('SHUTDOWN'), findsWidgets);
      await expectLater(tester, meetsGuideline(androidTapTargetGuideline));
      await expectLater(tester, meetsGuideline(labeledTapTargetGuideline));
    });
  }

  testWidgets('Lightbox scrolls instead of overflowing with large text in landscape', (tester) async {
    tester.platformDispatcher.textScaleFactorTestValue = 2.0;
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    await _pumpPhone(tester, landscape: true);

    await tester.tap(find.text('CAM_TEST'));
    await tester.pumpAndSettle(); // throws on RenderFlex overflow

    final scrollable = find.descendant(of: find.byType(Dialog), matching: find.byType(SingleChildScrollView));
    expect(scrollable, findsOneWidget);
    final shutdown = find.descendant(of: find.byType(Dialog), matching: find.text('SHUTDOWN'));
    await tester.dragUntilVisible(shutdown, scrollable, const Offset(0, -100));
    expect(shutdown.hitTestable(), findsOneWidget);
  });

  testWidgets('Camera card controls meet tap target and label guidelines', (tester) async {
    await _pumpPhone(tester, landscape: false);
    await expectLater(tester, meetsGuideline(androidTapTargetGuideline));
    await expectLater(tester, meetsGuideline(labeledTapTargetGuideline));
  });

  testWidgets('Compact card overlay controls are labelled', (tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => DashboardProvider(),
        child: MaterialApp(
          home: Scaffold(
            body: Center(child: SizedBox(width: 200, height: 220, child: CameraCard(camera: _camera))),
          ),
        ),
      ),
    );
    await expectLater(tester, meetsGuideline(labeledTapTargetGuideline));
    expect(find.bySemanticsLabel(RegExp('Status: EMPTY')), findsOneWidget);
  });
}
