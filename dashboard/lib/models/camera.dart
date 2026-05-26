class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;
  final String label;
  final String colorHex;

  BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.label,
    required this.colorHex,
  });

  factory BoundingBox.fromJson(List<dynamic> json) {
    return BoundingBox(
      x: (json[0] as num).toDouble(),
      y: (json[1] as num).toDouble(),
      width: (json[2] as num).toDouble(),
      height: (json[3] as num).toDouble(),
      label: json[4] as String,
      colorHex: json[5] as String,
    );
  }
}

class Camera {
  final String id;
  final String status;
  final String imagePath;
  final String powerState;
  final List<BoundingBox> boundingBoxes;

  Camera({
    required this.id,
    required this.status,
    required this.imagePath,
    required this.powerState,
    required this.boundingBoxes,
  });

  Camera copyWith({
    String? id,
    String? status,
    String? imagePath,
    String? powerState,
    List<BoundingBox>? boundingBoxes,
  }) {
    return Camera(
      id: id ?? this.id,
      status: status ?? this.status,
      imagePath: imagePath ?? this.imagePath,
      powerState: powerState ?? this.powerState,
      boundingBoxes: boundingBoxes ?? this.boundingBoxes,
    );
  }

  factory Camera.fromJson(String id, Map<String, dynamic> json, {String powerState = 'ON'}) {
    var bboxesJson = json['bboxes'] as List<dynamic>? ?? [];
    List<BoundingBox> bboxes = bboxesJson
        .map((b) => BoundingBox.fromJson(b as List<dynamic>))
        .toList();

    return Camera(
      id: id,
      status: (json['status'] as String? ?? 'UNKNOWN').toUpperCase(),
      imagePath: json['image_path'] as String? ?? '',
      powerState: powerState,
      boundingBoxes: bboxes,
    );
  }
}
