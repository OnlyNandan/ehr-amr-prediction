"""
Standalone YOLO Inference Server
For deployment as separate microservice
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from services.yolo_inference import BloodSmearAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize analyzer
analyzer = BloodSmearAnalyzer(
    model_path=os.getenv("YOLO_MODEL_PATH"),
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
    iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.45"))
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "yolo-inference"})


@app.route("/analyze", methods=["POST"])
def analyze_image():
    """
    Analyze blood smear image.
    
    Request body:
    {
        "image_base64": "base64_encoded_image"
    }
    """
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "image_base64 required"}), 400
        
        result = analyzer.analyze(data["image_base64"])
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
