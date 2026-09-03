"""
Dedicated financial chart API.
"""

import os
import tempfile

from flask import Blueprint, current_app, jsonify, request
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from models.uploads import UploadedFile
from services.file_service import allowed_file, generate_stored_filename
from services.insight_service import analyze_file_query


charts_bp = Blueprint("charts", __name__)


@charts_bp.route("/api/charts/generate", methods=["POST"])
@login_required
def generate_chart_endpoint():
    """
    Generate a financial chart from an uploaded CSV/Excel file.

    Preferred request:
      file_id=<existing UploadedFile id>
      question=<natural-language chart request>

    The endpoint also accepts a new multipart 'file' for standalone use.
    The response contains a base64 data URI, so the browser does not need
    server-side chart storage.
    """
    question = (
        request.form.get("question")
        or request.form.get("message")
        or ""
    ).strip()

    if not question:
        return jsonify({
            "success": False,
            "error": "A chart question is required."
        }), 400

    file_id = request.form.get("file_id", type=int)
    temporary_path = None

    try:
        if file_id:
            record = UploadedFile.query.filter_by(
                id=file_id,
                user_id=current_user.id
            ).first()

            if not record:
                return jsonify({
                    "success": False,
                    "error": "Uploaded file not found."
                }), 404

            filepath = os.path.join(
                current_app.config["UPLOAD_FOLDER"],
                record.stored_filename
            )

            if not os.path.exists(filepath):
                return jsonify({
                    "success": False,
                    "error": (
                        "The uploaded file is no longer available on the "
                        "server. Please upload it again."
                    )
                }), 410

            file_type = record.file_type

        else:
            uploaded = request.files.get("file")
            if not uploaded or not uploaded.filename:
                return jsonify({
                    "success": False,
                    "error": "Provide file_id or upload a CSV/Excel file."
                }), 400

            if not allowed_file(uploaded.filename):
                return jsonify({
                    "success": False,
                    "error": "Only CSV, XLSX and XLS files are supported for charts."
                }), 400

            file_type = secure_filename(uploaded.filename).rsplit(".", 1)[1].lower()

            if file_type not in {"csv", "xlsx", "xls"}:
                return jsonify({
                    "success": False,
                    "error": "Financial charts require CSV or Excel files."
                }), 400

            suffix = f".{file_type}"
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temporary_path = temp.name
            temp.close()
            uploaded.save(temporary_path)
            filepath = temporary_path

        result = analyze_file_query(filepath, file_type, question)

        return jsonify({
            "success": True,
            "chart": result["chart"],
            "insights": result["insights"],
            "analysis": {
                "metric_column": result["metric_column"],
                "group": result["group"],
                "labels": result["labels"],
                "values": result["values"],
            }
        })

    except ValueError as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 422
    except Exception:
        current_app.logger.exception("Chart generation failed")
        return jsonify({
            "success": False,
            "error": "Unable to generate the chart from this file."
        }), 500
    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
