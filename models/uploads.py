"""
Upload Model
============
Defines the UploadedFile model for tracking user file uploads
and their associated metadata including AI-generated summaries.
"""

from datetime import datetime, timezone
from extensions import db


class UploadedFile(db.Model):
    """
    Represents a file uploaded by a user for AI analysis.

    Attributes:
        id: Primary key identifier.
        user_id: Foreign key linking to the uploading user.
        filename: Original name of the uploaded file.
        stored_filename: Server-side stored filename (to avoid conflicts).
        file_type: MIME type or extension of the file.
        file_size: Size of the file in bytes.
        extracted_text: Text content extracted from the file.
        summary: AI-generated summary of the file contents.
        upload_date: Timestamp of when the file was uploaded.
    """
    __tablename__ = 'uploaded_files'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    file_size = db.Column(db.Integer, default=0)
    extracted_text = db.Column(db.Text, default='')
    summary = db.Column(db.Text, default='')
    upload_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Serialize the upload record to a dictionary."""
        return {
            'id': self.id,
            'filename': self.filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'summary': self.summary,
            'upload_date': self.upload_date.strftime('%Y-%m-%d %H:%M:%S') if self.upload_date else None
        }

    def __repr__(self):
        return f'<UploadedFile {self.filename}>'
