# EY Project Documentation

## 1. Project Overview
EY Project is a Flask-based AI assistant application that lets authenticated users chat with an AI, upload files for analysis, view analytics, and export chat conversations.

## 2. Main Features
- User registration and login
- AI chatbot conversations
- File upload and analysis for PDF, DOCX, TXT, CSV, XLSX, and XLS files
- AI-powered question answering over uploaded content
- Analytics dashboard for activity tracking
- Conversation export to TXT, DOCX, and PDF

## 3. Project Structure
- app.py: Application factory and app initialization
- config.py: Configuration and environment settings
- extensions.py: Flask extensions initialization
- routes/: HTTP route handlers for auth, chatbot, uploads, and analytics
- services/: AI, file extraction, and export logic
- models/: Database models for users, chats, uploads, and analytics
- templates/: HTML templates for the web UI
- uploads/: Storage for uploaded files
- tests/: Regression tests for core features

## 4. Setup Instructions
1. Install Python 3.13+.
2. Create a virtual environment if desired.
3. Install dependencies:
   - pip install -r requirements.txt
4. Set environment variables in a .env file:
   - GEMINI_API_KEY=your_key_here
   - FLASK_SECRET_KEY=change-me
5. Run the application:
   - python app.py
6. Open http://127.0.0.1:5000/

## 5. How the Chatbot Works
1. A user opens the chatbot page.
2. They can type a prompt and optionally attach one or more files.
3. The backend extracts text from supported files and sends the prompt plus extracted content to the AI service.
4. The AI response is saved to the conversation history and shown in the UI.

## 6. How File Uploads Work
1. The upload page accepts supported files.
2. The server stores the file temporarily, extracts readable content, and creates an AI summary.
3. Uploaded file details are stored in the database for later use.
4. Users can ask questions about the uploaded content through the AI assistant.

## 7. Example Usage
Example prompt for a sales workbook:
- Upload a sales Excel file.
- Ask: "What is the monthly sales of the first quarter?"
- The assistant uses the uploaded file content to answer.

## 8. Testing
The project includes a regression test for chatbot file uploads.
Run:
- python -m unittest discover -s tests -v

## 9. Notes
- The application uses SQLite by default.
- AI responses depend on the GEMINI_API_KEY being configured correctly.
- If the key is missing, the app returns a helpful fallback message.
