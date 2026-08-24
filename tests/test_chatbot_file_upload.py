import io
import os
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import Workbook

from app import create_app
from extensions import db
from models.user import User


class ChatbotFileUploadTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = self.app.test_client()

        self.user = User(username='tester', email='tester@example.com')
        self.user.set_password('secret123')
        db.session.add(self.user)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_chat_can_receive_uploaded_excel_and_forward_content_to_ai(self):
        self.client.post('/login', data={
            'email': 'tester@example.com',
            'password': 'secret123'
        }, follow_redirects=True)

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = 'Sales'
        sheet.append(['Month', 'Sales'])
        sheet.append(['Jan', 1200])
        sheet.append(['Feb', 1500])
        sheet.append(['Mar', 1800])

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            workbook.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            with open(tmp_path, 'rb') as fh:
                with patch('routes.chatbot.ai_service.get_chat_response', return_value={'response': 'ok', 'response_time': 0.1}) as mock_ai:
                    response = self.client.post('/chat/send', data={
                        'message': 'What is the monthly sales for the first quarter?',
                        'files': (io.BytesIO(fh.read()), 'sales.xlsx')
                    }, content_type='multipart/form-data')

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload['success'])
            self.assertTrue(mock_ai.called)
            prompt_text = mock_ai.call_args.args[0]
            self.assertIn('sales.xlsx', prompt_text)
            self.assertIn('Jan', prompt_text)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
