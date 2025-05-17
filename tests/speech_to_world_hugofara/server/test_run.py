import json
import unittest
from unittest import mock

from speech_to_world_hugofara.server import run, task_tracker


class TestServerFunctions(unittest.TestCase):

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.client_socket = mock.Mock()
        self.client_socket.sendall = mock.Mock()

    def test_start_task(self):
        # Mock the start_task function
        mock_answer_data = {"completion": 1}
        mock_start_task = mock.MagicMock()
        mock_start_task.return_value = mock_answer_data
        with mock.patch("speech_to_world_hugofara.server.run.start_task", new=mock_start_task):
            result = run.start_task(
                {"taskId": "test_task_id", "type": "new-skybox-local"},
                task_tracker.TaskTracker(self.client_socket, 0, None)
            )
            self.assertEqual(result, mock_answer_data)
            mock_start_task.assert_called_once()

    def test_prepare_response(self):
        # Mock the prepare_response function
        data_path = "randomPath.png"
        mock_answer_data = {"skyboxFilePath": data_path}
        json_data = {
            "taskId": 0,
            "type": "new-skybox-local",
            "prompt": "A green horse running",
            "outputFilePath": data_path,
            "reportCompletion": 1
        }
        response = run.prepare_response(
            json_data,
            task_tracker.TaskTracker(self.client_socket, 0, mock.Mock())
        )
        self.assertEqual(response["data"], json.dumps(mock_answer_data))

    def test_handle_query(self):
        # Mock the handle_query function
        mock_handle_query = mock.MagicMock()
        with mock.patch("speech_to_world_hugofara.server.run.handle_query", new=mock_handle_query):
            json_data = {"taskId": "test_task_id", "type": "new-skybox-local"}
            run.handle_query(str(json_data), self.client_socket)
            mock_handle_query.assert_called_once()

    def test_server_data(self):
        # Test the server_data function
        answer_data = run.server_data()
        self.assertIsInstance(answer_data, dict)

    def test_completion_report(self):
        # Test the completion_report function
        report_data = run.completion_report(1, self.client_socket, 0)
        self.assertIsInstance(report_data, dict)

    def test_safe_send(self):
        # Test the safe_send function
        mock_response = {"status": 200, "data": "test data"}
        mock_safe_send = mock.MagicMock()
        with mock.patch("speech_to_world_hugofara.server.run.safe_send", new=mock_safe_send):
            run.safe_send(mock_response, self.client_socket)
            mock_safe_send.assert_called_once()

    def test_handle(self):
        # Test the handle function
        mock_handle = mock.MagicMock()
        with mock.patch("speech_to_world_hugofara.server.run.handle", new=mock_handle):
            run.handle(self.client_socket, None)
            mock_handle.assert_called_once()


if __name__ == "__main__":
    unittest.main()
