"""
A task tracker object keeps track of a completion of a task and send reports at regular intervals.
"""
import time


class TaskTracker:
    """Tracks a long task."""

    def __init__(self, socket, task_id, reporter):
        """
        Start a task tracker.

        :param socket: Client socket to send reports to.
        :param int task_id: ID of the task being tracked.
        :param Callable reporter: Function to call to send reports.
        """
        self.socket = socket
        self.task_id = task_id
        self.reporter = reporter
        self.progress = 0
        self.last_report_time = 0

    def sending_report(self):
        """Check if the completion report should be sent."""
        send_report = time.time() - self.last_report_time > 2
        if send_report:
            self.last_report_time = time.time()
        return send_report

    def callback(self, pipe, step_index, _tensor, tensor_callback):
        """Callback function to pass to a diffusion model."""
        self.progress = step_index * 100 // pipe.num_timesteps
        if self.sending_report():
            self.reporter(self.progress, self.socket, self.task_id)
        return tensor_callback

    def incomplete_callback(self, max_progress):
        """
        Emulates an "incomplete progress": when a task needs several diffusion models.

        :param int max_progress: Max progress that can be set by this task.
        :return Callable: Function to pass to the diffusion model.
        """
        initial_progress = self.progress

        def local_faker(pipe, step_index, _tensor, tensor_callback):
            """Callback function to pass to a diffusion model that fakes the completion status."""
            self.progress = initial_progress + step_index * max_progress // pipe.num_timesteps
            if self.sending_report():
                self.reporter(self.progress, self.socket, self.task_id)
            return tensor_callback

        return local_faker
