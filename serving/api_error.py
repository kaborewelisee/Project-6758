class APIError(Exception):
    status_code = 400

    def __init__(self, message: str, status_code : int = None, payload=None):
        super().__init__()
        if status_code is not None:
            self.status_code = status_code
        self.message = message
        self.payload = payload

    def to_dict(self):
        return { 'message': self.message, 'error': self.payload }