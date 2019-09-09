import requests
from signalr import Connection

class Connect:
    def start(self):
        with requests.Session() as sess:
            sess.verify = False
            self._connection = Connection("https://localhost:5001/qtrader", sess)
            self._hub = self._connection.register_hub('qtrader')
            self._connection.start()
            self._connection.error += print_error

    def sent(self, json):
        self._client.send(json)