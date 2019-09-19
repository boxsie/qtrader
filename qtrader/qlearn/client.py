import websocket
import json
import threading

class Client:
    def __init__(self, address, port):
        self._is_connected = False
        self._client = websocket.WebSocketApp(f'ws://{address}:{port}',
                                on_message = self._on_message,
                                on_error = self._on_error,
                                on_close = self._on_close)
        self._client.on_open = self._on_open

    def start(self):
        try:
            wst = threading.Thread(target=self._client.run_forever)
            wst.daemon = True
            wst.start()
        except:
            print('Could not connect to ws server')

    def send_update(self, state, reward, stats):
        if not self._is_connected:
            return

        msg = json.dumps({
            'action': 'update',
            'state': state,
            'reward': reward,
            'stats': stats
        }, allow_nan=False)

        self._send_message(msg)

    def send_command(self, command):
        if not self._is_connected:
            return

        msg = json.dumps({
            'action': 'command',
            'command': command
        })
        self._send_message(msg)

    def _send_message(self, msg):
        self._client.send(msg)

    def _run_forever(self):
        self._client.run_forever()

    def _on_message(self, msg):
        obj = json.loads(msg)
        if obj['action'] == 'command':
            print(f"Command '{obj['command']}' received")

    def _on_error(self, error):
        print(error)

    def _on_close(self):
        self._is_connected = False
        print('Python connection closed')

    def _on_open(self):
        self._is_connected = True
        print('Python client connected')
        self._client.send('Hello from Python')
