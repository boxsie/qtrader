import websocket
import json
import threading

class Client:
    def __init__(self, address, port):
        self._client = websocket.WebSocketApp(f'ws://{address}:{port}',
                                on_message = self._on_message,
                                on_error = self._on_error,
                                on_close = self._on_close)
        self._client.on_open = self._on_open

    def start(self):
        wst = threading.Thread(target=self._client.run_forever)
        wst.daemon = True
        wst.start()

    def send_message(self, msg):
        self._client.send(msg)

    async def _run_forever(self):
        await self._client.run_forever()

    def _on_message(self, msg):
        print(f'Received: {msg}')

    def _on_error(self, error):
        print(error)

    def _on_close(self):
        print('Python connection closed')

    def _on_open(self):
        print('Python client connected')
        self._client.send('Hello from Python')

if __name__ == '__main__':
    client = Client('localhost', 5000)
    asyncio.run(client.start())