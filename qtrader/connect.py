import websocket
import json
import threading

class Connect:
    def __init__(self):
        websocket.enableTrace(True)
        self._ws = websocket.WebSocketApp("ws://127.0.0.1:5000",
                                on_message = self._ws_on_message,
                                on_error = self._ws_on_error,
                                on_close = self._ws_on_close)

        self._ws.on_open = self._ws_on_open
        self._wst = threading.Thread(target=self._ws.run_forever)
        self._wst.daemon = True
        self._wst.start()

    def send_message(self, json):
        # Reference: https://github.com/aspnet/SignalR/blob/release/2.2/specs/HubProtocol.md#invocation-message-encoding
        self._ws.send(self._encode_json({
            "type": 1,
            "target": "SendMessage",
            "arguments": ["qtrader", json]
        }))

    def _encode_json(self, obj):
        return json.dumps(obj) + chr(0x1E)

    def _ws_on_message(self, message):
        ignore_list = ['{"type":6}', '{}']
        # Split using record seperator, as records can be received as one message
        for msg in message.split(chr(0x1E)):
            if msg and msg not in ignore_list:
                print(f"From server: {msg}")

    def _ws_on_error(self, error):
        print(error)

    def _ws_on_close(self):
        print("### Disconnected from SignalR Server ###")

    def _ws_on_open(self):
        print("### Connected to SignalR Server via WebSocket ###")
        print("### Performing handshake request ###")
        self._ws.send(self._encode_json({
            "protocol": "json",
            "version": 1
        }))
        print("### Handshake request completed ###")