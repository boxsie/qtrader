import WebSocket from 'ws';

export class NodeClient {
    constructor(address, port) {
        this._address = address;
        this._port = port;
        this._client = null;
    }

    connect() {
        this._client = new WebSocket(`ws://${this._address}:${this._port}/`);
        this._client.on('open', () => this._onHandshake());
        this._client.on('error', (error) => this._onError(error));
        this._client.on('close', () => this._onClose());
        this._client.on('message', (msg) => this._onMessage(msg));
    }

    _onHandshake() {
        console.log('JS client connected');
        this._client.send('Hello from JS');
    }

    _onMessage(msg) {
        if (msg.type === 'utf8') {
            console.log(`Received: '${msg.utf8Data}`);
        }
    }

    _onClose() {
        console.log('JS connection closed');
    }

    _onError(error) {
        console.log(`JS connection error: ${error}`);
    }
}