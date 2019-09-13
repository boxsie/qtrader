const WebSocket = require('ws');

export class Server {
    constructor(name, port) {
        process.title = `${name}-hub`;
        this._port = port;
        this._server = null;
    }

    start() {
        this._server = new WebSocket.Server({ port: this._port });
        this._server.on('connection', (connection) => this._onHandshake(connection));
    }

    _onHandshake(connection) {
        connection.on('message', (msg) => this._onMessage(msg));
        connection.on('close', (connection) => this._onClose(connection));
        console.log(`Client connected`);
    }

    _onMessage(msg) {
        console.log(msg);
    }

    _onClose(connection) {
        console.log(`Client has disconnected`);
    }
}