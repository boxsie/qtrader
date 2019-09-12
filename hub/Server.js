import { server as WebSocketServer } from 'websocket';
import { createServer } from 'http';

export class Server {
    constructor(name, port) {
        process.title = `${name}-hub`;
        this._port = port;
        this._clients = [];
        this._createServer();
    }

    start() {
        this._server.listen(this._port, () => {
            console.log(`Hub started on port ${this._port}`);
        });
    }

    _createServer() {
        this._server = createServer((request, response) => {});
        this._wsServer = new WebSocketServer({
            httpServer: this._server
        });
        this._wsServer.on('connection', this._onHandshake);
    }

    _onHandshake(ws, request) {
        var connection = request.accept(null, request.origin);
        connection.on('message', this._onMessage);
        connection.on('close', this._onClose);

        ws.id = this._clients.length;
        this._clients.push(ws);

        console.log(`Client@${request.origin} connected`)
    }

    _onMessage(msg) {
        console.log(msg);
    }

    _onClose(ws) {
        this._clients = this._clients.filter(item => item.id !== ws.id);
    }
}