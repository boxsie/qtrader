import WebSocket from 'ws';
import http from 'http';
import fs from 'fs';
import url from 'url';
import path from 'path';

export class Server {
    constructor(name, httpPort, wsPort, indexPath) {
        process.title = `${name}-hub`;
        this._httpPort = httpPort;
        this._wsPort = wsPort;
        this._indexPath = indexPath;
        this._wsServer = null;
        this._httpServer = null;
    }

    start() {
        this._createHttpServer();
        this._createWebsocketServer();
    }

    _createHttpServer() {
        this._httpServer = http.createServer((request, response) => this._httpRequest(request, response)).listen(this._httpPort);

        console.log(`HTTP server running on port ${this._httpPort}`);
    }

    _createWebsocketServer() {
        this._wsServer = new WebSocket.Server({ port: this._wsPort, clientTracking: true });
        this._wsServer.on('connection', (connection) => this._onHandshake(connection));

        console.log(`Websocket server running on port ${this._wsPort}`);
    }

    _httpRequest(request, response) {
        var uri = url.parse(request.url).pathname;
        var filename = path.join(process.cwd(), this._indexPath, uri);

        if (fs.statSync(filename).isDirectory()) {
            filename += 'index.html';
        }

        console.log(`Request for: ${filename}`);

        fs.access(filename, fs.constants.F_OK, (err) => {
            if (err) {
                console.log(`${filename} does not exist`);
                response.writeHead(404, {"Content-Type": "text/plain"});
                response.write("404 Not Found\n");
                response.end();
                return;
            }

            fs.readFile(filename, "binary", function(err, file) {
                if(err) {
                    response.writeHead(500, {"Content-Type": "text/plain"});
                    response.write(err + "\n");
                    response.end();
                    return;
                }

                response.writeHead(200);
                response.write(file, "binary");
                response.end();
            });
        });
    }

    _onHandshake(connection) {
        connection.on('message', (msg) => this._onMessage(msg));
        connection.on('close', (code) => this._onClose());
        console.log(`Client connected`);
    }

    _onMessage(msg) {
        this._wsServer.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(msg);
              }
        })
    }

    _onClose() {
        console.log('Client has disconnected');
    }
}