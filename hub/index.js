import { spawn } from 'child_process';
import { Server } from './Server';
import { Client } from './Client';

const server = new Server('qtrader', 5000);
server.start();

const client = new Client('localhost', 5000);
client.connect();

const qtraderPy = spawn('python', ['-u', '../qtrader/main.py']);

var uint8arrayToString = (data) => {
    return String.fromCharCode.apply(null, data);
};

qtraderPy.stdout.on('data', (data) => {
    console.log(uint8arrayToString(data));
});

qtraderPy.stderr.on('data', (data) => {
    console.log(uint8arrayToString(data));
});

qtraderPy.on('exit', (code) => {
    console.log(`Process quit with code : ${code}`);
});

