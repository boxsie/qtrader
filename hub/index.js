import { Server } from './Server';
import { spawn } from 'child_process';

const server = new Server('qtrader', 5000)
server.start();

const qtraderPy = spawn('python', ['../qtrader/main.py']);

qtraderPy.stdout.on('data', (data) => {
    console.log(String.fromCharCode.apply(null, data));
});

qtraderPy.stderr.on('data', (data) => {
    console.log(String.fromCharCode.apply(null, data));
});