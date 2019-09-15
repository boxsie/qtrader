import { spawn } from 'child_process';
import { Server } from './hub/Server';

const server = new Server('qtrader', 5000, 5001, 'dashboard');
server.start();

const qtraderPy = spawn('python', ['-u', 'qtrader/main.py']);

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

