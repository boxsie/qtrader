import 'jquery';
import 'bootstrap';
import Vue from 'vue';
import Chart from 'chart.js'

import { BrowserClient } from '../hub/BrowserClient';

const chartColours = {
    red: 'rgb(255, 99, 132)',
    orange: 'rgb(255, 159, 64)',
    yellow: 'rgb(255, 205, 86)',
    green: 'rgb(75, 192, 192)',
    blue: 'rgb(54, 162, 235)',
    purple: 'rgb(153, 102, 255)',
    grey: 'rgb(201, 203, 207)'
};

const vm = new Vue({
    el: '#main',
    data: {
        client: new BrowserClient('localhost', 5001),
        stats: {
            maxWin: 0,
            maxLoss: 0,
            avgProfitPct: 0,
            balance: 0,
            currentPrice: 0,
            currentPosition: {
                pos: '',
                open: 0,
                quantity: 0,
                pnlPct: 0
            }
        },
        state: [],
        reward: 0,
        priceChart: null,
        stateChart: null,
        chartUpdateRate: 1,
        chartCurrentMins: 0,
        chartMaxTicks: 100,
        trades: []
    },
    methods: {
        onMessage(msg) {
            var obj = JSON.parse(msg);
            var action = obj['action'];

            switch (action) {
                case 'update':
                    this.parseLatestUpdate(obj);
                    break;
                case 'command':
                    console.log(`Command '${data}' received`);
            }
        },
        parseLatestUpdate(data) {
            this.priceChart.data.labels.push(this.chartUpdateRate);
            this.stateChart.data.labels.push(this.chartUpdateRate);

            this.updateStats(data.stats);
            this.updateState(data.state, data.reward);

            if (this.priceChart.data.labels.length > this.chartMaxTicks) {
                this.priceChart.data.labels.splice(0, 1);
            }

            if (this.stateChart.data.labels.length > this.chartMaxTicks) {
                this.stateChart.data.labels.splice(0, 1);
            }

            this.priceChart.data.datasets.forEach((dataset) => {
                if (dataset.data.length > this.chartMaxTicks) {
                    dataset.data.splice(0, 1);
                }
            });

            this.stateChart.data.datasets.forEach((dataset) => {
                if (dataset.data.length > this.chartMaxTicks) {
                    dataset.data.splice(0, 1);
                }
            });

            this.priceChart.update();
            this.stateChart.update();
        },
        updateStats(stats) {
            var oldPos = this.stats.currentPosition;
            this.stats = stats;

            this.priceChart.data.datasets[2].data.push(this.stats.currentPrice);

            if (oldPos && !this.stats.currentPosition) {
                var trade = oldPos;
                trade.close = this.stats.currentPrice;
                trade.profit = this.calculateProfit(oldPos, this.stats.currentPrice)
                this.trades.unshift(trade)
                if (this.trades.length > 100) {
                    this.trades.pop();
                }
                this.stats.currentPosition = null;
            }

            var ds_buy = this.priceChart.data.datasets[0].data;
            var ds_sell = this.priceChart.data.datasets[1].data;

            if (this.stats.currentPosition) {
                if (this.stats.currentPosition.pos == 'buy') {
                    ds_buy.push(this.stats.currentPosition.open);
                    ds_sell.push(null);
                } else {
                    ds_buy.push(null);
                    ds_sell.push(this.stats.currentPosition.open);
                }
            } else {
                ds_buy.push(null);
                ds_sell.push(null);
            }
        },
        updateState(state, reward) {
            this.state = state;
            this.reward = reward;
            this.stateChart.data.datasets[0].data.push(this.state[2 + 13]);
            this.stateChart.data.datasets[1].data.push(this.state[2 + (2 * 13)]);
            this.stateChart.data.datasets[2].data.push(this.state[2 + (3 * 13)]);
            this.stateChart.data.datasets[3].data.push(this.state[2 + (4 * 13)]);
            this.stateChart.data.datasets[4].data.push(this.state[2 + (5 * 13)]);
            this.priceChart.data.datasets[3].data.push(this.reward);
        },
        calculateProfit(pos, price) {
            var profit = 0;
            if (pos.pos == 'buy') {
                profit = (pos.quantity * price) - (pos.quantity * pos.open)
            } else if (pos.pos == 'sell') {
                profit = (pos.quantity * pos.open) - (pos.quantity * price)
            }
            return profit - ((pos.quantity * pos.open) * 0.015)
        }
    },
    mounted() {
        this.client.connect(this.onMessage);
        this.priceChart = new Chart(this.$refs.priceChart, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Buy ($)',
                    backgroundColor: chartColours.green,
                    borderColor: chartColours.green,
                    borderWidth: 2,
                    data: [],
                    fill: false,
                    lineTension: 0,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'Sell ($)',
                    backgroundColor: chartColours.red,
                    borderColor: chartColours.red,
                    borderWidth: 2,
                    data: [],
                    fill: false,
                    lineTension: 0,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'Price ($)',
                    backgroundColor: chartColours.blue,
                    borderColor: chartColours.blue,
                    borderWidth: 2,
                    data: [],
                    fill: false,
                    lineTension: 0,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'Reward',
                    backgroundColor: chartColours.grey,
                    borderColor: chartColours.grey,
                    borderWidth: 2,
                    data: [],
                    fill: false,
                    lineTension: 0,
                    yAxisID: 'y-axis-2'
                }]
            },
            options: {
                scales: {
                    xAxes: [{
                        display: false,
                        ticks: {
                            display: false
                        }
                    }],
                    yAxes: [{
                        display: true,
                        position: 'left',
                        id: 'y-axis-1'
                    }, {
                        display: true,
                        position: 'right',
                        id: 'y-axis-2'
                    }]
                },
                elements: {
                    point: {
                        radius: 0
                    }
                }
            }
        });

        this.stateChart = new Chart(this.$refs.stateChart, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'RSI',
                    backgroundColor: chartColours.red,
                    borderColor: chartColours.red,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'ATR',
                    backgroundColor: chartColours.purple,
                    borderColor: chartColours.purple,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-2'
                }, {
                    label: 'AO',
                    backgroundColor: chartColours.green,
                    borderColor: chartColours.green,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'MFI',
                    backgroundColor: chartColours.orange,
                    borderColor: chartColours.orange,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'TSI',
                    backgroundColor: chartColours.blue,
                    borderColor: chartColours.blue,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }]
            },
            options: {
                spanGaps: false,
                scales: {
                    xAxes: [{
                        display: false,
                        ticks: {
                            display: false
                        }
                    }],
                    yAxes: [{
                        display: true,
                        position: 'left',
                        id: 'y-axis-1'
                    }, {
                        display: true,
                        position: 'right',
                        id: 'y-axis-2'
                    }]
                },
                elements: {
                    point: {
                        radius: 0
                    }
                }
            }
        });
    }
});