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
        priceChartUpdateRate: 15,
        priceChartCurrentMins: 0,
        priceChartMaxTicks: 100
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
            this.priceChartCurrentMins++;
            if (this.priceChartCurrentMins % this.priceChartUpdateRate === 0) {
                this.priceChartCurrentMins = 0;
                this.priceChart.data.labels.push(this.priceChartUpdateRate);

                if (this.priceChart.data.labels.length > this.priceChartMaxTicks) {
                    this.priceChart.data.labels.splice(0, 1);
                }

                this.updateStats(data.stats);
                this.updateState(data.state, data.reward);

                this.priceChart.data.datasets.forEach((dataset) => {
                    if (dataset.data.length > this.priceChartMaxTicks) {
                        dataset.data.splice(0, 1);
                    }
                });

                this.priceChart.update();
            }
        },
        updateStats(stats) {
            this.stats = stats;
            if (this.stats.currentPosition) {
                if (this.stats.currentPosition.pos == 'buy') {
                    this.priceChart.data.datasets[0].data.push(this.stats.currentPosition.open);
                    this.priceChart.data.datasets[1].data.push(null);
                } else {
                    this.priceChart.data.datasets[0].data.push(null);
                    this.priceChart.data.datasets[1].data.push(this.stats.currentPosition.open);
                }
            } else {
                this.priceChart.data.datasets[0].data.push(null);
                this.priceChart.data.datasets[1].data.push(null);
            }

            this.priceChart.data.datasets[2].data.push(this.stats.currentPrice);
        },
        updateState(state, reward) {
            this.state = state;
            this.reward = reward;
            this.priceChart.data.datasets[3].data.push(this.state[2]);
            this.priceChart.data.datasets[4].data.push(this.state[3]);
            this.priceChart.data.datasets[5].data.push(this.reward);
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
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'Sell ($)',
                    backgroundColor: chartColours.red,
                    borderColor: chartColours.red,
                    borderWidth: 2,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'Price ($)',
                    backgroundColor: chartColours.yellow,
                    borderColor: chartColours.yellow,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'EMA 50 ($)',
                    backgroundColor: chartColours.blue,
                    borderColor: chartColours.blue,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'EMA 100 ($)',
                    backgroundColor: chartColours.purple,
                    borderColor: chartColours.purple,
                    borderWidth: 1,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-1'
                }, {
                    label: 'Reward',
                    backgroundColor: chartColours.grey,
                    borderColor: chartColours.grey,
                    borderWidth: 2,
                    data: [],
                    fill: false,
                    yAxisID: 'y-axis-2'
                }]
            },
            options: {
                spanGaps: false,
                scales: {
                    yAxes: [{
                        display: true,
                        position: 'left',
                        id: 'y-axis-1',
                        scaleLabel: {
                            display: true,
                            labelString: 'Value'
                        }
                    }, {
                        display: true,
                        position: 'right',
                        id: 'y-axis-2',
                        scaleLabel: {
                            display: true,
                            labelString: 'Value'
                        }
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