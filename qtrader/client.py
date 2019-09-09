from signalr import Connection

class SignalRClient:
    def start(self):
        self._connection = Connection("https://localhost:5000/qtrader", sess)
        self._hub = self._connection.register_hub('qtrader')
        self._connection.start()
        self._hub.client.on('newMessageReceived', print_received_message)
        self._hub.client.on('topicChanged', print_topic)
        self._connection.error += print_error

    def send(self, json):
        with connection:
            #post new message
            chat.server.invoke('send', 'Python is here')
            #change chat topic
            chat.server.invoke('setTopic', 'Welcome python!')
            #invoke server method that throws error
            chat.server.invoke('requestError')
            #post another message
            chat.server.invoke('send', 'Bye-bye!')
            #wait a second before exit
            connection.wait(1)

    def _print_received_message(self, data):
        print('received: ', data)

    def _print_topic(self, topic, user):
        print('topic: ', topic, user)

    def _print_error(self, error):
        print('error: ', error)