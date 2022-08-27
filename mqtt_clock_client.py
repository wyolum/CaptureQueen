import chess
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

group = 'capture_queen'

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("capture_queen.turn")
    client.subscribe("capture_queen.reset_pi")
    #publish.single(group + '.initial', "300", hostname="localhost")
    #publish.single(group + '.increment', "3", hostname="localhost")
    #publish.single(group + '.reset', "1", hostname="localhost")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    #print(msg.topic+" "+str(msg.payload))
    for cb in subscribers:
        cb(msg)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, keepalive=60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.

#publish.single(group + '.reset_clock', "", hostname="localhost")

def mqtt_start():
    client.loop_start()

subscribers = []
def mqtt_subscribe(callback):
    subscribers.append(callback)

def mqtt_publish_position(fen, lastmove_uci=None):
    payload = f'{lastmove_uci}//{fen}'
    publish.single(group + '.position', payload, hostname="localhost")


def mqtt_clock_pause(paused):
    paused = int(paused)
    publish.single(group + '.paused', paused,
                   hostname="localhost")
    
def mqtt_clock_reset(initial, increment):
    publish.single(group + '.initial_seconds', initial,
                   hostname="localhost")
    publish.single(group + '.increment_seconds', increment,
                   hostname="localhost")
    publish.single(group + '.reset', 1, hostname="localhost")
    print('mqtt_clock reset')
    
class MqttRenderer:
    def render(board, side, colors=None):
        '''
        Colors not used
        '''
        if board is None:
            board = chess.Board()
        fen = board.fen()
        if len(board.move_stack) > 0:
            lastmove_uci = board.move_stack[-1].uci()
        else:
            lastmove_uci = None
        mqtt_publish_position(fen, lastmove_uci)
    