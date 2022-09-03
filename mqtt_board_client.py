'''
Used to display game on local mqtt network
'''
import chess
import chess.svg
import paho.mqtt.client as mqtt
import defaults

board = chess.Board()
board.push_uci("e2e4")
#pgr = pygame_render.PygameRender(640)
#pgr.render(board, False, colors=defaults.colors)
svg = chess.svg.board(board,size=640, flipped=False,
                      colors=defaults.colors)
open(".board.svg", 'w').write(svg)

class MqttMessageHandler:
    def __init__(self):
        self.board = chess.Board()
        self.colors = defaults.colors
        
    def on_connect(self, client, userdata=None, flags=None, rc=None):
        client.subscribe("capture_queen.reset")
        client.subscribe('capture_queen.position')
        print('connected')
        
    def on_message(self, client, userdata, msg):
        topic = msg.topic
        subtopic = topic.split('.')[1]
        print(topic, msg.payload)
        if subtopic == 'reset':
            self.board = chess.Board()
        if subtopic == 'position':
            payload = str(msg.payload)[2:-1]
            lastmove, fen = payload.split('//')
            if lastmove != 'None':
                lastmove = chess.Move(chess.parse_square(lastmove[:2]),
                                      chess.parse_square(lastmove[2:4]))
            else:
                lastmove = None
            board.set_fen(fen)
            svg = chess.svg.board(board,size=640, flipped=False,
                                  lastmove=lastmove,
                                  colors=self.colors)
            open(".board.svg", 'w').write(svg)
            print('wrote .board.svg')

def old_on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    if msg.topic == 'capture_queen.fen':
        fen = str(msg.payload)[2:-1]
        board = chess.Board(fen)
        svg = chess.svg.board(board,size=640, flipped=False,
                              colors=defaults.colors)
        open(".board.svg", 'w').write(svg)
        #pgr.render(board, False, colors=defaults.colors)

client = mqtt.Client()
handler = MqttMessageHandler()
client.on_connect = handler.on_connect
client.on_message = handler.on_message

client.connect("192.168.7.130", 1883, keepalive=60)
client.loop_forever()
