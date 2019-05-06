import zmq
import random
import time

port = "5556"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:{}".format(port))

while True:
    topic = random.randrange(9999,10005)
    messagedata = random.randrange(1,215) - 80
    msg = "{} {}".format(topic, messagedata)
    print(msg)
    socket.send(msg.encode())
    time.sleep(0.1)