import zmq

port = "5556"

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:{}".format(port))

socket.setsockopt(zmq.SUBSCRIBE, b"")

# Process 5 updates
total_value = 0
for update_nbr in range (5):
    string = socket.recv().decode()
    topic, messagedata = string.split()
    total_value += int(messagedata)
    print(topic, messagedata)

print("Average messagedata value for topic '{}' was {}F".format('ALL', total_value / update_nbr))