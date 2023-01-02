'''Class representing packet which stores the starting position, current position,
destination node, and time steps sent alive'''
class Packet(object):
    def __init__(self, startPos, endPos, curPos, index, weight, time = 0, flag=0, times=0):
        self._startPos = startPos
        self._endPos = endPos
        self._curPos = curPos
        self._index = index
        self._weight = weight
        self._time = time
        self._flag = flag
        self._times = times
    def get_startPos(self):
        return self._startPos

    def get_endPos(self):
        return self._endPos

    def get_curPos(self):
        return self._curPos
        
    def get_index(self):
        return self._index

    def get_weight(self):
        return self._weight
        
    def get_time(self):
        return self._time

    def get_flag(self):
        return self._flag
        
    def set_startPos(self, startNode):
        self._startPos = startNode

    def set_endPos(self, endNode):
        self._endPos = endNode

    def set_curPos(self, curNode):
        self._curPos = curNode
    
    def set_index(self, index):
        self._index = index

    def set_weight(self, weight):
        self._weight = weight
        
    def set_time(self, time):
        self._time = time 

    def set_flag(self, flag):
        self._flag = flag

    def congestion_times(self, times):
        self._times = times

    def get_congestion_times(self):
        return self._times
'''Class which stores all the packets in the network'''
class Packets(object):
    def __init__(self, packetList):
        self.packetList = packetList
        self.num_Packets = len(packetList)