# generate data for training
import csv


class DataPreprocessor:


    def get_xy(self,input):
        #take input from data in which the data is organized as inputs moves from move1-to move9 and last class to indicate game status
        board_state_dataX = []
        board_state_dataY  = []
  
        input = [x for x in input if x != '?']

        board = [0] * 9
        for i in range(len(input)): 
            if i % 2 == 0: 
                board[int(input[i])] = 1 
                board_state_dataX.append(board.copy()) 
            else: 
                newboard = [0]*9

                newboard[int(input[i])] = 1 
                board[int(input[i])] = -1 
                board.copy()
                board_state_dataY.append(newboard)

       
        return board_state_dataX,board_state_dataY

    def generate(self,filename):
        full_board_statesX = []
        full_board_statesY = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader) # skip the header row
            for row in reader:
                if(row[len(row)-1] != "draw"):
                    board_stateX,board_stateY = self.get_xy(row[0:len(row)-1]) #remove the last feature and get all the moves
                    
                    if(len(board_stateX) and  len(board_stateY) > 0):
                        if(len(board_stateX) == len(board_stateY)):
                            for states in board_stateX:
                                full_board_statesX.append(states)

                            for states in board_stateY:
                                full_board_statesY.append(states)
            
        return full_board_statesX, full_board_statesY     
            
        


