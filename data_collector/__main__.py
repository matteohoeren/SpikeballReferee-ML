import itertools
from utils import welcome, connect_to_arduino, print_hit, delete_last_line, remove_non_numbers, delete_multiple_lines
from datawarehouse import DataWarehouse
from datetime import datetime
from time import sleep
from pynput import keyboard # type: ignore
import threading
import plotly.graph_objects as go # type: ignore
from collections import deque
import time
import copy
import subprocess


DATA_LABELS = ["net", "rim", "discard", "rollup"]
DATA_DATE_FOLDER = "recorded_data/{}".format(datetime.now().strftime("%d-%m-%Y"))
TRIGGER_VALUE = 750
RECORD_LENGTH = 600
SAMPLES_BEFORE_TRIGGER = 10
PLAY_SOUNDS = False


input_index = ""
last_label = "net"
suspend_keyboard_listener = False

def on_press(key):
    global last_label, last_duration, last_samples, last_end_reason, warehouse_index, input_index, suspend_keyboard_listener
    #print(key)
    if key == keyboard.KeyCode.from_char('´'):
        suspend_keyboard_listener = not suspend_keyboard_listener
        print("Keyboard listener "+ str(suspend_keyboard_listener))

    if suspend_keyboard_listener:
        return


    if key == keyboard.KeyCode.from_char('r') or key == keyboard.Key.page_up:
        if PLAY_SOUNDS:
            subprocess.Popen("play -q ./sounds/wrong_quiet.mp3", shell=True)
        #print('Relabel to "rim"')   
        if(last_label == "rim"):
            return
        warehouse.move_last_dataset('rim')
        last_label = "rim"
        print_hit(warehouse_index-1, last_duration, last_samples, last_end_reason, last_label, True)
    if key == keyboard.KeyCode.from_char('n') or key == keyboard.Key.page_down:
        if PLAY_SOUNDS:
            subprocess.Popen("play -q ./sounds/net.wav", shell=True)
        #print('Relabel to "net"')   
        if(last_label == "net"):
            return
        warehouse.move_last_dataset('net')
        last_label = "net"
        print_hit(warehouse_index-1, last_duration, last_samples, last_end_reason, last_label, True)
    if key == keyboard.KeyCode.from_char('u'):
        #print('Relabel to "rollup"')   
        warehouse.move_last_dataset('rollup')
        last_label = "rollup"
        print_hit(warehouse_index-1, last_duration, last_samples, last_end_reason, last_label, True)
    if key == keyboard.KeyCode.from_char('d') or key == keyboard.KeyCode.from_char('.'):
        if PLAY_SOUNDS:
            subprocess.Popen("play -q ./sounds/del.wav", shell=True)
        if input_index != "":
            input_index = int(remove_non_numbers(input_index))
            if input_index > 0 and input_index < warehouse_index:
                delete_multiple_lines(warehouse_index - input_index + 1)
                for i in range(warehouse_index, input_index, -1):
                    printIndex = warehouse_index + input_index - i
                    warehouse.move_dataset_byid(i-1, 'discard')
                    print_hit(printIndex, last_duration, last_samples, last_end_reason, "discard", False)
                input_index = ""
        else:
            warehouse.move_last_dataset('discard')
            last_label = "discard"
            print_hit(warehouse_index-1, last_duration, last_samples, last_end_reason, last_label, True)
            input_index = ""

    if key == keyboard.KeyCode.from_char('p'):
        if input_index != "":
            input_index = int(remove_non_numbers(input_index))
            if input_index > 0 and input_index < warehouse_index:
                plotly = warehouse.get_dataset(input_index)
                if plotly is not None:
                    plot(plotly)
                input_index = ""
            else:
                print("Index {} out of range.".format(input_index))
                input_index = ""
    if key == keyboard.Key.backspace:
        delete_last_line()
    if key == keyboard.KeyCode.from_char('i'):
        input_index = ""
        print("Input index reset.")
    
    if key == keyboard.KeyCode.from_char('0'): input_index += "0"
    if key == keyboard.KeyCode.from_char('1'): input_index += "1"
    if key == keyboard.KeyCode.from_char('2'): input_index += "2"
    if key == keyboard.KeyCode.from_char('3'): input_index += "3"
    if key == keyboard.KeyCode.from_char('4'): input_index += "4"
    if key == keyboard.KeyCode.from_char('5'): input_index += "5"
    if key == keyboard.KeyCode.from_char('6'): input_index += "6"
    if key == keyboard.KeyCode.from_char('7'): input_index += "7"
    if key == keyboard.KeyCode.from_char('8'): input_index += "8"
    if key == keyboard.KeyCode.from_char('9'): input_index += "9"        
def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
def stop_listener(listener):
    with keyboard.Listener(on_press=on_press) as listener:
        listener.stop()


def plot(plot_data,ma_long, ma_short):
    trig = False
    if ma_long is None or ma_short is None:
        trig = False
    else:
        trig = True
    fig = go.Figure()

    timestamps, x_acc, y_acc, z_acc = zip(*plot_data)
    if trig is not False:
        fig.add_trace(go.Scatter(x=timestamps, y=ma_long, mode='lines', name='MA Long'))
        fig.add_trace(go.Scatter(x=timestamps, y=ma_short, mode='lines', name='MA Short'))

    # Plot x, y, and z accelerations
    fig.add_trace(go.Scatter(x=timestamps, y=x_acc, mode='lines', name='X Acceleration'))
    fig.add_trace(go.Scatter(x=timestamps, y=y_acc, mode='lines', name='Y Acceleration'))
    fig.add_trace(go.Scatter(x=timestamps, y=z_acc, mode='lines', name='Z Acceleration'))
        
    # Add labels and title
    fig.update_traces()
    fig.update_layout(title='Acceleration vs Timestamp '  + str(datetime.now().minute) + "m " + str(datetime.now().second) + "s, Hit Nr. " + str(warehouse_index), xaxis_title='Timestamp', yaxis_title='Acceleration')
    fig.show()


def calc_long_ma(data):
    sum = 0
    if len(data) > 0:
        for i in range(0, len(data)):
            sum += (abs(data[i][1]) + abs(data[i][2]) + abs(data[i][3]) / 3)
        sum = sum / len(data)
        if sum > TRIGGER_VALUE:
            return sum + 200
        else:
            return TRIGGER_VALUE
        
def calc_short_ma(data):
    timeframe = 10
    sum = 0
    if len(data) >= timeframe:
        for i in range(len(data)-timeframe, len(data)):
            sum += (abs(data[i][1]) + abs(data[i][2]) + abs(data[i][3]) / 3)
        sum = sum / len(data)
        return sum

def above_trigger(x,y,z, threshold=TRIGGER_VALUE):
    isTriggered = abs(x) > TRIGGER_VALUE or abs(y) > TRIGGER_VALUE or abs(z) > TRIGGER_VALUE
    return isTriggered

samplingdata = []
serial_data_queue = deque(maxlen=1000)
trigger_start = 0
time_elapsed = 0
def read_serial(ser):
    global serial_data_queue, trigger_start
    
    while True:
        line = ser.readline().decode("utf-8").strip()
        if line:
            lineArray = line.split(",")
            try:
                lineArray = [int(element) for element in lineArray]
            except ValueError:
                print("An element in lineArray could not be parsed to an integer. IMU might be in calibrating.")
            if len(lineArray) == 4:
                serial_data_queue.append(lineArray)
                evaluate_data_line(lineArray)

def find_position(timestamp):
    for i, sublist in enumerate(serial_data_queue):
        if sublist[0] == timestamp:
            return i
    print("Index not found. Timestamp: " + str(timestamp) + " " + str(len(serial_data_queue)))
    return -1

def normalize_timestamps(queue):
    deque_slice_copy = copy.deepcopy(queue)
    first_timestamp = deque_slice_copy[0][0]
    for i in range(len(deque_slice_copy)):
        deque_slice_copy[i][0] -= first_timestamp
    return deque_slice_copy

moving_average_temp = []
moving_average_temp_short = []
def evaluate_data_line(line):
    global trigger_start, time_elapsed
    try:
        timestamp = int(line[0])    
        x = int(line[1])
        y = int(line[2])
        z = int(line[3])
    except ValueError:  
        print("An element in line could not be parsed to an integer. Skipping...")
        return

    # This means that the trigger has been started and elapsed time is being calculated
    if trigger_start != 0:
        time_elapsed = timestamp - trigger_start

    if trigger_start == 0:
        # Start measurement
        if above_trigger(x,y,z):
            trigger_start = timestamp

    # When trigger was started AND within the RECORD_LENGTH
    if time_elapsed < RECORD_LENGTH and trigger_start != 0:

        slice_pos = find_position(trigger_start)
        if slice_pos < 0:
            print("Slice position is negative. Skipping ma calculation.")
        else:
            if len(serial_data_queue) - slice_pos > 5:
                slice = deque(itertools.islice(serial_data_queue,slice_pos , len(serial_data_queue)))
                moving_average_temp_short.append(calc_short_ma(slice))
                moving_average_temp.append(calc_long_ma(slice))
            else:   
                moving_average_temp.append(0)
                moving_average_temp_short.append(0)
        

    elif trigger_start != 0 and time_elapsed > RECORD_LENGTH:
        stop_and_save()

    if end_early(line) and trigger_start != 0:
        stop_and_save(timestamp)


def end_early(line):
    timestamp = int(line[0])
    x = int(line[1])
    y = int(line[2])
    z = int(line[3])

    if(len(moving_average_temp) > 0):
        exceed_long_ma = above_trigger(x,y,z, moving_average_temp[-1]+5000)
        if exceed_long_ma and time_elapsed > 300:
            #print("Second hit detected outside 300ms range. Stopping recording.")
            #print("Exceeded " + str(moving_average_temp[-1]) + "at time interval " + str(time_elapsed) + "ms")
            return False
    return False


warehouse_index = 0
def stop_and_save(ts=0):
    global trigger_start, warehouse_index, time_elapsed
    global last_label, last_duration, last_samples, last_end_reason

    if PLAY_SOUNDS:
        subprocess.Popen("play -q ./sounds/correct.mp3", shell=True)
  
    if ts == 0:
        pos = find_position(trigger_start) - SAMPLES_BEFORE_TRIGGER
        endpos = len(serial_data_queue)
    else:
        pos = find_position(trigger_start) - SAMPLES_BEFORE_TRIGGER
        endpos = len(serial_data_queue) - SAMPLES_BEFORE_TRIGGER + 1
    deque_slice = list(itertools.islice(serial_data_queue, pos, endpos))
    #print("Saved slice with start-ts " + str(deque_slice[0][0]) + "ms until " + str(deque_slice[-1][0]) + "ms to warehouse." + str(ts)) 
    norm_slice = normalize_timestamps(deque_slice)
    plot(norm_slice, moving_average_temp, moving_average_temp_short)
    warehouse.save_dataset(warehouse_index, norm_slice)
    print_hit(warehouse_index, time_elapsed, len(norm_slice), "ENDEARLY" if ts != 0 else "TIMEOUT", "net")
    last_label = "net"
    last_duration = time_elapsed
    last_samples = len(norm_slice)
    last_end_reason = "ENDEARLY" if ts != 0 else "TIMEOUT"
    warehouse_index += 1
    trigger_start = 0
    time_elapsed = 0
    moving_average_temp.clear()
    moving_average_temp_short.clear()
    
                        
if __name__ == "__main__":
    try:
        welcome()
        listener_thread = threading.Thread(target=start_listener)
        listener_thread.start()
        warehouse = DataWarehouse(DATA_DATE_FOLDER, DATA_LABELS)
        warehouse_index = warehouse.get_index()
        connection = connect_to_arduino()
        if connection is not None:
            read_serial(connection)
    except KeyboardInterrupt:
        print("###############################################################################")
        warehouse.save_warehouse()
        print("Bye bye!")
        try:
           stop_listener(listener_thread)
           listener_thread.join()
           exit(0)
        except KeyboardInterrupt:
           exit(0)

