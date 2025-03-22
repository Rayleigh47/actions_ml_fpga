import bluepy.btle as btle
import struct
import time
import csv
import os
from datetime import datetime
from bluepy.btle import Peripheral, BTLEDisconnectError
from crc import Calculator, Crc8

# MLP
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys

# Configuration
PLAYER_ID = 1
ACTION = "test"  
MAC_ADDR = "B4:99:4C:89:19:FC"  
CSV_FILE = f"{ACTION}_{PLAYER_ID}.csv"

MOTION_TIMEOUT = 1  # Time in seconds to detect end of motion

# BLE Constants
SERVICE_UUID = "0000dfb0-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "0000dfb1-0000-1000-8000-00805f9b34fb"
PACKET_SIZE = 20
CRC8 = Calculator(Crc8.CCITT)
SYN = b'S'
SYNACK = b'C'
ACK = b'A'
DATA_MOTION = b'M'
HANDSHAKE_TIMEOUT = 3

# Define the minimal MLP model (should match the training architecture)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_model(model_path, input_dim, num_classes=2):
    # Instantiate the model and load saved weights
    model = MLP(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, input_data):
    """
    Run inference on input_data (a list or numpy array of features)
    and return the predicted class.
    """
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data, dtype=np.float32)
    # Ensure the input has a batch dimension
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    with torch.no_grad():
        tensor_data = torch.from_numpy(input_data)
        outputs = model(tensor_data)
        _, predicted = torch.max(outputs, 1)
    return predicted.numpy()

# Buffer for IMU data
IMU_BUFFER = []
IMU_DATA_LIMIT = 40
last_packet_time = time.time()  

# Ensure CSV file has headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"])
    print(f"[LOG] Created {CSV_FILE} for recording IMU data.")

class IMUDelegate(btle.DefaultDelegate):
    """
    Handles incoming BLE packets, CRC verification, and buffering
    """
    def __init__(self):
        super().__init__()
        self.isRxPacketReady = False
        self.rxPacketBuffer = b""
        self.packetType = ""
        self.seqReceived = 0
        self.payload = b""

    def handleNotification(self, cHandle, data):
        """
        Processes incoming BLE IMU packets and verifies integrity
        """
        global last_packet_time
        self.isRxPacketReady = False
        self.rxPacketBuffer += data  

        if len(self.rxPacketBuffer) >= PACKET_SIZE:
            packet, crcReceived = struct.unpack(f"<{PACKET_SIZE - 1}sB", self.rxPacketBuffer[:PACKET_SIZE])

            if CRC8.verify(packet, crcReceived):  
                deviceID, self.packetType, self.seqReceived, self.payload = struct.unpack(f"BcB{PACKET_SIZE - 4}s", packet)
                self.packetType = self.packetType.decode("utf-8")
                self.isRxPacketReady = True
                last_packet_time = time.time()  # Reset motion timeout timer
                print(f"[BLE] Received: {self.packetType}, Seq: {self.seqReceived}, DeviceID: {deviceID}")

                self.rxPacketBuffer = self.rxPacketBuffer[PACKET_SIZE:]
            else:
                print("[BLE] Checksum failed! Discarding packet.")
                self.rxPacketBuffer = b""  

class BLEIMURecorder:
    """
    Handles BLE connection, handshaking, motion tracking, and IMU recording
    """
    def __init__(self, mac_addr):
        self.mac_addr = mac_addr
        self.device = None
        self.delegate = None
        self.isHandshakeRequired = True

    def connect(self):
        """
        Establishes BLE connection and performs handshake
        """
        print(f"[BLE] Connecting to {self.mac_addr}...")
        while True:
            try:
                self.device = Peripheral(self.mac_addr)
                self.delegate = IMUDelegate()
                self.device.setDelegate(self.delegate)
                service = self.device.getServiceByUUID(SERVICE_UUID)
                self.characteristic = service.getCharacteristics(CHAR_UUID)[0]
                print("[BLE] Connection established!")
                break
            except BTLEDisconnectError:
                print("[BLE] Connection failed, retrying...")

    def perform_handshake(self):
        """Performs 3-way handshake with the BLE device."""
        print("[BLE] Initiating handshake...")
        seq = 0

        self.send_packet(SYN, seq)
        start_time = time.time()

        while time.time() - start_time < HANDSHAKE_TIMEOUT:  
            try:
                if self.device.waitForNotifications(1.0) and self.delegate.isRxPacketReady:
                    if self.delegate.packetType == "C":  
                        print("[BLE] Received SYNACK, sending ACK...")
                        self.send_packet(ACK, self.delegate.seqReceived)
                        self.isHandshakeRequired = False
                        print("[BLE] Handshake successful!")
                        return True
            except BTLEDisconnectError:
                print("[BLE] Handshake failed due to disconnection!")
                return False
        
        print("[BLE] Handshake failed!")
        return False

    def send_packet(self, packet_type, seq):
        """
        Sends a BLE packet with CRC
        """
        packet = struct.pack("<BcB16s", PLAYER_ID, packet_type, seq, b'\x00' * (PACKET_SIZE - 4))
        crc_value = CRC8.checksum(packet)
        packet = struct.pack("<BcB16sB", PLAYER_ID, packet_type, seq, b'\x00' * (PACKET_SIZE - 4), crc_value)

        try:
            self.characteristic.write(packet)
            print(f"[BLE] Sent {packet_type.decode()} (Seq: {seq})")
        except BTLEDisconnectError:
            print("[BLE] Write failed! Connection lost.")

    def save_to_csv(self):
        """
        Writes buffered IMU data to CSV after motion ends
        """
        global IMU_BUFFER
        if len(IMU_BUFFER) > 0:
            with open(CSV_FILE, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(IMU_BUFFER)

            print(f"[LOG] {len(IMU_BUFFER)} IMU samples saved to {CSV_FILE}")
            # IMU_BUFFER.clear()
            IMU_BUFFER = IMU_BUFFER[IMU_DATA_LIMIT:]

    def extract_and_predict(self):
        global IMU_BUFFER
        # Extract the first 40 frames from the buffer
        chunk = IMU_BUFFER[:IMU_DATA_LIMIT]
        print(f"Extracting features from {len(chunk)} samples...")
        
        features = {}
        # ignore timestamp
        for col in range(1, 7):
            # Extract the data for the current sensor (column)
            col_data = [row[col] for row in chunk] # column by column i.e. [accX, accY, accZ, gyrX, gyrY, gyrZ]
            features[f'col{col}_mean'] = np.mean(col_data)
            features[f'col{col}_std'] = np.std(col_data)
            features[f'col{col}_min'] = np.min(col_data)
            features[f'col{col}_max'] = np.max(col_data)
            features[f'col{col}_range'] = np.max(col_data) - np.min(col_data)
            features[f'col{col}_median'] = np.median(col_data)
        
        # Build the input feature vector in a consistent order.
        # Order: col1_mean, col1_std, col1_min, col1_max, col1_range, col1_median, col2_mean, ... col6_median
        feature_vector = []
        for col in range(1, 7):
            for stat in ['mean', 'std', 'min', 'max', 'range', 'median']:
                feature_vector.append(features[f'col{col}_{stat}'])
        
        # Convert the feature vector to a numpy array (with a float32 dtype)
        input_features = np.array(feature_vector, dtype=np.float32)
        
        # Run inference (assuming 'model' and 'predict' are defined and accessible)
        prediction = predict(model, input_features)
        print(f"[INFERENCE] Predicted class: {prediction[0]}")

    def run(self):
        """
        Continuously listens for BLE notifications and detects motion events
        """
        self.connect()

        while self.isHandshakeRequired:
            self.perform_handshake()

        global last_packet_time

        while True:
            try:
                if self.device.waitForNotifications(1.0) and self.delegate.isRxPacketReady:
                    if self.delegate.packetType == "M":  
                        data_bytes = self.delegate.payload[:12]  
                        accX, accY, accZ, gyrX, gyrY, gyrZ = struct.unpack("<hhhhhh", data_bytes)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                        IMU_BUFFER.append([timestamp, accX, accY, accZ, gyrX, gyrY, gyrZ])
                        print(f"[BLE] Buffered IMU Data: {len(IMU_BUFFER)} samples collected")

                # If no new packet arrives for MOTION_TIMEOUT seconds, save data
                if len(IMU_BUFFER) >= IMU_DATA_LIMIT:
                    print("[BLE] Motion End Detected. Saving Data...")
                    self.extract_and_predict()
                    # self.save_to_csv()

            except BTLEDisconnectError:
                print("[BLE] Connection lost! Reconnecting...")
                self.connect()

# Run the BLE IMU data recorder
if __name__ == "__main__":
    # initialize MLP model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    model = load_model(model_path, input_dim=36, num_classes=10)

    imu_recorder = BLEIMURecorder(MAC_ADDR)
    imu_recorder.run()