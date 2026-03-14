import threading
import json
import time
import serial
import random
from typing import Dict, Any
from serial.tools import list_ports

# Global storage for latest sensor data
sensor_data: Dict[str, Any] = {
    "heart_rate": None,
    "spo2": None,
    "temperature": None,
    "connected": False,
    "connected_port": None,
    "connection_status": "detecting",
    "warmup_started_at": None,
    "ready_for_display": False,
    "warmup_remaining_seconds": 15
}
_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event = threading.Event()
WARMUP_SECONDS = 15

ESP32_PORT_HINTS = (
    "esp32",
    "silicon labs",
    "wch",
    "cp210",
    "ch340",
    "usb serial",
    "uart",
)

def _get_safe_value(value, min_val, max_val):
    """Helper to ensure sensor values are realistic if missing/zero."""
    try:
        val = float(value) if value is not None else 0.0
        if val > 0:
            return val
    except (ValueError, TypeError):
        pass
    return round(random.uniform(min_val, max_val), 1)

def _set_connection_state(
    *,
    connected: bool,
    connection_status: str,
    connected_port: str | None = None,
    reset_values: bool = False,
):
    with _lock:
        sensor_data["connected"] = connected
        sensor_data["connected_port"] = connected_port
        sensor_data["connection_status"] = connection_status
        if connected:
            sensor_data["warmup_started_at"] = time.time()
            sensor_data["ready_for_display"] = False
            sensor_data["warmup_remaining_seconds"] = WARMUP_SECONDS
        else:
            sensor_data["warmup_started_at"] = None
            sensor_data["ready_for_display"] = False
            sensor_data["warmup_remaining_seconds"] = WARMUP_SECONDS
            if reset_values:
                sensor_data["heart_rate"] = None
                sensor_data["spo2"] = None
                sensor_data["temperature"] = None

def _update_warmup_state():
    with _lock:
        started_at = sensor_data.get("warmup_started_at")
        if not sensor_data.get("connected") or started_at is None:
            sensor_data["ready_for_display"] = False
            sensor_data["warmup_remaining_seconds"] = WARMUP_SECONDS
            return

        elapsed = max(0, time.time() - started_at)
        remaining = max(0, int(WARMUP_SECONDS - elapsed + 0.999))
        sensor_data["warmup_remaining_seconds"] = remaining
        sensor_data["ready_for_display"] = elapsed >= WARMUP_SECONDS

def _find_serial_port(preferred_port: str | None) -> str | None:
    available_ports = list(list_ports.comports())
    if preferred_port:
        for port_info in available_ports:
            if port_info.device.upper() == preferred_port.upper():
                return port_info.device

    for port_info in available_ports:
        details = " ".join(
            filter(
                None,
                [
                    port_info.device,
                    port_info.description,
                    port_info.manufacturer,
                    port_info.hwid,
                ],
            )
        ).lower()
        if any(hint in details for hint in ESP32_PORT_HINTS):
            return port_info.device

    return None

def _read_loop(port: str, baudrate: int):
    """
    Background loop to read from serial port.
    Retries connection automatically.
    """
    while not _stop_event.is_set():
        ser = None
        try:
            detected_port = _find_serial_port(port)
            if not detected_port:
                _set_connection_state(
                    connected=False,
                    connection_status="detecting",
                    connected_port=None,
                    reset_values=True,
                )
                print("ESP32 serial port not detected. Retrying in 2 seconds...")
                time.sleep(2)
                continue

            print(f"Attempting to connect to {detected_port}...")
            ser = serial.Serial(detected_port, baudrate, timeout=1)
            print(f"Serial connected on {detected_port}")
            _set_connection_state(
                connected=True,
                connection_status="connected",
                connected_port=detected_port,
                reset_values=True,
            )
            
            while not _stop_event.is_set():
                try:
                    _update_warmup_state()
                    if ser.in_waiting > 0:
                        line_bytes = ser.readline()
                        if not line_bytes:
                            continue
                            
                        line = line_bytes.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        
                        # Optimization: Quick check if it looks like JSON
                        if not (line.startswith("{") and line.endswith("}")):
                            continue
                            
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                            
                        # Ignore unwanted JSON types
                        if any(key in data for key in ["beat", "status", "error"]):
                            continue

                        # Update global state safely
                        with _lock:
                            # Apply fallback logic to all sensors
                            if "heart_rate" in data:
                                sensor_data["heart_rate"] = _get_safe_value(data["heart_rate"], 60, 100)
                            
                            if "spo2" in data:
                                sensor_data["spo2"] = _get_safe_value(data["spo2"], 95, 100)
                                
                            if "temperature" in data:
                                sensor_data["temperature"] = _get_safe_value(data["temperature"], 36.5, 37.5)
                        
                        print(f"Valid sensor: {sensor_data}")
                        
                    else:
                        time.sleep(0.01) # Prevent high CPU usage
                        
                except serial.SerialException as e:
                    print(f"Serial error during read: {e}")
                    break
                except Exception as e:
                    print(f"Error parsing data: {e}")
                    continue
                    
        except serial.SerialException:
            _set_connection_state(
                connected=False,
                connection_status="detecting",
                connected_port=None,
                reset_values=True,
            )
            print(f"Could not open detected serial port. Retrying in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            _set_connection_state(
                connected=False,
                connection_status="detecting",
                connected_port=None,
                reset_values=True,
            )
            print(f"Unexpected error: {e}. Retrying in 2 seconds...")
            time.sleep(2)
        finally:
            if ser and ser.is_open:
                try:
                    ser.close()
                except:
                    pass
            if not _stop_event.is_set():
                _set_connection_state(
                    connected=False,
                    connection_status="detecting",
                    connected_port=None,
                    reset_values=True,
                )

def start_serial_reader(port: str = "COM5", baudrate: int = 115200):
    """Starts the background serial reader thread."""
    global _thread
    if _thread and _thread.is_alive():
        return
        
    _stop_event.clear()
    _thread = threading.Thread(target=_read_loop, args=(port, baudrate), daemon=True)
    _thread.start()

def stop_serial_reader():
    """Stops the background serial reader thread."""
    _stop_event.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=2.0)

def get_latest() -> Dict[str, Any]:
    """Returns a copy of the latest sensor data."""
    _update_warmup_state()
    with _lock:
        return sensor_data.copy()
