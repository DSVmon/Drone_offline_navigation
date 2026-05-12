
import sys
from unittest.mock import MagicMock

# Мокаем rclpy и сообщения
class DummyNode:
    def __init__(self, name):
        self.name = name
    def get_logger(self):
        return MagicMock()
    def create_subscription(self, *args, **kwargs):
        return MagicMock()
    def create_publisher(self, *args, **kwargs):
        return MagicMock()
    def create_timer(self, *args, **kwargs):
        return MagicMock()
    def get_clock(self):
        clock = MagicMock()
        clock.now.return_value.nanoseconds = 1000
        return clock

sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['rclpy.node'].Node = DummyNode
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()
sys.modules['nav_msgs.msg'] = MagicMock()

import os
import importlib
sys.path.append(os.path.join(os.getcwd(), 'src', 'drone_control'))

# Импортируем наш класс
import drone_control.control_node as control_node
importlib.reload(control_node)
from drone_control.control_node import ControlNode, State

def test_drone_safety_logic():
    print("Running Logic Test: Drone Physical Turn (Phase 4.1)...")
    
    # Инициализируем узел
    node = ControlNode()
    
    # 1. Проверяем начальное состояние (Должен лететь вперед)
    print("Step 1: No obstacles. Checking flight status...")
    node.control_loop()
    if node.state == State.MOVING:
        print("SUCCESS: Drone is in MOVING state.")
    else:
        print("FAILED: Drone state is", node.state)
        return False
    
    # 2. Имитируем обнаружение препятствия стереозрением
    print("Step 2: Stereo obstacle detected! Checking transition to TURNING...")
    alert_msg = MagicMock()
    alert_msg.data = True
    node.front_stereo_callback(alert_msg)
    
    node.control_loop()
    if node.state == State.TURNING:
        print("SUCCESS: Drone transitioned to TURNING state.")
    else:
        print("FAILED: Drone is still in", node.state)
        return False

    # 3. Имитируем завершение разворота
    print("Step 3: Simulating rotation completion...")
    # Устанавливаем текущий yaw равным целевому
    node.current_yaw = node.target_yaw
    
    node.control_loop()
    if node.state == State.MOVING:
        print("SUCCESS: Drone returned to MOVING state after turn.")
    else:
        print("FAILED: Drone failed to finish turn.")
        return False

    print("\nFINAL RESULT: Logic test PASSED. Physical turn behavior is correct.")
    return True

if __name__ == "__main__":
    if not test_drone_safety_logic():
        sys.exit(1)
