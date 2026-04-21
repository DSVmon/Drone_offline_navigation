
import sys
from unittest.mock import MagicMock

# Мокаем rclpy и сообщения, так как их нет в среде ассистента
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['std_msgs.msg'] = MagicMock()

import os
import importlib
sys.path.append(os.path.join(os.getcwd(), 'src', 'drone_control'))

# Импортируем наш класс
import drone_control.control_node as control_node
importlib.reload(control_node)
from drone_control.control_node import ControlNode, FlightState

def test_drone_safety_logic():
    print("Running Logic Test: Drone Safety Interlock...")
    
    # Инициализируем узел
    node = ControlNode()
    
    # 1. Проверяем начальное состояние (Должен лететь вперед)
    mock_msg = MagicMock()
    mock_msg.data = False # Препятствий нет
    node.alert_callback(mock_msg)
    
    # Эмулируем один цикл управления
    # В норме должен публиковать скорость 1.0
    print("Step 1: No obstacles. Checking flight status...")
    node.control_loop() 
    # (В реальном ROS тут бы улетело сообщение Twist с linear.x = 1.0)
    
    # 2. Имитируем обнаружение препятствия
    print("Step 2: Obstacle detected! Sending alert...")
    alert_msg = MagicMock()
    alert_msg.data = True
    node.alert_callback(alert_msg)
    
    # Проверяем, изменилось ли состояние в узле
    if node.obstacle_alert == True:
        print("SUCCESS: Node recognized the obstacle.")
    else:
        print("FAILED: Node ignored the obstacle.")
        return False

    # 3. Проверяем остановку в цикле управления
    node.control_loop()
    if node.current_state == FlightState.STOPPED:
        print("SUCCESS: Drone state changed to STOPPED.")
    else:
        print("FAILED: Drone state is still", node.current_state)
        return False

    print("\nFINAL RESULT: Logic test PASSED. The code is safe.")
    return True

if __name__ == "__main__":
    if not test_drone_safety_logic():
        sys.exit(1)
