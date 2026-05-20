import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
import time

def main():
    rclpy.init()
    node = Node('gz_service_diagnostic')
    
    service_name = '/gazebo/set_entity_state'
    node.get_logger().info(f'Starting diagnostic for service: {service_name}')
    
    # 1. Check if service exists in graph
    service_names_and_types = node.get_service_names_and_types()
    found = False
    for name, types in service_names_and_types:
        if name == service_name:
            node.get_logger().info(f'SUCCESS: Service {name} found in graph with types: {types}')
            found = True
            break
    
    if not found:
        node.get_logger().error(f'FAILED: Service {service_name} NOT found in ROS graph.')
        node.get_logger().info('Available services:')
        for name, _ in service_names_and_types:
            node.get_logger().info(f'  - {name}')
        return

    # 2. Try to create client and wait
    client = node.create_client(SetEntityState, service_name)
    node.get_logger().info('Waiting for service via client.wait_for_service()...')
    
    start_time = time.time()
    ready = client.wait_for_service(timeout_sec=5.0)
    
    if ready:
        node.get_logger().info(f'SUCCESS: Service is READY (waited {time.time() - start_time:.2f}s)')
        
        # 3. Test call (Get current state would be better, but let's try a safe set)
        node.get_logger().info('Attempting a dummy call to SetEntityState...')
        req = SetEntityState.Request()
        req.state.name = 'rescue_drone'
        # We won't actually move it, just send current (assume 1.0)
        req.state.pose.position.z = 1.0
        
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
        
        if future.done():
            res = future.result()
            if res.success:
                node.get_logger().info(f'SUCCESS: Service call returned success! Message: {res.status_message}')
            else:
                node.get_logger().error(f'FAILED: Service call returned error: {res.status_message}')
        else:
            node.get_logger().error('FAILED: Service call timed out.')
    else:
        node.get_logger().error('FAILED: Service wait_for_service() timed out after 5s.')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
