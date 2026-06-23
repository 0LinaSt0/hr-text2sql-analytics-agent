import time
from functools import wraps


class NodeRegistry:
    _all_nodes = set()
    
    @classmethod
    def register_node(cls, node_name: str):
        cls._all_nodes.add(node_name)
    
    @classmethod
    def get_all_nodes(cls):
        return cls._all_nodes
    
    @classmethod
    def initialize_logs(cls, state):
        if 'execution_logs' not in state:
            state['execution_logs'] = {}
        
        for node_name in cls._all_nodes:
            if node_name not in state['execution_logs']:
                state['execution_logs'][node_name] = {
                    'node_name': node_name,
                    'total_executions': 0,
                    'total_duration_sec': 0.0,
                    'executions': []
                }


def log_node_execution(node_name: str = None):
    def decorator(func):
        actual_node_name = node_name or func.__name__
        NodeRegistry.register_node(actual_node_name)
        
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            if 'execution_logs' not in state:
                NodeRegistry.initialize_logs(state)
            
            node_log = state['execution_logs'].get(actual_node_name)
            if not node_log:
                node_log = {
                    'node_name': actual_node_name,
                    'total_executions': 0,
                    'total_duration_sec': 0.0,
                    'executions': []
                }
                state['execution_logs'][actual_node_name] = node_log
            
            execution_id = f"{actual_node_name}_{node_log['total_executions'] + 1}"
            start_time = time.time()
            execution_record = {
                'execution_id': execution_id,
                'duration_sec': None,
                'executed': False,
                'error': None
            }
            
            node_log['executions'].append(execution_record)
            current_idx = len(node_log['executions']) - 1
            
            try:
                result = func(state, *args, **kwargs)
                
                end_time = time.time()
                duration = round(end_time - start_time, 3)
                
                node_log['executions'][current_idx].update({
                    'duration_sec': duration,
                    'executed': True
                })
                
                node_log['total_executions'] += 1
                node_log['total_duration_sec'] = round(
                    node_log['total_duration_sec'] + duration, 3
                )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = round(end_time - start_time, 3)
                
                node_log['executions'][current_idx].update({
                    'duration_sec': duration,
                    'executed': False,
                    'error': str(e)
                })
                
                node_log['total_executions'] += 1
                raise
        
        return wrapper
    return decorator