from functools import wraps

GLOBAL_DEBUG_MODE = True

def set_debug_mode(is_debug: bool=True):
    """
    设置全局调试模式
    :param is_debug: 是否开启调试模式，类型为bool
    """
    global GLOBAL_DEBUG_MODE
    GLOBAL_DEBUG_MODE = is_debug

def Debug(func):
    """
    调试装饰器
    :param func: 需要被包装的函数对象
    :return: 包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if GLOBAL_DEBUG_MODE:
            print(f'[DEBUG] 正在执行函数: {func.__name__}')
        return func(*args, **kwargs)
    return wrapper
