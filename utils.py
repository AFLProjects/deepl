import time

# Decorator to measure and print the execution time of a function
def time_perf(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.3e} seconds")
        return result
    return wrapper