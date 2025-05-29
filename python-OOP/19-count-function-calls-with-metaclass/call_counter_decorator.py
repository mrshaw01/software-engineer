def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)

    helper.calls = 0
    helper.__name__ = func.__name__
    return helper


# Manual usage example
if __name__ == "__main__":

    def greet():
        print("Hello")

    greet = call_counter(greet)
    print(greet.calls)  # Output: 0
    greet()
    greet()
    print(greet.calls)  # Output: 2
