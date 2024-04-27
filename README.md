
# Executors and callbacks:

Two components control the execution of Callbacks: executors and Callback groups.

Executors are responsible for the actual execution of Callbacks.

Callback groups are used to enforce concurrency rules for Callbacks.

## You will find two types of executors:

MultiThreadedExecutor: Runs Callbacks in a pool of threads.

SingleThreadedExecutor: Runs Callbacks in the thread that calls Executor.spin()

A callback group controls when a callback can be executed.

## Callback groups:
A Callback Group controls when Callbacks are allowed to be executed. This class should not be instantiated. Instead, classes should extend it and implement `can_execute()`, `beginning_execution()` and `ending_execution()`. You can find different types of Callback Groups:

- **ReentrantCallbackGroup**: Allow Callbacks to be executed in parallel without restriction.
- **MutuallyExclusiveCallbackGroup**: Allow only one Callback to be executed at a time.

# An executor controls which threads callbacks get executed in.
