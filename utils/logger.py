import logging


# This Logger class simplifies the creation and configuration of logging instances
# to record relevant runtime information.
class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        # Create a logger instance, and set its log level and name
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)

        # Create two handlers: one for writing logs to a file, and one for outputting to the console
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()

        # Define log output format and apply it to both handlers
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

    """
    Example usage:
    This sets up logging to both the file 'app.log' and the console.
    'my_logger' is the name of the logger instance.

    logger = Logger('app.log', logging.INFO, 'my_logger').get_log()
    logger.info('This is an info message')
    logger.error('This is an error message')
    """
