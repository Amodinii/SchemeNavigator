'''
This file is used to create a custom exception in our project.
'''
import sys
class SchemeException(Exception):
    def __init__(self, error_message, error_details: sys):
        
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (
            f"Error occurred while running the python script, "
            f"in file [{self.file_name}], line [{self.lineno}]. \n"
            f"Error Message: [{self.error_message}]"
        )

