import logging, os

# Ensure CASA is installed
from .check_imports import is_casa_installed

casa_enabled = is_casa_installed()
if is_casa_installed():
    from . import casaStuff

DefaultLevel = 'DEBUG'


class PipelineLogger(logging.getLoggerClass()):
    """A smart logging module which will redirect to CASA logger if in CASA otherwise to screen and file."""
    def __init__(self, name, level=None, logfile=None):
        self.name = name
        self.origin = ''
        self.screen_log_format = '[%(levelname).4s] [%(funcName)25s] %(message)s'
        self.file_log_format = '[%(asctime)-15s] [%(levelname)08s]  [%(name)s] [%(funcName)s] %(message)s'
        self.file_handler = None
        super(PipelineLogger, self).__init__(name)
        self.setup(name, level, logfile)
    
    def __enter__(self):
        #print('PipelineLogger.__enter__')
        return self
    
    def __exit__(self, type, value, traceback):
        #print('PipelineLogger.__exit__')
        if self.file_handler is not None:
            self.file_handler.close()
    
    def __del__(self):
        #print('PipelineLogger.__del__')
        if self.file_handler is not None:
            self.file_handler.close()

    def hasCasaLog(self):
        casa_enabled = is_casa_installed()
        return casa_enabled
        #return ('casalog' in globals())
    
    def setCasaOrigin(self):
        if self.hasCasaLog():
            self.origin = 'casa'
            casaStuff.casalog.origin(self.name)
    
    def restoreCasaOrigin(self):
        if self.hasCasaLog():
            casaStuff.casalog.origin(self.origin)
    
    def setup(self, name, level=None, logfile=None):
        self.name = name
        if not self.hasCasaLog():
            if level is None:
                level = DefaultLevel
            self.setLevel(logging.getLevelName(level))
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logging.Formatter(self.screen_log_format))
            self.addHandler(consoleHandler)
            if logfile is not None:
                fileHandler = logging.FileHandler(logfile, mode='a')
                fileHandler.setFormatter(logging.Formatter(self.file_log_format))
                self.addHandler(fileHandler)
                self.file_handler = fileHandler
    
    def debug(self, message):
        if self.hasCasaLog():
            self.setCasaOrigin()
            casaStuff.casalog.post(message, 'DEBUGGING')
            self.restoreCasaOrigin()
        else:
            super(PipelineLogger, self).debug(message)
    
    def info(self, message):
        if self.hasCasaLog():
            self.setCasaOrigin()
            casaStuff.casalog.post(message, 'NORMAL')
            self.restoreCasaOrigin()
        else:
            super(PipelineLogger, self).info(message)
    
    def warning(self, message):
        if self.hasCasaLog():
            self.setCasaOrigin()
            casaStuff.casalog.post(message, 'WARN')
            self.restoreCasaOrigin()
        else:
            super(PipelineLogger, self).warning(message)
    
    def error(self, message):
        if self.hasCasaLog():
            self.setCasaOrigin()
            casaStuff.casalog.post(message, 'SEVERE')
            self.restoreCasaOrigin()
        else:
            super(PipelineLogger, self).error(message)

    #def findCallerPy37(self, stack_info=False):
    #    """
    #    Find the stack frame of the caller so that we can note the source
    #    file name, line number and function name.
    #    
    #    See "lib/python3.7/logging/__init__.py".
    #    """
    #    currentframe = logging.currentframe # customizing here
    #    _srcfile = logging._srcfile # customizing here
    #    
    #    f = currentframe()
    #    #On some versions of IronPython, currentframe() returns None if
    #    #IronPython isn't run with -X:Frames.
    #    if f is not None:
    #        f = f.f_back
    #    if f is not None: # customizing here
    #        f = f.f_back # customizing here
    #    rv = "(unknown file)", 0, "(unknown function)", None
    #    while hasattr(f, "f_code"):
    #        co = f.f_code
    #        filename = os.path.normcase(co.co_filename)
    #        if filename == _srcfile:
    #            f = f.f_back
    #            continue
    #        sinfo = None
    #        if stack_info:
    #            sio = io.StringIO()
    #            sio.write('Stack (most recent call last):\n')
    #            traceback.print_stack(f, file=sio)
    #            sinfo = sio.getvalue()
    #            if sinfo[-1] == '\n':
    #                sinfo = sinfo[:-1]
    #            sio.close()
    #        rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
    #        break
    #    return rv

    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        
        See "lib/python3.9/logging/__init__.py".
        """
        currentframe = logging.currentframe # customizing here
        _srcfile = logging._srcfile # customizing here
        
        f = currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        if f is not None: # customizing here
            f = f.f_back # customizing here
        orig_f = f
        while f and stacklevel > 1:
            f = f.f_back
            stacklevel -= 1
        if not f:
            f = orig_f
        rv = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write('Stack (most recent call last):\n')
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv

