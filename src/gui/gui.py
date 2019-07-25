from PyQt5 import QtWidgets, QtCore
import qtmodern.styles
import qtmodern.windows

from mainwindow import Ui_MainWindow  # importing our generated file
 
import sys
 
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
 
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
 

class mywindow(QtWidgets.QMainWindow):
 
    def __init__(self):
 
        super(mywindow, self).__init__()
    
        self.ui = Ui_MainWindow()
    
        self.ui.setupUi(self)
    
app = QtWidgets.QApplication([])
qtmodern.styles.dark(app)

application = mywindow()

qtmodern.windows.ModernWindow(application).show()
    
sys.exit(app.exec())