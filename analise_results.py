
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import experiment
import experiment_results
import experiments_manager
import sys
import PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout, QMenu
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui


Qt = QtCore.Qt

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return QtCore.QVariant(str(
                    self._data.values[index.row()][index.column()]))
        return QtCore.QVariant()



class App(QWidget):

    def __init__(self):
        super(App, self).__init__()
        self.title = 'Experiments viewer'
        self.left = 0
        self.top = 0
        self.width = 1200
        self.height = 800
        self.initUI()
        self.delete_safegaurd = None

        # refreshAction = QAction('Refresh', self)
        # refreshAction.triggered.connect(self.refresh)
        # self.toolbar = self.addToolBar('Refresh')
        # self.toolbar.addAction(refreshAction)


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createTable()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)

        # Show widget
        self.show()

    def update_from_df(self, df):
        self.tableWidget.clear()
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iget_value(i, j)))
                self.tableWidget.setItem(i, j, item)

        for i in range(len(df.index)):
            flagname = df.index[i]
            self.tableWidget.setVerticalHeaderItem(i, QTableWidgetItem(flagname))

        mgr = experiments_manager.ExperimentsManager()
        for j in range(len(df.columns)):
            widget = self.tableWidget.item(0, j)
            if widget is None:
                print 'No item at column ' + str(j)
                continue
            expr_id = self.get_experiment_id_by_widget_item(widget)
            expr = self.get_experiment_by_id(expr_id)
            #expr = mgr.load_experiment(expr)
            if len(expr.results) == 0:
                epochs_ran = 0
            else:
                epochs_ran = len(expr.results[0].trainError)
            item = QTableWidgetItem(str(epochs_ran))
            self.tableWidget.setItem(len(df.index), j, item)

        self.tableWidget.setVerticalHeaderItem(len(df.index), QTableWidgetItem('ran epochs'))

    def refresh(self):
        mgr = experiments_manager.ExperimentsManager()
        mgr.refresh_metadata()
        df = mgr.print_experiments()
        self.update_from_df(df)

    def createTable(self):
       # Create table
        mgr = experiments_manager.ExperimentsManager()
        df = mgr.print_experiments()

        self.tableWidget = QTableWidget(parent=self)


        self.tableWidget.setColumnCount(len(df.columns))
        self.tableWidget.setRowCount(len(df.index) + 1)

        self.update_from_df(df)
        self.tableWidget.move(0,0)


        # table selection change
        self.tableWidget.doubleClicked.connect(self.on_click)

    def get_experiment_by_id(self, id):
        mgr = experiments_manager.ExperimentsManager()
        return mgr.get_experiment_by_id(id)

    def get_experiment_id_by_widget_item(self, widgetItem):

        id_idx = None
        for i in range(1000):
            if self.tableWidget.verticalHeaderItem(i) is not None and self.tableWidget.verticalHeaderItem(i).data(0) == 'id':
                id_idx = int(i)
                break

        return int(self.tableWidget.item(id_idx, widgetItem.column()).data(0))

    def get_selected_experiment_ids(self):
        ids = []
        #first find the index of where the id is
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            id = self.get_experiment_id_by_widget_item(currentQTableWidgetItem)
            ids.append(int(id))
        return ids

    def delete_experiments(self, event):
        if self.delete_safegaurd is None:
            print 'Safegaurd, delete again to activate!'
            self.delete_safegaurd = 1
            return

        ids = self.get_selected_experiment_ids()
        mgr = experiments_manager.ExperimentsManager()
        print 'deleting ' + str(ids)
        for id in ids:
            e = self.get_experiment_by_id(id)
            mgr.delete_experiment(e)

        df = mgr.print_experiments()
        self.update_from_df(df)

    def contextMenuEvent(self, event):
        self.menu = QMenu(self)

        for plot_type in ['train', 'test', 'debug', 'train_and_test']:
            showAction = PyQt5.QtWidgets.QAction('Show ' + plot_type, self)
            showAction.triggered.connect(lambda e, plot_type=plot_type: self.display_selected_results(plot_type))
            self.menu.addAction(showAction)

        deleteAction = PyQt5.QtWidgets.QAction('Delete', self)
        deleteAction.triggered.connect(lambda: self.delete_experiments(event))
        self.menu.addAction(deleteAction)

        refreshAction = PyQt5.QtWidgets.QAction('Refresh', self)
        refreshAction.triggered.connect(lambda: self.refresh())
        self.menu.addAction(refreshAction)

        # add other required actions
        self.menu.popup(QtGui.QCursor.pos())

    def display_selected_results(self, plot_type='train'):

        experiments = {}
        ids = self.get_selected_experiment_ids()
        for id in ids:
            experiments[len(experiments)] = self.get_experiment_by_id(id)
            assert (experiments[len(experiments) - 1] is not None)


        loaded_experiments = {}
        i = 0
        for e in experiments.values():
            try:
                loaded_experiments[i] = experiments_manager.ExperimentsManager.get().load_experiment(e)
            except Exception,e:
                print 'Failed loading, try again...'
                print str(e)
                return
            if not loaded_experiments[i].has_data():
                print '###############################################'
                print 'No data for expr ' + str(loaded_experiments[i])
                print '###############################################'
                del loaded_experiments[i]
                continue
            i += 1

        comperator = experiment_results.ExperimentComperator(loaded_experiments)
        comperator.set_y_logscale(True)

        comperator.compare(group_by='b', error_type=plot_type)
        plt.show()


    @pyqtSlot()
    def on_click(self):
        self.display_selected_results()


app = QApplication(sys.argv)
ex = App()
sys.exit(app.exec_())


