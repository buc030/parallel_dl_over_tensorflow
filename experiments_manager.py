


import pickle
import os

#Singleton Not thread safe!!
class ExperimentsManager:
    BASE_PATH = '/tmp/generated_data/ExperimentsManager/'
    METADATA_FILE = BASE_PATH + 'metadata'
    inst = None

    def __init__(self):
        #self.running = None
        #self.metadata is a dict from Experiment to path under BASE_PATH.
        self.curr_experiment = None
        if not os.path.exists(os.path.dirname(ExperimentsManager.METADATA_FILE)):
            try:
                os.makedirs(os.path.dirname(ExperimentsManager.METADATA_FILE))
            except os.OSError as exc:  # Guard against race condition
                if exc.errno != os.errno.EEXIST:
                    raise

        self.metadata = {}
        self.refresh_metadata()

    #private

    #in case another process added experiments and took away free indexes!
    def refresh_metadata(self):
        try:
            with open(ExperimentsManager.METADATA_FILE, 'rb') as f:
                self.metadata.update(pickle.load(f))
        except:
            print ExperimentsManager.METADATA_FILE + ' does not exist yet, programmer should uncomment the line that creates it'
            self.dunp_metadata()


    def dunp_metadata(self):
        with open(ExperimentsManager.METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)

    def find_free_index(self):
        if len(self.metadata.values()) == 0:
            return 0

        return max(self.metadata.values()) + 1

    def lookup_experiment_path(self, experiment):
        if experiment not in self.metadata:
            return None
        return ExperimentsManager.BASE_PATH + str(self.metadata[experiment])

    #params is a dictionary from flag names to values
    def allocate_experiment(self, experiment):
        #assert that the experiment does not exists yet
        assert(self.lookup_experiment_path(experiment) == None)
        self.metadata[experiment] = self.find_free_index()
        self.dunp_metadata()
        return self.lookup_experiment_path(experiment)


    ######################   API   ############################:

    @classmethod
    def get(cls):
        if ExperimentsManager.inst is None:
            ExperimentsManager.inst = ExperimentsManager()
        return ExperimentsManager.inst

    def set_current_experiment(self, experiment):
        self.curr_experiment = experiment

    def get_current_experiment(self):
        return self.curr_experiment

    def dump_experiment(self, experiment):

        path = self.lookup_experiment_path(experiment)
        if path == None:
            path = self.allocate_experiment(experiment)

        with open(path, 'wb') as f:
            return pickle.dump(experiment, f)

    def load_experiment(self, experiment):
        path = self.lookup_experiment_path(experiment)
        if path is None:
            return None
        with open(path, 'rb') as f:
            res = pickle.load(f)
            experiment.results = res.results
            return res






