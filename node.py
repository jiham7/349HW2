class Node:
    def __init__(self):
        self.label = None #in training data: Y/N/?
        self.children = {}
        self.attribute = None
	# you may want to add additional fields here...
    def get_label(self):
        return self.label
    def get_children(self):
        return self.children
    def get_attribute(self):
        return self.attribute