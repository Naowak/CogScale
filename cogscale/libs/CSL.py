import numpy as np

NO_ROLE = 0
ACTION = 1
OBJECT = 2
COLOR = 3
NB_ROLES = 4

class TwoSituationCSLDataset:
    """Class to create a dataset of sentences, their roles and predicates."""

    def __init__(self, objects, colors, positions):
        """Create a dataset of sentences of two situations, their roles and predicates.

        Args:
            objects (list): A list of objects.
            colors (list): A list of colors.
            positions (list): A list of positions.
        """
        # Define objects, colors and positions
        self.objects = objects
        self.colors = colors
        self.positions = positions

        # Create one situation datasets
        self.osb = OneSituationCSLDataset(objects, colors, positions)
        self.sentences, self.roles = self._combine_situation()
        self.predicates = [[p1, p2] for p1 in self.osb.predicates for p2 in self.osb.predicates]

        # Create one hot vectors & labels
        self.input_encoder = OneHotEncoder(self.sentences)
        self.output_encoder = self.osb.output_encoder
        self.X = np.array([self.input_encoder.encode(s) for s in self.sentences])
        self.Y = np.array([np.concatenate([y1, y2]) for y1 in self.osb.Y for y2 in self.osb.Y])


    def _combine_situation(self):
        """Combine two situations into one sentence and role list.

        Returns:
            tuple: A tuple containing a list of sentences and a list of roles.
        """
        combine = {'and': [NO_ROLE]}
        one_situation = {k: v for k, v in list(zip(self.osb.sentences, self.osb.roles))}
        two_situation = create_sentences(one_situation, combine, one_situation)
        sentences, roles = zip(*two_situation.items())
        return list(sentences), list(roles)



class OneSituationCSLDataset:
    """Class to create a dataset of sentences, their roles and predicates."""
    
    def __init__(self, objects, colors, positions):
        """Create a dataset of sentences of one situation, their roles and predicates.
        
        Args:
            objects (list): A list of objects.
            colors (list): A list of colors.
            positions (list): A list of positions.
        """
        # Define objects, colors and positions
        self.objects = objects
        self.colors = colors
        self.positions = positions
        self.others = ['this', 'that', 'is']

        # Create dataset
        self.sentences, self.roles = self._create_dataset()
        self.predicates = [Predicates(s, r, objects, colors, positions+self.others) for s, r in zip(self.sentences, self.roles)]

        # Create one hot vectors & labels
        self.input_encoder = OneHotEncoder(self.sentences)
        self.output_encoder = Labeler(objects, colors, positions, self.others)
        self.X = np.array([self.input_encoder.encode(s) for s in self.sentences])
        self.Y = np.array([self.output_encoder.encode(s, r) for s, r in zip(self.sentences, self.roles)])

    
    def _create_dataset(self):
        """Create a dataset of sentences and their roles.
        
        Returns:
            tuple: A tuple containing a list of sentences and a list of roles.
        """
        # Objects, Colors, Positions
        obj = create_dict_from_labels(self.objects, [OBJECT])
        col = create_dict_from_labels(self.colors, [COLOR])
        pos = create_dict_from_labels(self.positions, [ACTION])

        # Complementary words
        is_action = {'is': [ACTION]}
        is_norole = {'is': [NO_ROLE]}
        to_the = {'on the': [NO_ROLE, NO_ROLE]}
        this_is = {'this is': [ACTION, NO_ROLE], 'that is': [ACTION, NO_ROLE]}
        there_is = {'there is': [NO_ROLE, NO_ROLE]}
        det = {'a': [NO_ROLE], 'the': [NO_ROLE]}

        # Create sentences with corresponding roles
        a_color_object = create_sentences(det, {**col, '': []}, obj) # An (color) Object
        to_the_position = create_sentences(to_the, pos) # On the (position)
        one_situation = {
            **create_sentences(this_is, a_color_object), # This is a_color_object
            **create_sentences(det, obj, is_action, col), # An object is a color
            **create_sentences(det, obj, to_the_position, is_norole, col), # An object on the position is a color
            **create_sentences(a_color_object, is_norole, to_the_position), # a_color_object on the position
            **create_sentences(there_is, a_color_object, to_the_position), # There is a_color_object on the position
            **create_sentences(to_the_position, {**is_norole, **there_is}, a_color_object) # On the position there is a_color_object
        }

        # Return sentences and roles
        sentences, roles = zip(*one_situation.items())
        return list(sentences), list(roles)


class OneHotEncoder:
    """Class to encode and decode one hot vectors."""

    def __init__(self, sentences):
        """Create a one hot encoder from a list of sentences.
        
        Args:
            sentences (list): A list of sentences.
        """
        self.words = list(set(' '.join(sentences).split()))
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.vocab_size = len(self.words)
        self.max_length = max([len(s.split()) for s in sentences])
    
    def encode(self, sentence):
        """Encode a sentence into a one hot vector.
        
        Args:
            sentence (str): The sentence to encode.
        
        Returns:
            np.array: The one hot vector.
        """
        # Create matrix and fill with one hot vectors
        words = sentence.split()
        matrix = np.zeros((len(words), self.vocab_size))
        for i, word in enumerate(words):
            matrix[i, self.word2idx[word]] = 1

        # Padd start of sequence with zeros
        matrix = np.pad(matrix, ((self.max_length - len(words), 0), (0, 0)))

        return matrix


class Labeler:
    """Class to encode and decode labels."""

    def __init__(self, objects, colors, positions, others):
        """Create a labeler from a list of sentences and roles.
        
        Args:
            objects (list): A list of objects.
            colors (list): A list of colors.
            positions (list): A list of positions.
            others (list): A list of other words.
        """
        self.labels = objects + colors + positions + others
        self.idx2label = {i: l if isinstance(l, str) else l[0] for i, l in enumerate(self.labels)}
        self.objects2idx = create_dict_from_labels(objects)
        self.colors2idx = create_dict_from_labels(colors, base_index=len(objects))
        self.actions2idx = create_dict_from_labels(positions+others, base_index=len(objects)+len(colors))

    def encode(self, sentence, roles):
        """Encode a sentence of one situation and its roles into a labels vector.
        
        Args:
            sentence (str): The sentence of one situation to encode.
            roles (list): The roles of the sentence.
        
        Returns:
            np.array: The encoded label.
        """
        label = np.zeros(len(self.labels))
        for i, word in enumerate(sentence.split()):
            if roles[i] == OBJECT:
                label[self.objects2idx[word]] = 1 
            elif roles[i] == COLOR:
                label[self.colors2idx[word]] = 1
            elif roles[i] == ACTION:
                label[self.actions2idx[word]] = 1
        return label

    def decode(self, label):
        """Decode a labels vector from one or several situation into corresponding words.
        
        Args:
            label (np.array): The label to decode.
            
        Returns:
            str: The decoded label.
        """
        if label.shape[0] % len(self.labels) != 0:
            return 'Invalid label shape'
        
        nb_situations = label.shape[0] // len(self.labels)
        label = label.reshape(nb_situations, len(self.labels))
        return [' '.join([self.idx2label[i] for i in np.where(l == 1)[0]]) for l in label]


class Predicates:
    """Class to create a predicate from a sentence and its roles."""

    def __init__(self, sentence, roles, objects, colors, actions):
        """Create a predicate from a sentence and its roles.
        
        Args:
            sentence (str): The sentence to create the predicate from.
            roles (list): The roles of the sentence.
            objects (list): A list of objects.
            colors (list): A list of colors.
            positions (list): A list of positions.
        """
        # Define objects, colors and actions predicates
        obj_pred = create_dict_from_labels(objects, value='first')
        col_pred = create_dict_from_labels(colors, value='first')
        act_pred = create_dict_from_labels(actions, value='first')

        # Split sentence & prepare role list
        words = sentence.split(' ')
        found_roles = {x : None for x in [ACTION, OBJECT, COLOR]}
        self.is_invalid = False

        # Check if the sentence is valid
        for i, role  in enumerate(roles):
            # Check if the role is valid
            if role == NO_ROLE:
                continue
            if found_roles[role] is not None:
                self.is_invalid = True
                return

            # Set the role
            if role == ACTION:
                found_roles[role] = act_pred[words[i]]
            elif role == OBJECT:
                found_roles[role] = obj_pred[words[i]]
            elif role == COLOR:
                found_roles[role] = col_pred[words[i]]


        # Check if all roles are present
        if found_roles[ACTION] is None or found_roles[OBJECT] is None:
            self.is_invalid = True
            return
        
        # Set the roles
        self.action = found_roles[ACTION]
        self.object = found_roles[OBJECT]
        self.color = found_roles[COLOR]

    def __str__(self):
        """Return the predicate as a string."""
        if self.is_invalid:
            return 'INVALID'
        if self.color is None:
            return self.action + '(' + self.object + ')'
        return self.action + '(' + self.object + ', ' + self.color + ')'

    def __repr__(self):
        """Return the predicate as a string."""
        return self.__str__()



def create_sentences(*grammars):
    """Create sentences from a list of grammars.

    Args:
        grammars (list): A list of grammars to create sentences from.

    Returns:
        dict: A dictionary of sentences.
    """
    # If no grammars, return empty dictionary
    if not grammars:
        return {}

    # Create sentences
    sentences = grammars[0]
    for grammar in grammars[1:]:
        new_grammar = {}
        for s1, r1 in sentences.items():
            for s2, r2 in grammar.items():
                new_grammar[f"{s1} {s2}".strip()] = r1 + r2
        sentences = new_grammar
    
    return sentences

def create_dict_from_labels(labels, value=None, base_index=0):
    """Create a dictionary from a list of labels.
    
    Args:
        labels (list): A list of labels, can contain tuples. Each value in the tuple will be assigned the same value.
        value : If none, the value will be the index in the list. If 'first', the value will be the first element in the tuple, otherwise the value will be the one defined.
        base_index (int): The base index to start from.
    
    Returns:
        dict: A dictionary with labels as keys and value or index as values.
    """
    b = {}
    for i, item in enumerate(labels):
        if isinstance(item, tuple):
            for key in item:
                if value == None:
                    b[key] = base_index + i
                elif value == 'first':
                    b[key] = item[0]
                else:
                    b[key] = value 
        else:
            if value == None:
                b[item] = base_index + i
            elif value == 'first':
                b[item] = item
            else:
                b[item] = value
    return b



# Example usage
if __name__ == '__main__':
    objects = ['glass', 'orange', 'cup', 'bowl']
    colors = ['blue', 'orange', 'green', 'red']
    positions = ['left', 'right', ('center', 'middle')]

    dataset = TwoSituationCSLDataset(objects=objects, colors=colors, positions=positions)
    print(f'Shape of X: {dataset.X.shape}')
    print(f'Shape of Y: {dataset.Y.shape}')
    print()

    random_indices = np.random.choice(len(dataset.sentences), 5, replace=False)
    print("5 random sentences:")
    for i in random_indices:
        print(dataset.sentences[i])
        print(dataset.predicates[i])
        print()