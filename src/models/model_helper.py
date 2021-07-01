

class NERModelOutput(object):
    def __init__(
            self,
            loss=None,
            logits=None,
            hidden_states=None,
            attentions=None,
            ):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions=attentions

